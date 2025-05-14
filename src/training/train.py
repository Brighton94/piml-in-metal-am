"""Full-run trainer with train/val split, epoch-wise tqdm bar, metric logging."""

import argparse
import contextlib
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from datasets.multi_build import build_dataset_from_keys
from models.segformer_baseline import load_model


def train_one_epoch(model, loader, opt, scaler, device, desc):
    """Run a full optimisation epoch and return (loss, mIoU)."""

    model.train()
    inter = union = loss_sum = 0.0
    n = 0

    use_cuda = device.startswith("cuda")

    def autocast():
        return (
            torch.amp.autocast(device_type="cuda")
            if use_cuda
            else contextlib.nullcontext
        )

    for imgs, masks in tqdm(loader, desc=desc, leave=False):
        imgs, masks = imgs.to(device), masks.to(device)

        with autocast():
            out = model(pixel_values=imgs).logits
            logits = F.interpolate(
                out, size=masks.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, masks.float())

        opt.zero_grad(set_to_none=True)
        if use_cuda:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        preds = torch.sigmoid(logits) > 0.5
        inter += torch.logical_and(preds, masks).sum().item()
        union += torch.logical_or(preds, masks).sum().item()

        loss_sum += loss.item()
        n += 1

    return loss_sum / n, inter / (union + 1e-6)


@torch.no_grad()
def eval_one_epoch(model, loader, device, desc="val"):
    """Run a full evaluation epoch and return (loss, mIoU)."""

    model.eval()
    inter = union = loss_sum = 0.0
    n = 0
    for imgs, masks in tqdm(loader, desc=desc, leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        out = model(pixel_values=imgs).logits
        logits = F.interpolate(
            out, size=masks.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, masks.float())

        preds = torch.sigmoid(logits) > 0.5
        inter += torch.logical_and(preds, masks).sum().item()
        union += torch.logical_or(preds, masks).sum().item()

        loss_sum += loss.item()
        n += 1

    return loss_sum / n, inter / (union + 1e-6)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--keys",
        nargs="+",
        required=True,
        help="Dataset keys defined in config.DATASET_PATHS",
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="AdamW weight-decay (L2 regularisation)",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    # 1) dataset
    full_ds = build_dataset_from_keys(args.keys, size=512, augment=True)
    n_val = int(len(full_ds) * args.val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # 2) model
    _, model = load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,  # ← passes Sonar rule S6973
    )
    scaler = (
        torch.amp.GradScaler(device_type="cuda") if device.startswith("cuda") else None
    )

    # 3) training loop
    hist = {"train_loss": [], "train_iou": [], "val_loss": [], "val_iou": []}

    for ep in range(args.epochs):
        tr_loss, tr_iou = train_one_epoch(
            model, train_loader, opt, scaler, device, desc=f"train e{ep:02d}"
        )
        vl_loss, vl_iou = eval_one_epoch(
            model, val_loader, device, desc=f"val e{ep:02d}"
        )

        hist["train_loss"].append(tr_loss)
        hist["train_iou"].append(tr_iou)
        hist["val_loss"].append(vl_loss)
        hist["val_iou"].append(vl_iou)

        print(
            f"[{ep:02d}] "
            f"train loss={tr_loss:.4f}  IoU={tr_iou:.3f}  |  "
            f"val  loss={vl_loss:.4f}  IoU={vl_iou:.3f}"
        )

    # 4) plot
    epochs = range(args.epochs)
    plt.figure(figsize=(7, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist["train_loss"], label="train")
    plt.plot(epochs, hist["val_loss"], label="val")
    plt.title("BCE-logits loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist["train_iou"], label="train")
    plt.plot(epochs, hist["val_iou"], label="val")
    plt.title("mIoU")
    plt.legend()

    plt.tight_layout()
    out_png = "training_history.png"
    plt.savefig(out_png, dpi=120)
    print(f"\nTraining curves saved ➜  {os.path.abspath(out_png)}")


if __name__ == "__main__":
    main()
