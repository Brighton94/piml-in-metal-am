import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from datasets.peregrine import PeregrineSpatterDataset
from models.segformer_baseline import load_model
from utils.metrics import BinaryIoU

H5_PATH = "data/peregrine.h5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 4

ds = PeregrineSpatterDataset(H5_PATH)
proc, model = load_model()
loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=4)

model.to(DEVICE)
opt = optim.AdamW(model.parameters(), lr=3e-4)
scaler = GradScaler()
metric = BinaryIoU().to(DEVICE)

for epoch in range(20):
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        with autocast():
            logits = model(pixel_values=imgs).logits.squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, masks.float())
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        metric.update(torch.sigmoid(logits), masks)
    print(f"Epoch {epoch:02d}  mIoU = {metric.compute():.3f}")
    metric.reset()
