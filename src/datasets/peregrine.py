"""Lazy-loading Dataset for Peregrine L-PBF images + masks."""

import h5py
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class PeregrineDataset(Dataset):
    def __init__(self, h5_path, layers=None, size=512, augment=False):
        self.h5_path = h5_path  # only the path; real file opened lazily
        self.h5 = None  # per-worker handle
        self.layers = list(layers) if layers is not None else None

        tf = [T.Resize(size)]
        if augment:
            tf += [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
        tf += [T.ToTensor()]
        self.tf = T.Compose(tf)

    # lazy HDF5 open
    def _lazy_init(self):
        if self.h5 is None:  # runs once per worker
            self.h5 = h5py.File(self.h5_path, "r")
            self.imgs = self.h5["slices/camera_data/visible/0"]
            self.sp = self.h5["slices/segmentation_results/8"]  # spatter
            self.st = self.h5["slices/segmentation_results/3"]  # streaks

    # Dataset protocol
    def __len__(self):
        self._lazy_init()
        return len(self.layers) if self.layers is not None else len(self.imgs)

    def __getitem__(self, idx):
        self._lazy_init()
        i = self.layers[idx] if self.layers is not None else idx

        img = Image.fromarray(self.imgs[i]).convert("RGB")
        mask = (self.sp[i] | self.st[i]).astype(np.uint8)  # union

        img_t = self.tf(img)  # [3,H,W]
        mask_t = self.tf(Image.fromarray(mask)).squeeze(0)  # [H,W]

        return img_t, mask_t
