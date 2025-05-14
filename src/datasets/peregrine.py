"""Custom Dataset for Peregrine L-PBF images + masks."""

import h5py
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


class PeregrineDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, layers=None, transforms=None):
        self.h5 = h5py.File(h5_path, "r")
        self.imgs = self.h5["slices/camera_data/visible/0"]
        self.sp = self.h5["slices/segmentation_results/8"]
        self.st = self.h5["slices/segmentation_results/3"]  # streaking
        self.layers = range(len(self.imgs)) if layers is None else layers
        self.tf = transforms or T.Compose(
            [
                T.Resize(512),  # fits in 16×16 ViT patch grid
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        i = self.layers[idx]
        img = Image.fromarray(self.imgs[i])
        mask = (self.sp[i] | self.st[i]).astype(np.uint8)  # union → 0/1
        img, mask = self.tf(img), self.tf(Image.fromarray(mask))
        return img, mask.squeeze()

    def __len__(self):
        return len(self.layers)
