import os
import shutil
from math import floor

# adjust these paths to your project
BUILD = "tcr_phase1_build2"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "data", BUILD))
IMG_T = os.path.join(BASE, "images", "train")
LBL_T = os.path.join(BASE, "labels", "train")
IMG_V = os.path.join(BASE, "images", "val")
LBL_V = os.path.join(BASE, "labels", "val")

os.makedirs(IMG_V, exist_ok=True)
os.makedirs(LBL_V, exist_ok=True)

files = sorted(f for f in os.listdir(IMG_T) if f.endswith(".png"))
N = len(files)
n_val = floor(N * 0.20)

# take the *last* 20% for val
val_files = files[-n_val:]
train_files = files[:-n_val]  # the remaining 80%

# move them
for fname in val_files:
    shutil.move(os.path.join(IMG_T, fname), os.path.join(IMG_V, fname))
    shutil.move(os.path.join(LBL_T, fname), os.path.join(LBL_V, fname))

print(
    f"Split complete ⇒ train: {len(train_files)} images, val: {len(val_files)} images"
)
