import src.config as config
from src.datasets.peregrine import PeregrineDataset
from torch.utils.data import ConcatDataset


def build_dataset_from_keys(keys, size=512, augment=True):
    """Keys : list[str]  e.g. ["tcr_phase1_build1", "tcr_phase1_build3"]."""

    parts = []
    for k in keys:
        p = config.get_dataset_path(k)
        if p is None:
            continue  # warn already printed in config
        parts.append(PeregrineDataset(p, size=size, augment=augment))

    if not parts:
        raise RuntimeError("No valid datasets resolved from keys:", keys)

    return ConcatDataset(parts)
