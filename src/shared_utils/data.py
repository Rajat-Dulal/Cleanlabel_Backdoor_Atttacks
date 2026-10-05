"""CIFAR-10 loading helpers (auto-downloads into ./data by default).

Two normalisation conventions are used in this project:
  * "cifar"     - per-channel CIFAR-10 mean/std (Phase 1 / SIT723, ResNet-18 pipeline)
  * "diffusion" - mean=std=0.5, i.e. images in [-1, 1] (Phase 2 / SIT746, matches the DDPM)
"""
from pathlib import Path

import numpy as np
import torchvision
import torchvision.transforms as T

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

DEFAULT_DATA_DIR = str(Path(__file__).resolve().parents[2] / "data")


def get_cifar10(root: str = DEFAULT_DATA_DIR, train: bool = True, normalization: str = "diffusion"):
    if normalization == "cifar":
        norm = T.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    elif normalization == "diffusion":
        norm = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    else:
        raise ValueError(f"unknown normalization: {normalization}")
    return torchvision.datasets.CIFAR10(
        root=root, train=train, download=True, transform=T.Compose([T.ToTensor(), norm])
    )


def class_indices(dataset, class_id: int):
    """Indices of every sample with label == class_id (reads .targets, no image decoding)."""
    return np.where(np.array(dataset.targets) == class_id)[0].tolist()


def class_name_to_idx(name: str) -> int:
    return CIFAR10_CLASSES.index(name)


def denorm_diffusion(img):
    """[-1, 1] -> [0, 1] for plotting / SSIM."""
    return (img.clamp(-1, 1) + 1) / 2
