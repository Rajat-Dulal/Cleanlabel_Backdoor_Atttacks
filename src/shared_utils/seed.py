import random

import numpy as np
import torch


def set_seed(seed: int = 101) -> None:
    """Seed python, numpy and torch. (Phase 2 notebooks used 101, Phase 1 used 42.)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
