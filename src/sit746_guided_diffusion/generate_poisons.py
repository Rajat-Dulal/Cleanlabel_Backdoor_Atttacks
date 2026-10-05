#!/usr/bin/env python
"""Generate diffusion-guided clean-label poisons (SIT746 / Phase 2).

    python src/sit746_guided_diffusion/generate_poisons.py --t_start 250 --guidance 100

Defaults mirror the notebooks (cat poisons -> dog target, 25 poisons, guidance 20, LPIPS 0.15, seed 101).
Writes data/poisons/<name>.pt (+ .json metadata, .png grid) and caches the surrogate in
data/checkpoints/surrogate.pt.  Requires network access for CIFAR-10 and google/ddpm-cifar10-32.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from diffusion_poison import (DiffusionPoisoner, compute_ssim_scores, generate_poison_set,  # noqa: E402
                              save_poisons)
from shared_utils.data import CIFAR10_CLASSES, DEFAULT_DATA_DIR, class_name_to_idx, get_cifar10  # noqa: E402
from shared_utils.models import SurrogateResNet  # noqa: E402
from shared_utils.plotting import save_poison_grid  # noqa: E402
from shared_utils.seed import get_device, set_seed  # noqa: E402
from shared_utils.training import train_surrogate  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def get_surrogate(train_set, device, epochs, ckpt_path):
    ckpt_path = Path(ckpt_path)
    if ckpt_path.exists():
        print(f"Loading cached surrogate: {ckpt_path}")
        m = SurrogateResNet().to(device)
        m.load_state_dict(torch.load(ckpt_path, map_location=device))
        return m.eval()
    m = train_surrogate(train_set, device, epochs=epochs)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), ckpt_path)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="cat", help="class the poisons are drawn from (keeps its label)")
    ap.add_argument("--target", default="dog", help="class whose test images should be misclassified as --base")
    ap.add_argument("--num_poisons", type=int, default=25)
    ap.add_argument("--t_start", type=int, default=250, help="partial-diffusion depth (0-1000)")
    ap.add_argument("--guidance", type=float, default=20.0, help="guidance scale g")
    ap.add_argument("--lpips_weight", type=float, default=0.15)
    ap.add_argument("--surrogate_epochs", type=int, default=15)
    ap.add_argument("--surrogate_ckpt", default=str(REPO / "data" / "checkpoints" / "surrogate.pt"))
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", default=None, help="output .pt path (default: data/poisons/<auto-name>.pt)")
    args = ap.parse_args()

    device = get_device()
    set_seed(args.seed)
    train_set = get_cifar10(args.data_dir, True, "diffusion")
    test_set = get_cifar10(args.data_dir, False, "diffusion")
    base_idx, target_idx_cls = class_name_to_idx(args.base), class_name_to_idx(args.target)

    surrogate = get_surrogate(train_set, device, args.surrogate_epochs, args.surrogate_ckpt)
    poisoner = DiffusionPoisoner(device)
    poisons, target_img, target_idx = generate_poison_set(
        poisoner, surrogate, train_set, test_set, base_idx, target_idx_cls,
        args.num_poisons, args.t_start, args.guidance, args.lpips_weight)

    ssims = compute_ssim_scores(poisons, train_set)
    meta = {**vars(args), "target_test_idx": target_idx,
            "ssim_mean": float(np.mean(ssims)) if ssims else None,
            "ssim_min": float(np.min(ssims)) if ssims else None}
    name = f"{args.base}2{args.target}_t{args.t_start}_g{args.guidance:g}_n{args.num_poisons}_s{args.seed}"
    out = Path(args.out) if args.out else REPO / "data" / "poisons" / f"{name}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_poisons(out, poisons, target_img, target_idx, meta)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    save_poison_grid(poisons, target_img, CIFAR10_CLASSES, out.with_suffix(".png"))
    print(f"Saved {len(poisons)} poisons -> {out}\nMean SSIM vs originals: {meta['ssim_mean']}")


if __name__ == "__main__":
    main()
