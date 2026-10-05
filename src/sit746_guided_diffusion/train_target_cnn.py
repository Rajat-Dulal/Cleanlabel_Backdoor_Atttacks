#!/usr/bin/env python
"""Inject generated poisons, train clean + poisoned victims, and report CDA / ASR / confusion matrices.

    python src/sit746_guided_diffusion/train_target_cnn.py --poisons data/poisons/<file>.pt            # same-arch victim
    python src/sit746_guided_diffusion/train_target_cnn.py --poisons data/poisons/<file>.pt --arch resnet  # transfer victim

ASR (Phase 2 definition) = fraction of the 1000 held-out dog test images classified as cat.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from diffusion_poison import load_poisons  # noqa: E402
from experiment import load_or_train_clean_victim, run_poisoned_experiment  # noqa: E402
from shared_utils.data import DEFAULT_DATA_DIR, class_name_to_idx, get_cifar10  # noqa: E402
from shared_utils.seed import get_device, set_seed  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poisons", required=True, help="path to .pt written by generate_poisons.py")
    ap.add_argument("--arch", choices=["surrogate", "resnet"], default="surrogate",
                    help="victim architecture: 'surrogate' (same as guidance model) or 'resnet' (transfer)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out_dir", default=str(REPO / "results"))
    args = ap.parse_args()

    device = get_device()
    set_seed(args.seed)
    train_set = get_cifar10(args.data_dir, True, "diffusion")
    test_set = get_cifar10(args.data_dir, False, "diffusion")
    poisons, target_img, target_idx, meta = load_poisons(args.poisons)
    base, target = class_name_to_idx(meta["base"]), class_name_to_idx(meta["target"])

    clean = load_or_train_clean_victim(train_set, device, args.arch, args.epochs, args.batch_size,
                                       ckpt_path=REPO / "data" / "checkpoints" / f"victim_clean_{args.arch}.pt")
    tag = f"{Path(args.poisons).stem}_{args.arch}"
    out_dir = Path(args.out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    result, victim = run_poisoned_experiment(
        poisons, target_img, target_idx, train_set, test_set, clean, device, base, target,
        arch=args.arch, victim_epochs=args.epochs, batch_size=args.batch_size,
        fig_prefix=str(out_dir / "figures" / tag))
    result["poison_meta"] = meta
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2))
    torch.save(victim.state_dict(), REPO / "data" / "checkpoints" / f"victim_poisoned_{tag}.pt")
    print(f"Saved {out_dir / (tag + '.json')}")


if __name__ == "__main__":
    main()
