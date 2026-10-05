#!/usr/bin/env python
"""Reproduce the SIT723 baseline (Phase 1): feature interpolation + PGD inversion.

    python src/sit723_feature_interpolation/train_and_poison.py            # paper settings
    python src/sit723_feature_interpolation/train_and_poison.py --poison_rate 0.1 --lam 0.5

Paper settings (Table I): lam=0.35, eps=8/255, poison_rate=5% of base class (250 imgs), PGD 60 steps @ 0.02,
surrogate 15 epochs, victim 20 epochs.  Reference output: results/sit723_baseline.json
(CDA 0.8037, ASR 0.1450, mean SSIM 0.9993, 0/250 L-inf violations).
Runtime: not recorded in the original notebooks - note it here after your first run.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # makes `shared_utils` importable

import torch  # noqa: E402

import feature_interpolation as fi  # noqa: E402
from shared_utils.data import CIFAR10_CLASSES, DEFAULT_DATA_DIR, get_cifar10  # noqa: E402
from shared_utils.seed import get_device, set_seed  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target_class", type=int, default=3, help="class the surrogate prototype is built from (3=cat)")
    ap.add_argument("--base_class", type=int, default=5, help="class whose images are poisoned (5=dog)")
    ap.add_argument("--lam", type=float, default=0.35, help="interpolation coefficient lambda in [0,1]")
    ap.add_argument("--epsilon", type=float, default=8 / 255, help="L-inf budget in [0,1] pixel space")
    ap.add_argument("--poison_rate", type=float, default=0.05, help="fraction of base class to poison")
    ap.add_argument("--pgd_steps", type=int, default=60)
    ap.add_argument("--pgd_lr", type=float, default=0.02)
    ap.add_argument("--surrogate_epochs", type=int, default=15)
    ap.add_argument("--victim_epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out_dir", default=str(REPO / "results" / "sit723_run"))
    args = ap.parse_args()

    device = get_device()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | target={CIFAR10_CLASSES[args.target_class]} | base={CIFAR10_CLASSES[args.base_class]}")

    train_ds = get_cifar10(args.data_dir, train=True, normalization="cifar")
    test_ds = get_cifar10(args.data_dir, train=False, normalization="cifar")

    surrogate = fi.FeatureExtractor(10, pretrained=True)
    surrogate = fi.train_surrogate(surrogate, train_ds, args.target_class, args.surrogate_epochs, args.batch_size, device)
    ckpt_dir = REPO / "data" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(surrogate.state_dict(), ckpt_dir / "sit723_surrogate.pth")
    prototype = fi.extract_prototype(surrogate, train_ds, args.target_class, args.batch_size, device)

    poisoned_ds, stats = fi.build_poisoned_dataset(
        surrogate, train_ds, args.base_class, prototype, args.lam, args.epsilon,
        args.poison_rate, args.pgd_steps, args.pgd_lr, device)
    fi.visualize_poison_pairs(train_ds, poisoned_ds, out_dir)

    victim = fi.FeatureExtractor(10, pretrained=True)
    victim = fi.train_victim(victim, poisoned_ds, args.victim_epochs, args.batch_size, device)
    torch.save(victim.state_dict(), ckpt_dir / "sit723_victim_poisoned.pth")

    trigger = fi.extract_trigger(poisoned_ds, train_ds)
    cda, asr = fi.evaluate(victim, test_ds, trigger, args.base_class, args.target_class, args.batch_size, device)
    print(f"\n{'=' * 50}\n  Final CDA : {cda:.4f}\n  Final ASR : {asr:.4f}\n{'=' * 50}")

    result = {"target_class": args.target_class, "base_class": args.base_class, "lambda_interp": args.lam,
              "epsilon": args.epsilon, "poison_rate": args.poison_rate, "pgd_steps": args.pgd_steps,
              "pgd_lr": args.pgd_lr, "seed": args.seed, "cda": cda, "asr": asr, **stats}
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
