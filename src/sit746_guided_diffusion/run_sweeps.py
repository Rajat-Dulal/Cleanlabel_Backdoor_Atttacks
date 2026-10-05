#!/usr/bin/env python
"""Parameter sweeps from Phase2_Extensive_Experimentation.ipynb (and the cross-architecture run).

    python src/sit746_guided_diffusion/run_sweeps.py --sweep t_start                 # 50/100/200/300
    python src/sit746_guided_diffusion/run_sweeps.py --sweep num_poisons             # 25/100/250/500 @ t_start=200
    python src/sit746_guided_diffusion/run_sweeps.py --sweep guidance                # 10/20/30/40  @ t_start=200
    python src/sit746_guided_diffusion/run_sweeps.py --sweep t_start --arch resnet   # cross-architecture victim
    python src/sit746_guided_diffusion/run_sweeps.py --sweep t_start --values 50 100 300 --guidance 100

The clean baseline victim is trained ONCE per (arch) and shared by all sweep points. Each sweep point uses a
different random target image / poison subset (the global RNG is not reset between points, as in the notebook).
Results: results/sweeps/<sweep>_<arch>.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion_poison import DiffusionPoisoner, generate_poison_set  # noqa: E402
from experiment import load_or_train_clean_victim, run_poisoned_experiment  # noqa: E402
from generate_poisons import get_surrogate  # noqa: E402
from shared_utils.data import DEFAULT_DATA_DIR, class_name_to_idx, get_cifar10  # noqa: E402
from shared_utils.seed import get_device, set_seed  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_VALUES = {"t_start": [50, 100, 200, 300], "num_poisons": [25, 100, 250, 500], "guidance": [10, 20, 30, 40]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", choices=list(DEFAULT_VALUES), required=True)
    ap.add_argument("--values", type=float, nargs="+", default=None, help="override the swept values")
    ap.add_argument("--arch", choices=["surrogate", "resnet"], default="surrogate")
    ap.add_argument("--base", default="cat")
    ap.add_argument("--target", default="dog")
    ap.add_argument("--num_poisons", type=int, default=25)
    ap.add_argument("--t_start", type=int, default=200, help="fixed t_start for the non-t_start sweeps")
    ap.add_argument("--guidance", type=float, default=20.0)
    ap.add_argument("--lpips_weight", type=float, default=0.15)
    ap.add_argument("--victim_epochs", type=int, default=8)
    ap.add_argument("--surrogate_epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    args = ap.parse_args()

    device = get_device()
    set_seed(args.seed)
    train_set = get_cifar10(args.data_dir, True, "diffusion")
    test_set = get_cifar10(args.data_dir, False, "diffusion")
    base, target = class_name_to_idx(args.base), class_name_to_idx(args.target)

    surrogate = get_surrogate(train_set, device, args.surrogate_epochs, REPO / "data" / "checkpoints" / "surrogate.pt")
    poisoner = DiffusionPoisoner(device)
    clean = load_or_train_clean_victim(train_set, device, args.arch, args.victim_epochs, 256,
                                       ckpt_path=REPO / "data" / "checkpoints" / f"victim_clean_{args.arch}.pt")

    values = args.values or DEFAULT_VALUES[args.sweep]
    out_dir = REPO / "results" / "sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)
    (REPO / "results" / "figures").mkdir(parents=True, exist_ok=True)
    rows = []
    for v in values:
        cfg = {"num_poisons": args.num_poisons, "t_start": args.t_start, "guidance": args.guidance}
        cfg[args.sweep] = int(v) if args.sweep != "guidance" else float(v)
        print(f"\n{'=' * 70}\nSWEEP {args.sweep} = {cfg[args.sweep]}  (arch={args.arch}, cfg={cfg})\n{'=' * 70}")
        poisons, target_img, target_idx = generate_poison_set(
            poisoner, surrogate, train_set, test_set, base, target,
            cfg["num_poisons"], cfg["t_start"], cfg["guidance"], args.lpips_weight)
        res, _ = run_poisoned_experiment(
            poisons, target_img, target_idx, train_set, test_set, clean, device, base, target,
            arch=args.arch, victim_epochs=args.victim_epochs,
            fig_prefix=str(REPO / "results" / "figures" / f"sweep_{args.sweep}_{cfg[args.sweep]}_{args.arch}"))
        rows.append({"config": cfg, **res})
        (out_dir / f"{args.sweep}_{args.arch}.json").write_text(json.dumps(rows, indent=2))  # incremental save
    print(f"\nSaved {out_dir / f'{args.sweep}_{args.arch}.json'}")


if __name__ == "__main__":
    main()
