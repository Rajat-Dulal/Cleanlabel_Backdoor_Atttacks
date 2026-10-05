# Reproducibility notes & known discrepancies

Written while moving the notebooks into this repo. Each item compares the **saved notebook outputs**
(see `results/*_notebook_outputs.json`) with what the **paper** (`paper/IEEE_Final.pdf`) states.
Resolve these before submitting the paper or tagging a final release.

## A. Numbers in the paper that the saved notebooks do not reproduce
| # | Paper says | Saved notebook output says | Action |
|---|-----------|----------------------------|--------|
| 1 | Table III, `t_start=100`: clean acc 72.89%, clean ASR 0.70%, poisoned ASR 30.80%, poison acc 78.74% (also the abstract's "0.70% -> 30.80%") | `Phase2_Extensive_Experimentation.ipynb`: clean acc 71.45%, poison acc 75.14%, poisoned ASR **1.80%** (+1.0 pp). No saved run produces 72.89 / 30.80. | Locate the original run, or re-run `run_sweeps.py --sweep t_start` and update Table III + abstract. |
| 2 | Table III caption: **guidance scale = 100** | The `t_start=50` and `t_start=300` rows in the paper match the notebook **exactly** (28.30%/78.34%, 45.00%/70.08%) but that notebook passes `guidance_scale=20.0`. | Check which value is right; fix the caption or the rows. |
| 3 | Table IV (cross-arch): `t_start = 100` | `Phase2_Backdoor_Diffusion_TransferLearning.ipynb` generates poisons with `t_start=250` (guidance default 20). The numbers (80.51 / 75.27 / 7.5 / 16.1) match the notebook. | Fix the caption/text (or re-run at t_start=100). |
| 4 | Sec. V-B: "clear stealth/effectiveness trade-off" as t_start grows | Full sweep in the notebook is non-monotonic: ASR 28.3 / 1.8 / 10.0 / 45.0 % for t_start = 50 / 100 / 200 / 300. | Soften the claim, or run multiple seeds/targets. |

## B. Statistical caveats (single seed / single run)
* Every sweep point draws a different random target image and poison subset and trains one victim for 8 epochs.
  Run-to-run noise is large: e.g. poisons 25 -> 26.3% ASR but 100 -> 0.7%, 250 -> 2.8%, 500 -> 20.5%
  (`phase2_extensive_notebook_outputs.json`). Report mean +- std over >= 5 seeds before drawing trends.
* "Poisoned-model accuracy above the clean baseline" (e.g. 78.7% vs 72.9%) is within the spread you get between
  two independent 8-epoch trainings of the same architecture, so it is not by itself evidence of the clean-label
  property.
* ASR is measured over all 1000 dog test images, although poisons are optimised against **one** target image
  (single-target feature collision). It is therefore a class-level confusion rate, not the single-target success
  rate used in Poison Frogs-style papers (the single-target check is also logged in each result JSON).

## C. Paper statements with no code in the uploaded notebooks
* **DiffPure-style purification module** (paper Sec. IV, "implemented but not run") - not in any notebook. `src/` has no purifier.
* **MNIST proof of concept** (`1aurent/ddpm-mnist`, Fig. 3) - no notebook uploaded.
* **ImageNet-pretrained victim negative result** (Sec. V-B-2) - no notebook uploaded.
* **LPIPS in Phase 1** - paper says LPIPS is used in the Phase 1 inversion step; the Phase 1 code uses only the L2
  embedding loss + L-inf projection.
* Phase 1 surrogate/victim are ImageNet-**pretrained** ResNet-18s (`pretrained=True`, 32x32 inputs). The paper's
  negative result says pretrained victims resist the Phase 2 attack; Phase 1 is a different attack/ASR definition,
  but this should be stated explicitly.

## D. Definitions that differ between phases (numbers are not directly comparable)
| | Phase 1 (SIT723) | Phase 2 (SIT746) |
|---|---|---|
| Poisoned class (keeps label) | dog (5) | cat (3) |
| "Target" | cat prototype | dog test images |
| ASR | 200 random dog test images **+ averaged fixed trigger** classified as cat | all 1000 dog test images (no trigger) classified as cat |
| Victim | ImageNet-pretrained ResNet-18, SGD, 20 ep | 5-conv CNN (or small ResNet), Adam, 8 ep, from scratch |
| Poison rate | 5% of dog class (250 imgs = 0.5% of data) | 25 imgs = 0.05% of data |
| Normalisation | CIFAR mean/std | [-1, 1] |

## E. Refactoring changes vs. the notebooks (behaviour-preserving unless noted)
* Class indices come from `dataset.targets` instead of iterating (and transforming) all 50k images - same indices, much faster.
* `make_poison`: removed a dead duplicate `total_loss` line; UNet set to `eval()` + `requires_grad_(False)` (gradient w.r.t. `x_t` unchanged).
* Phase 1: `random`/`numpy` are now seeded as well as torch (notebook seeded torch only), so exact numbers will differ from run to run of the old notebook.
* Phase 1 evaluation takes `base_class` as an argument instead of reading a global.
* Added (not in notebooks as runnable code): poison/SSIM saving in `generate_poisons.py` (adapted from a commented-out notebook cell), JSON result files, `run_sweeps.py`.
* `generate_poisons.py` defaults to guidance **20** (what the notebooks actually ran). To use the paper's stated value pass `--guidance 100`.
* **Not verified by execution**: the dev environment used to build this repo had no GPU, no internet to HuggingFace/CIFAR hosts,
  and could not install PyTorch, so the scripts were only syntax/static-checked. Run once on Colab/Kaggle and fix any runtime issues before tagging.
