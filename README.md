# Clean-Label Poisoning of CNNs: From Feature Interpolation to Guided Diffusion Sampling

Handover repository for the SIT723 -> SIT746 research programme (author: Rajat Dulal, Deakin University).
Phase 1 builds clean-label poisons by **feature interpolation + PGD inversion**; Phase 2 replaces PGD with
**guided reverse diffusion** from a frozen pretrained DDPM (`google/ddpm-cifar10-32`).
Paper draft: [`paper/IEEE_Final.pdf`](paper/IEEE_Final.pdf).

> **Status: preliminary.** The purification-robustness comparison (DiffPure vs. the PGD baseline), which is the paper's
> central open question, has **not** been run. Before citing numbers, read
> [`docs/reproducibility_notes.md`](docs/reproducibility_notes.md) - some paper figures differ from the saved notebook outputs.

## 3. Repository structure
```
handover-repo/
├── README.md
├── requirements.txt
├── literature/                     # bibliography.bib, taxonomy + gap analysis
├── src/
│   ├── sit723_feature_interpolation/   # Phase 1: feature_interpolation.py, train_and_poison.py
│   ├── sit746_guided_diffusion/        # Phase 2: diffusion_poison.py, generate_poisons.py,
│   │                                   #          train_target_cnn.py, run_sweeps.py, experiment.py
│   └── shared_utils/                   # data, models, training, metrics (CDA/ASR/confusion), plotting, seed
├── data/                           # datasets / poisons / checkpoints (git-ignored; see data/README.md)
├── notebooks/                      # the 4 original Kaggle/Colab notebooks (with outputs)
├── results/                        # JSON metrics, confusion matrices & figures
├── docs/                           # research design, project plan, reproducibility notes
└── paper/                          # IEEE_Final.pdf
```

## 4. Artefact inventory
| # | Artefact | Location | Version / status |
|---|---|---|---|
| 1 | Literature review & gap analysis (feature collision / adversarial-generative / surrogate-based; diffusion as target vs. defence vs. generator) | `/literature/README.md` | Final for SIT746 scope |
| 2 | Annotated bibliography (BibTeX) | `/literature/bibliography.bib` | v1.0 - 3 entries still TODO (Narcissus, Witches' Brew, CleanSheet); annotations to add |
| 3 | Research design (aim, RQs, threat model) | `/docs/research_design.md` | Draft - confirm RQ wording, add threat-model diagram |
| 4 | SIT723 baseline: feature-interpolation poisoning | `/src/sit723_feature_interpolation` | tag `v723-final` |
| 5 | SIT746 artefact: guided reverse diffusion (feature-collision + LPIPS, SDEdit) | `/src/sit746_guided_diffusion` | tag `v746-preliminary` |
| 6 | Training / evaluation code (surrogate CNN, victim CNNs incl. ResNet-style transfer target, ASR / CDA / SSIM, confusion matrices) | `/src/shared_utils`, `/src/sit746_guided_diffusion/experiment.py` | commit: `_____` |
| 7 | Data: CIFAR-10, generated poisons, checkpoints | `/data` | See `/data/README.md` |
| 8 | Results & analysis: t_start / #poisons / guidance sweeps, cross-architecture, confusion matrices | `/results` | Preliminary |
| 9 | Project-management artefacts | `/docs/project_plan.md` | Template - fill in |
| 10 | Research paper / workshop draft (IEEE S&P workshops, CVPR workshops) | `/paper` | Draft: ___% |
| 11 | Presentation deck | `/docs/presentation.pptx` | Not yet added |

## 5. Rebuilding the project
**Environment** (Python 3.10+; a GPU is strongly recommended - the original work used a free-tier T4):
```bash
git clone <your-repo-url> && cd handover-repo
python -m venv .venv && source .venv/bin/activate        # or: conda create -n poison python=3.11
pip install -r requirements.txt
```
CIFAR-10 is auto-downloaded by `src/shared_utils/data.py`; the pretrained DDPM `google/ddpm-cifar10-32` is
fetched from the HuggingFace Hub by `diffusers` on first use (Phase 1 also downloads ImageNet ResNet-18 weights).

**SIT723 baseline (Phase 1)** - should land near `results/sit723_baseline.json` (CDA 0.8037 / ASR 0.1450 / SSIM 0.9993):
```bash
python src/sit723_feature_interpolation/train_and_poison.py          # -> results/sit723_run/
```

**SIT746 diffusion pipeline (Phase 2)**
```bash
# 1. generate 25 cat poisons aimed at dog test images
python src/sit746_guided_diffusion/generate_poisons.py --t_start 250 --guidance 100
# 2. inject, train clean + poisoned victims, evaluate (same-architecture victim)
python src/sit746_guided_diffusion/train_target_cnn.py --poisons data/poisons/cat2dog_t250_g100_n25_s101.pt
# 3. cross-architecture (transfer) victim: residual network
python src/sit746_guided_diffusion/train_target_cnn.py --poisons data/poisons/cat2dog_t250_g100_n25_s101.pt --arch resnet
```
**Sweeps** (the notebooks' settings; each point = one poison set + one victim, ~hours on a T4 for the full set):
```bash
python src/sit746_guided_diffusion/run_sweeps.py --sweep t_start          # 50/100/200/300
python src/sit746_guided_diffusion/run_sweeps.py --sweep num_poisons      # 25/100/250/500
python src/sit746_guided_diffusion/run_sweeps.py --sweep guidance         # 10/20/30/40
python src/sit746_guided_diffusion/run_sweeps.py --sweep t_start --arch resnet   # cross-architecture
```
> The notebooks that produced the recorded results used `guidance_scale=20` (the script default). The paper's Table III
> says 100. Pass `--guidance` explicitly and see `docs/reproducibility_notes.md`. Seeds default to 101 (Phase 2) / 42 (Phase 1).
> To run the original notebooks instead, open them in `notebooks/` on Kaggle/Colab.

Parameter descriptions are in each script's header and `--help`; hardware assumptions and the full design are in `docs/research_design.md`.


<!-- ## Not (yet) in this repo
DiffPure-style purification module and the purification-robustness comparison; the MNIST proof-of-concept;
the ImageNet-pretrained-victim experiment; Neural Cleanse / STRIP / ANP evaluation. (Described in the paper; no code was
available when this repo was assembled.) -->

