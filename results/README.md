# Results

| File | What it is |
|---|---|
| `sit723_baseline.json` | Phase 1 baseline (CDA 0.8037, ASR 0.1450, SSIM 0.9993, 0/250 L-inf violations) - from the notebook output; matches paper Table I |
| `phase2_extensive_notebook_outputs.json` | Sweeps over t_start, #poisons, guidance scale **as actually recorded** in `Phase2_Extensive_Experimentation.ipynb` |
| `phase2_base_and_transfer_notebook_outputs.json` | Same-arch run and cross-arch (transfer) run from the other two Phase 2 notebooks |
| `paper_reported_tables.json` | Tables I, III, IV as printed in the paper, with a note where they disagree with the notebooks |
| `figures/` | Confusion matrices + poison grids extracted from the notebook outputs |
| `sweeps/`, `sit723_run/` | Written by `run_sweeps.py` / `train_and_poison.py` when you re-run them |

**Read `docs/reproducibility_notes.md` before citing any number** - at least one row of the paper's Table III
(t_start=100) is not reproduced by the saved notebooks.
