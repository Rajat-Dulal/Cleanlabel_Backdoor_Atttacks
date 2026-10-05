# Notebooks (original Kaggle/Colab notebooks, unchanged - outputs retained as evidence)

| Notebook | Purpose | Superseded by |
|---|---|---|
| `Phase1_Backdoor_FeatureInversion_via_PGD.ipynb` | SIT723 feature interpolation + PGD (Table I) | `src/sit723_feature_interpolation/` |
| `Phase2_Base_Code_Diifusion.ipynb` | First end-to-end guided-diffusion run (t_start=250) | `generate_poisons.py` + `train_target_cnn.py` |
| `Phase2_Backdoor_Diffusion_TransferLearning.ipynb` | Cross-architecture victim (`TransferResNet`, Table IV) | `train_target_cnn.py --arch resnet` |
| `Phase2_Extensive_Experimentation.ipynb` | Sweeps: t_start, #poisons, guidance (Table III) | `run_sweeps.py` |

Caution: they begin with Kaggle-specific shell cells (`!rm -rf /kaggle/working/*`, `!find /kaggle/working ... rm -rf`).
Harmless elsewhere (path doesn't exist) but **delete those cells before running on a machine where `/kaggle/working` matters**.
