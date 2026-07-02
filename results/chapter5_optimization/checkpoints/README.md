# Chapter 5 Checkpoints

Surrogate models used by the Chapter 5 optimizer workbench.

| Model ID | Files | Thesis use |
|----------|-------|------------|
| `row1_uncertainty_disagreement_B1000_T2500_best` | `model.pt`, `standardizer.npz`, `model_card.json` | Primary MLP surrogate (all §5.2–§5.4 MLP runs) |
| `hist_gradient_boosting_best_task30_labelblind_rank1` | `model.joblib`, `model_card.json` | Tree control: HGB coordinate + sphere-shrink |
| `xgboost_best_task30_labelblind_rank1` | `model.joblib`, `model_card.json` | Tree control: XGB probability-guided coordinate |

**Threshold τ:** runtime CLI parameter (`--tau`); not stored as a checkpoint file.

Each `model_card.json` documents training data, architecture, and holdout
metrics using the dataset paths of this repository.
