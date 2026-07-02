# Chapter 5 Results — Surrogate-Based Optimization

Frozen checkpoints and table-export summaries for Chapter 5. Full optimizer run
folders live with the viewer under `viewers/chapter5_clevis_optimization/data/`.

Thesis anchor: `tab:one-shot-gradient` through `tab:sparse-norm-selection`.

---

## Checkpoints (`checkpoints/`)

| Model ID | Size | Files |
|----------|------|-------|
| `row1_uncertainty_disagreement_B1000_T2500_best` | ~24 KB | `model.pt`, `standardizer.npz`, `model_card.json` |
| `hist_gradient_boosting_best_task30_labelblind_rank1` | ~668 KB | `model.joblib`, `model_card.json` |
| `xgboost_best_task30_labelblind_rank1` | ~256 KB | `model.joblib`, `model_card.json` |

Holdout BAC (MLP): 0.9926. See `checkpoints/README.md`.

---

## Frozen comparison summaries (`comparisons/`)

| File | Thesis tables |
|------|---------------|
| `optimizer_comparison.csv` | `tab:multifixed-penalty`, `tab:optimizer-headlines` |
| `optimizer_comparison.md` | Supporting |
| `task34_main_tau_sweep.csv` | `tab:adaptive-multistep` |
| `task34_penalty_sweep.csv` | `tab:adaptive-penalty-sweep` |
| `task34_comparison.md` | Supporting |
| `task34_subgroup_breakdown.csv` | Supporting |
| `task34_tradeoff_points.csv` | Supporting |
| `task34_sparse_norms.csv` | Supporting |

---

## Parameter lock study (`constraint_studies/`)

| File | Thesis tables |
|------|---------------|
| `thesis_ready_summary.csv` | `tab:semantic-locks` |
| `robustness_summary.md` | Supporting |

---

## Sparse norm analysis (`analysis/`)

| File | Thesis tables |
|------|---------------|
| `sparse_norm_optimizer_results.md` | `tab:sparse-norm-selection` |

---

## Tree control report (`tree_controls/`)

| File | Thesis tables |
|------|---------------|
| `optimizer_report_labelblind.csv` | `tab:tree-gradient-free-control` |

---

## Writable rerun outputs

```text
results/chapter5_optimization/runs/<run_id>/
```

---

## Viewer runs

Allowlisted run artifacts live under `viewers/chapter5_clevis_optimization/data/` (80 run folders, ~880 MB viewer JSON).
