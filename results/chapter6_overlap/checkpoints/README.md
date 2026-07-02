# Chapter 6 Checkpoints

Thesis-facing overlap surrogate checkpoints used for strict repair (§6.4) and
the alternating overlap/assemblability pipeline (§6.5).

---

## Checkpoints

| Directory | Size | Thesis role |
|-----------|------|-------------|
| `overlap_regressor_regression_v3_selected/` | ~190 KB | **Deployed single-head** regressor (256-128-64-32); holdout MAE_log=0.375 |
| `overlap_regressor_multitask_v1_selected/` | ~192 KB | Two-head volume+binary model; §6.4 multitask repair rows |

Each folder contains:

- `model_state.pt` — PyTorch weights
- `standardizer.npz` — input feature standardization
- `architecture.json` — hidden layer sizes, head layout
- `model_card.md` — holdout metrics and training summary

**v3 single-head SHA-256 (`model_state.pt`):** `1b561d32807eee33e032c2a064d6e19c69d28075c107478c7149b8d685d71326`

---

## Training data reference

Both checkpoints trained on 15,000 uniformly random labels from
`datasets/chapter6_overlap_clevis/labeled/acquired_random_15k.csv`.

Target: log-scaled normalized overlap, log1p(y / τ₀) with τ₀ = 5×10⁻⁵.
