# Overlap Regressor — Chapter 6 selected (v3)

- Architecture: `e_256_128_64_32` hidden `(256, 128, 64, 32)`
- Selected by: lowest holdout `mae_log`
- Training labels: `15000` random-labelled rows from `acquired_random.csv`
- Target: `log1p(total_overlap_norm / 5e-05)`, SmoothL1 loss
- Holdout `mae_log`: `0.37502`, `mae_norm`: `0.000762759`
