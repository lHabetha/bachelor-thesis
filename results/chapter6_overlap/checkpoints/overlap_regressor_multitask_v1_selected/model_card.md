# Overlap Regressor — Chapter 6 #5e multitask (v1)

- Architecture: `multitask_256_128_64_32` hidden `(256, 128, 64, 32)` (shared trunk + two heads)
- Training labels: `15000` random-labelled rows from `acquired_random.csv`
- Joint loss: `L_vol (SmoothL1 on log1p(y/5e-05)) + lambda_bin * L_bin (BCEWithLogits)`
- Selected `lambda_bin`: `1.0` (by volume MAE_log parity; ties by binary AUC)

## Volume head (repair gradients / MAE_log)

- Holdout `mae_log`: `0.38345`, `mae_norm`: `0.000972519`
- Parity vs lambda_bin=0 volume-only control: `-0.01204` MAE_log
- Same role as the deployed single-head regressor: predicted overlap / log-overlap for repair descent.

## Binary head (P(overlap), repair stop)

- Binary label rule: `total_overlap_norm > 5e-05` (= tau0 clean threshold).
- Holdout binary F1: `0.9182`, AUC: `0.9724`, accuracy: `0.9180`, positive rate: `0.517`
- Repair stop criterion (#5d): declare "no overlap" when `P(overlap) < tau_bin`, sweep `tau_bin in {0.05, 0.1, 0.2}` (analogous to the Chapter 5 assemblability tau).

## Deployment note

- Loadable via `multitask_overlap_model.load_multitask_overlap_model`.
- `strict_repair` / `zigzag` defaults are NOT switched to this checkpoint yet; integration is tracked in #5d.
