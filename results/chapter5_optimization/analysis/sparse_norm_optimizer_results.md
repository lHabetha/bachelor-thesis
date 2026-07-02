# Sparse Norm Optimizer Results

Date: 2026-06-01

Benchmark: `blocked_200_v1`

Tau: `0.75`

## Purpose

This batch tests whether Chapter 5 should measure optimized design edits only by
normalized L2 distance, or whether sparse edit metrics are more thesis-relevant.
The sparse variants prefer changing fewer parameters, or smaller total absolute
normalized movement, while keeping the same trained MLP surrogate and the same
exact oracle reporting. No MLP was retrained.

Sparse metrics use normalized displacement

$$
\Delta_i = \frac{x_i - x_{0,i}}{\sigma_i}.
$$

The reported sparse metrics are

$$
L_1=\sum_i |\Delta_i|,\qquad
L_2=\sqrt{\sum_i \Delta_i^2},\qquad
L_0=\#\{i:|\Delta_i|>10^{-4}\}.
$$

## Variants

Three optimizer families were tested:

- `trust_region_hybrid_v1`;
- `coordinate_axis_bracket_v1`;
- `penalized_proximity_descent_v1`.

For each family, three sparse selection modes were run:

- `l1_l2_l0`: select valid `P >= tau` candidates by `(L0, L1, L2, step_idx)`;
- `l0_l2`: select by `(L0, L2, step_idx)`;
- `l1`: select by `(L1, L2, step_idx)`.

## Headline Results

| Run | Oracle OK | False OK | Mean L2 | Mean L1 | Mean L0 | Median L0 |
|-----|-----------|----------|---------|---------|---------|-----------|
| `trust_region_tau0.75` baseline | 197 | 3 | 0.5044 | 0.7717 | 5.34 | 1.0 |
| `trust_region_sparse_l1_l2_l0_tau0.75` | 197 | 3 | 0.5122 | 0.5122 | 0.995 | 1.0 |
| `trust_region_sparse_l0_l2_tau0.75` | 197 | 3 | 0.5122 | 0.5122 | 0.995 | 1.0 |
| `trust_region_sparse_l1_tau0.75` | 197 | 3 | 0.5091 | 0.5118 | 1.805 | 1.0 |
| `coordinate_axis_tau0.75` baseline | 197 | 3 | 0.5122 | 0.5122 | 0.995 | 1.0 |
| `coordinate_axis_sparse_l1_l2_l0_tau0.75` | 197 | 3 | 0.5122 | 0.5122 | 0.995 | 1.0 |
| `coordinate_axis_sparse_l0_l2_tau0.75` | 197 | 3 | 0.5122 | 0.5122 | 0.995 | 1.0 |
| `coordinate_axis_sparse_l1_tau0.75` | 197 | 3 | 0.5122 | 0.5122 | 0.995 | 1.0 |
| `penalized_proximity_tau0.75` baseline | 119 | 0 | 0.2232 | 0.6056 | 11.68 | 13.0 |
| `penalized_proximity_sparse_l1_l2_l0_tau0.75` | 149 | 0 | 0.2471 | 0.4948 | 5.410 | 4.0 |
| `penalized_proximity_sparse_l0_l2_tau0.75` | 150 | 1 | 0.2550 | 0.4875 | 4.750 | 3.0 |
| `penalized_proximity_sparse_l1_tau0.75` | 144 | 0 | 0.2383 | 0.5123 | 7.420 | 8.0 |

## Interpretation

The trust-region sparse variants show the clearest win for sparse scoring. The
baseline trust-region run had the same oracle-confirmed count as the sparse
variants, but its mean active-coordinate count was 5.34 and mean L1 distance was
0.7717. The strict `l1_l2_l0` and `l0_l2` sparse modes keep the same 197/200
oracle-confirmed result and the same three false successes, but reduce mean
active coordinates to 0.995 and mean L1 distance to 0.5122. The mean L2 distance
increases only slightly, from 0.5044 to 0.5122.

The coordinate-axis variants are an important sanity check. They are unchanged
because the original coordinate-axis optimizer is already sparse: each non-start
candidate changes exactly one coordinate. All three sparse modes reproduce the
baseline result exactly.

The penalized-proximity sparse variants are the most interesting scientifically.
The original proximity run was close in L2 but spread movement over many
coordinates: mean L0 11.68, median L0 13. Sparse penalized variants cut active
coordinates roughly in half or better and improve coverage:

- mixed `l1_l2_l0`: 149/200 oracle-confirmed, zero false successes, median L0 4;
- `l0_l2`: 150/200 oracle-confirmed, one false success, median L0 3;
- `l1`: 144/200 oracle-confirmed, zero false successes, median L0 8.

The cost is a modest L2 increase relative to the proximity baseline. This is a
good trade-off if the thesis values practical editability and interpretable
parameter changes over pure L2 minimality.

## Best Choices

Best reliable sparse-edit optimizer:

- `trust_region_sparse_l1_l2_l0_tau0.75` or
  `trust_region_sparse_l0_l2_tau0.75`;
- both keep 197/200 oracle-confirmed with three false successes;
- both reduce mean active coordinates from 5.34 to 0.995 compared with the
  trust-region baseline.

Best proximity-family sparse improvement:

- `penalized_proximity_sparse_l1_l2_l0_tau0.75`;
- it improves coverage from 119/200 to 149/200 while keeping zero false
  successes and reducing median active coordinates from 13 to 4.

Best sanity baseline:

- `coordinate_axis_sparse_*`;
- unchanged from coordinate axis, confirming that the sparse metric code behaves
  as expected when every candidate is already one-dimensional.

## Verification

All nine sparse runs passed:

- `verify_runs.py --skip-browser`;
- a custom sparse audit recomputing L0/L1/L2 from raw trajectories;
- wrapper-level oracle-leakage scan;
- browser/API smoke on `trust_region_sparse_l1_l2_l0_tau0.75`.

The sparse optimizer wrappers do not use oracle labels during optimization.
Oracle labels remain reporting-only through the standard Chapter 5 finalization
path.
