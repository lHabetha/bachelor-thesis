# Optimization Statistics: multifixed_penalty_direction_frontier_lam0_10_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 199 |
| Oracle confirmed | 199 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 1 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.5224 | 0.5474 |
| Oracle confirmed | 0.5251 | 0.5576 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.0000 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 66 | 66 | 0.2894 |
| nearest_splint_clearance | 134 | 133 | 133 | 0.6372 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 1.9900 |
| Median active coordinates | 2.0000 |
| Mean L1 distance | 0.7373 |
| Median L1 distance | 0.7726 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 167 |
| `leg_length` | 157 |
| `overhang_span_y` | 42 |
| `main_hole_offset_from_open_end` | 32 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `grad_bottom_11` |
| Scenario kind | `gradient_ranked` |
| Starts with locks | 200 |
| Mean locked parameter count | 11.00 |
| Mean locked gradient mass | 0.5268 |
| Mean locked delta mass | 0.4998 |
| Paired baseline | `ch5d20_multifixed_penalty_lam0_10_tau0.75` |
| Baseline oracle OK | 167 |
| Constrained oracle OK | 199 |
| Oracle success drop | -32 |
| Recoverability | 100.0% |
| Mean distance delta | 0.1843 |
| Median distance delta | 0.1336 |
| Same-or-better distance successes | 5 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `cross_hole_distance_from_free_end` | 200 |
| `cross_hole_radius` | 200 |
| `depth` | 200 |
| `leg_length` | 42 |
| `main_hole_offset_from_open_end` | 167 |
| `main_hole_radius` | 200 |
| `main_pin_length` | 33 |
| `main_pin_radius` | 200 |
| `outer_span` | 200 |
| `overhang_span_y` | 158 |
| `splint_length` | 200 |
| `splint_radius` | 200 |
| `wall_thickness` | 200 |
