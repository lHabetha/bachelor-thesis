# Optimization Statistics: multifixed_penalty_direction_frontier_lam0_10_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 200 |
| Oracle confirmed | 200 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 0 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.4373 | 0.4482 |
| Oracle confirmed | 0.4373 | 0.4482 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.0000 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 66 | 66 | 0.2667 |
| nearest_splint_clearance | 134 | 134 | 134 | 0.5213 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 1.0000 |
| Median active coordinates | 1.0000 |
| Mean L1 distance | 0.4373 |
| Median L1 distance | 0.4482 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 113 |
| `leg_length` | 87 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `grad_bottom_12` |
| Scenario kind | `gradient_ranked` |
| Starts with locks | 200 |
| Mean locked parameter count | 12.00 |
| Mean locked gradient mass | 0.7406 |
| Mean locked delta mass | 0.7015 |
| Paired baseline | `ch5d20_multifixed_penalty_lam0_10_tau0.75` |
| Baseline oracle OK | 167 |
| Constrained oracle OK | 200 |
| Oracle success drop | -33 |
| Recoverability | 100.0% |
| Mean distance delta | 0.0991 |
| Median distance delta | 0.0199 |
| Same-or-better distance successes | 48 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `cross_hole_distance_from_free_end` | 200 |
| `cross_hole_radius` | 200 |
| `depth` | 200 |
| `leg_length` | 113 |
| `main_hole_offset_from_open_end` | 200 |
| `main_hole_radius` | 200 |
| `main_pin_length` | 87 |
| `main_pin_radius` | 200 |
| `outer_span` | 200 |
| `overhang_span_y` | 200 |
| `splint_length` | 200 |
| `splint_radius` | 200 |
| `wall_thickness` | 200 |
