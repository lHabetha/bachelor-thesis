# Optimization Statistics: multifixed_penalty_direction_frontier_lam0_10_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 187 |
| Oracle confirmed | 187 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 13 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.4003 | 0.3894 |
| Oracle confirmed | 0.4103 | 0.4048 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.2568 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 64 | 64 | 0.2359 |
| nearest_splint_clearance | 134 | 123 | 123 | 0.4813 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 2.9550 |
| Median active coordinates | 3.0000 |
| Mean L1 distance | 0.6862 |
| Median L1 distance | 0.6685 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `leg_length` | 166 |
| `main_pin_length` | 165 |
| `overhang_span_y` | 123 |
| `main_hole_offset_from_open_end` | 74 |
| `splint_length` | 32 |
| `outer_span` | 31 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `grad_bottom_10` |
| Scenario kind | `gradient_ranked` |
| Starts with locks | 200 |
| Mean locked parameter count | 10.00 |
| Mean locked gradient mass | 0.3821 |
| Mean locked delta mass | 0.3625 |
| Paired baseline | `ch5d20_multifixed_penalty_lam0_10_tau0.75` |
| Baseline oracle OK | 167 |
| Constrained oracle OK | 187 |
| Oracle success drop | -20 |
| Recoverability | 98.2% |
| Mean distance delta | 0.0622 |
| Median distance delta | 0.0015 |
| Same-or-better distance successes | 54 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `cross_hole_distance_from_free_end` | 200 |
| `cross_hole_radius` | 200 |
| `depth` | 200 |
| `leg_length` | 31 |
| `main_hole_offset_from_open_end` | 123 |
| `main_hole_radius` | 200 |
| `main_pin_length` | 33 |
| `main_pin_radius` | 200 |
| `outer_span` | 169 |
| `overhang_span_y` | 77 |
| `splint_length` | 167 |
| `splint_radius` | 200 |
| `wall_thickness` | 200 |
