# Optimization Statistics: multifixed_penalty_direction_frontier_lam0_10_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 151 |
| Oracle confirmed | 156 |
| False successes | 1 |
| False success rate | 0.7% |
| No crossing (stuck) | 49 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.3518 | 0.4167 |
| Oracle confirmed | 0.3906 | 0.4487 |
| False success | 0.4463 | — |
| Stuck / no crossing | 0.2122 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 46 | 48 | 0.2815 |
| nearest_splint_clearance | 134 | 105 | 108 | 0.3865 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 9.1800 |
| Median active coordinates | 10.0000 |
| Mean L1 distance | 0.8362 |
| Median L1 distance | 1.0169 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `leg_length` | 187 |
| `main_hole_offset_from_open_end` | 187 |
| `overhang_span_y` | 186 |
| `splint_radius` | 186 |
| `cross_hole_distance_from_free_end` | 185 |
| `wall_thickness` | 184 |
| `splint_length` | 183 |
| `outer_span` | 182 |
| `cross_hole_radius` | 179 |
| `depth` | 177 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `lock_pin_user` |
| Scenario kind | `semantic_group` |
| Starts with locks | 200 |
| Mean locked parameter count | 3.00 |
| Mean locked gradient mass | 0.2706 |
| Mean locked delta mass | 0.2547 |
| Paired baseline | `ch5d20_multifixed_penalty_lam0_10_tau0.75` |
| Baseline oracle OK | 167 |
| Constrained oracle OK | 156 |
| Oracle success drop | 11 |
| Recoverability | 93.4% |
| Mean distance delta | 0.0137 |
| Median distance delta | 0.0006 |
| Same-or-better distance successes | 57 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `main_hole_radius` | 200 |
| `main_pin_length` | 200 |
| `main_pin_radius` | 200 |
