# Optimization Statistics: multifixed_penalty_direction_frontier_lam0_10_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 171 |
| Oracle confirmed | 174 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 29 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.3466 | 0.3530 |
| Oracle confirmed | 0.3691 | 0.3914 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.2181 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 58 | 59 | 0.1995 |
| nearest_splint_clearance | 134 | 113 | 115 | 0.4191 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 9.3600 |
| Median active coordinates | 10.0000 |
| Mean L1 distance | 0.8728 |
| Median L1 distance | 0.8815 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 193 |
| `main_pin_radius` | 191 |
| `overhang_span_y` | 191 |
| `cross_hole_distance_from_free_end` | 190 |
| `leg_length` | 188 |
| `main_hole_offset_from_open_end` | 188 |
| `main_hole_radius` | 188 |
| `outer_span` | 187 |
| `wall_thickness` | 179 |
| `depth` | 177 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `lock_splint_user` |
| Scenario kind | `semantic_group` |
| Starts with locks | 200 |
| Mean locked parameter count | 3.00 |
| Mean locked gradient mass | 0.0824 |
| Mean locked delta mass | 0.0777 |
| Paired baseline | `ch5d20_multifixed_penalty_lam0_10_tau0.75` |
| Baseline oracle OK | 167 |
| Constrained oracle OK | 174 |
| Oracle success drop | -7 |
| Recoverability | 99.4% |
| Mean distance delta | 0.0085 |
| Median distance delta | 0.0002 |
| Same-or-better distance successes | 56 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `cross_hole_radius` | 200 |
| `splint_length` | 200 |
| `splint_radius` | 200 |
