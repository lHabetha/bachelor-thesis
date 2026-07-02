# Optimization Statistics: multifixed_penalty_direction_frontier_lam0_10_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 166 |
| Oracle confirmed | 172 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 34 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.4457 | 0.3718 |
| Oracle confirmed | 0.5089 | 0.4285 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.0632 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 54 | 56 | 0.2177 |
| nearest_splint_clearance | 134 | 112 | 116 | 0.5581 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 6.4350 |
| Median active coordinates | 7.0000 |
| Mean L1 distance | 0.7754 |
| Median L1 distance | 0.6221 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `cross_hole_distance_from_free_end` | 186 |
| `main_pin_length` | 186 |
| `cross_hole_radius` | 185 |
| `main_pin_radius` | 185 |
| `splint_radius` | 185 |
| `splint_length` | 181 |
| `main_hole_radius` | 179 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `lock_main_bracket_user` |
| Scenario kind | `semantic_group` |
| Starts with locks | 200 |
| Mean locked parameter count | 6.00 |
| Mean locked gradient mass | 0.6086 |
| Mean locked delta mass | 0.5763 |
| Paired baseline | `ch5d20_multifixed_penalty_lam0_10_tau0.75` |
| Baseline oracle OK | 167 |
| Constrained oracle OK | 172 |
| Oracle success drop | -5 |
| Recoverability | 94.6% |
| Mean distance delta | 0.1076 |
| Median distance delta | 0.0264 |
| Same-or-better distance successes | 20 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `depth` | 200 |
| `leg_length` | 200 |
| `main_hole_offset_from_open_end` | 200 |
| `outer_span` | 200 |
| `overhang_span_y` | 200 |
| `wall_thickness` | 200 |
