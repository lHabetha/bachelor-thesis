# Optimization Statistics: adaptive_multistep_penalty_lam0_10_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 168 |
| Oracle confirmed | 171 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 32 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.3390 | 0.3471 |
| Oracle confirmed | 0.3607 | 0.3865 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.2256 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 57 | 58 | 0.1990 |
| nearest_splint_clearance | 134 | 111 | 113 | 0.4080 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 9.4200 |
| Median active coordinates | 10.0000 |
| Mean L1 distance | 0.8539 |
| Median L1 distance | 0.8729 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 194 |
| `main_pin_radius` | 192 |
| `overhang_span_y` | 192 |
| `cross_hole_distance_from_free_end` | 191 |
| `leg_length` | 189 |
| `main_hole_offset_from_open_end` | 189 |
| `main_hole_radius` | 189 |
| `outer_span` | 188 |
| `depth` | 180 |
| `wall_thickness` | 180 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `lock_splint_user` |
| Scenario kind | `semantic_group` |
| Starts with locks | 200 |
| Mean locked parameter count | 3.00 |
| Mean locked gradient mass | 0.0824 |
| Mean locked delta mass | 0.0796 |
| Paired baseline | `task34_adaptive_penalty_lam0_10_tau0.75` |
| Baseline oracle OK | 165 |
| Constrained oracle OK | 171 |
| Oracle success drop | -6 |
| Recoverability | 99.4% |
| Mean distance delta | 0.0091 |
| Median distance delta | 0.0003 |
| Same-or-better distance successes | 49 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `cross_hole_radius` | 200 |
| `splint_length` | 200 |
| `splint_radius` | 200 |
