# Optimization Statistics: random_sphere_coordinate_shrink_128_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 167 |
| Oracle confirmed | 174 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 33 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.6106 | 0.5000 |
| Oracle confirmed | 0.5502 | 0.4443 |
| False success | 0.0000 | — |
| Stuck / no crossing | 1.1030 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 63 | 64 | 0.3196 |
| nearest_splint_clearance | 134 | 104 | 110 | 0.7540 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 3.8700 |
| Median active coordinates | 4.0000 |
| Mean L1 distance | 1.0636 |
| Median L1 distance | 0.7046 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 186 |
| `cross_hole_distance_from_free_end` | 120 |
| `main_pin_radius` | 118 |
| `cross_hole_radius` | 114 |
| `splint_radius` | 106 |
| `splint_length` | 80 |
| `main_hole_radius` | 50 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `lock_main_bracket_user` |
| Scenario kind | `semantic_group` |
| Starts with locks | 200 |
| Mean locked parameter count | 6.00 |
| Mean locked gradient mass | 0.6086 |
| Mean locked delta mass | 0.4463 |
| Paired baseline | `ch5_sphere_shrink_128_matched_tau0.90` |
| Baseline oracle OK | 185 |
| Constrained oracle OK | 174 |
| Oracle success drop | 11 |
| Recoverability | 90.3% |
| Mean distance delta | 0.0453 |
| Median distance delta | 0.0191 |
| Same-or-better distance successes | 57 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `depth` | 200 |
| `leg_length` | 200 |
| `main_hole_offset_from_open_end` | 200 |
| `outer_span` | 200 |
| `overhang_span_y` | 200 |
| `wall_thickness` | 200 |
