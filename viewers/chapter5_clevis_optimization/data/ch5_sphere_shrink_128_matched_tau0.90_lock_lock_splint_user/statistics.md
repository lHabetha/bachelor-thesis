# Optimization Statistics: random_sphere_coordinate_shrink_128_v1

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
| All 200 starts | 0.4681 | 0.4259 |
| Oracle confirmed | 0.4681 | 0.4259 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.0000 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 66 | 66 | 0.3045 |
| nearest_splint_clearance | 134 | 134 | 134 | 0.5487 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 4.1350 |
| Median active coordinates | 4.0000 |
| Mean L1 distance | 0.7866 |
| Median L1 distance | 0.6632 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 144 |
| `overhang_span_y` | 135 |
| `main_pin_radius` | 104 |
| `leg_length` | 98 |
| `cross_hole_distance_from_free_end` | 97 |
| `main_hole_offset_from_open_end` | 85 |
| `main_hole_radius` | 66 |
| `depth` | 51 |
| `outer_span` | 38 |
| `wall_thickness` | 9 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `lock_splint_user` |
| Scenario kind | `semantic_group` |
| Starts with locks | 200 |
| Mean locked parameter count | 3.00 |
| Mean locked gradient mass | 0.0824 |
| Mean locked delta mass | 0.1689 |
| Paired baseline | `ch5_sphere_shrink_128_matched_tau0.90` |
| Baseline oracle OK | 185 |
| Constrained oracle OK | 200 |
| Oracle success drop | -15 |
| Recoverability | 100.0% |
| Mean distance delta | -0.0972 |
| Median distance delta | -0.0214 |
| Same-or-better distance successes | 119 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `cross_hole_radius` | 200 |
| `splint_length` | 200 |
| `splint_radius` | 200 |
