# Optimization Statistics: random_sphere_coordinate_shrink_128_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 129 |
| Oracle confirmed | 144 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 71 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.9267 | 0.8621 |
| Oracle confirmed | 0.8114 | 0.6930 |
| False success | 0.0000 | — |
| Stuck / no crossing | 1.3451 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 46 | 51 | 0.7924 |
| nearest_splint_clearance | 134 | 83 | 93 | 0.9928 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 6.1900 |
| Median active coordinates | 6.0000 |
| Mean L1 distance | 1.9912 |
| Median L1 distance | 1.4763 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_radius` | 140 |
| `cross_hole_distance_from_free_end` | 127 |
| `main_hole_radius` | 120 |
| `cross_hole_radius` | 117 |
| `splint_radius` | 117 |
| `outer_span` | 111 |
| `splint_length` | 108 |
| `main_hole_offset_from_open_end` | 105 |
| `depth` | 94 |
| `wall_thickness` | 87 |
| `overhang_span_y` | 68 |
| `main_pin_length` | 24 |
| `leg_length` | 20 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `grad_top_3` |
| Scenario kind | `gradient_ranked` |
| Starts with locks | 200 |
| Mean locked parameter count | 3.00 |
| Mean locked gradient mass | 0.6179 |
| Mean locked delta mass | 0.4653 |
| Paired baseline | `ch5_sphere_shrink_128_matched_tau0.90` |
| Baseline oracle OK | 185 |
| Constrained oracle OK | 144 |
| Oracle success drop | 41 |
| Recoverability | 77.3% |
| Mean distance delta | 0.3613 |
| Median distance delta | 0.2654 |
| Same-or-better distance successes | 17 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `leg_length` | 169 |
| `main_hole_offset_from_open_end` | 77 |
| `main_pin_length` | 167 |
| `outer_span` | 31 |
| `overhang_span_y` | 123 |
| `splint_length` | 33 |
