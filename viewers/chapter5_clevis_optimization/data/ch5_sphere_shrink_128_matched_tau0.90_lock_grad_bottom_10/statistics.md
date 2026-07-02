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
| All 200 starts | 0.4286 | 0.4235 |
| Oracle confirmed | 0.4286 | 0.4235 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.0000 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 66 | 66 | 0.2571 |
| nearest_splint_clearance | 134 | 134 | 134 | 0.5130 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 2.0050 |
| Median active coordinates | 2.0000 |
| Mean L1 distance | 0.5221 |
| Median L1 distance | 0.5225 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 147 |
| `overhang_span_y` | 97 |
| `leg_length` | 76 |
| `main_hole_offset_from_open_end` | 51 |
| `splint_length` | 23 |
| `outer_span` | 7 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `grad_bottom_10` |
| Scenario kind | `gradient_ranked` |
| Starts with locks | 200 |
| Mean locked parameter count | 10.00 |
| Mean locked gradient mass | 0.3821 |
| Mean locked delta mass | 0.5347 |
| Paired baseline | `ch5_sphere_shrink_128_matched_tau0.90` |
| Baseline oracle OK | 185 |
| Constrained oracle OK | 200 |
| Oracle success drop | -15 |
| Recoverability | 100.0% |
| Mean distance delta | -0.1368 |
| Median distance delta | -0.0481 |
| Same-or-better distance successes | 137 |

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
