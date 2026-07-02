# Optimization Statistics: random_sphere_coordinate_shrink_v1

Model artifact: `hist_gradient_boosting_best_task30_labelblind_rank1`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 100 |
| Oracle confirmed | 168 |
| False successes | 2 |
| False success rate | 2.0% |
| No crossing (stuck) | 100 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.8999 | 0.8455 |
| Oracle confirmed | 0.9284 | 0.8552 |
| False success | 0.7275 | — |
| Stuck / no crossing | 1.2195 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 44 | 62 | 0.7420 |
| nearest_splint_clearance | 134 | 56 | 106 | 0.9777 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 7.9550 |
| Median active coordinates | 10.0000 |
| Mean L1 distance | 2.1903 |
| Median L1 distance | 1.4718 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 165 |
| `overhang_span_y` | 165 |
| `leg_length` | 138 |
| `main_hole_offset_from_open_end` | 134 |
| `splint_length` | 127 |
| `main_pin_radius` | 119 |
| `splint_radius` | 110 |
| `main_hole_radius` | 108 |
| `cross_hole_distance_from_free_end` | 107 |
| `cross_hole_radius` | 106 |
| `outer_span` | 106 |
| `depth` | 104 |
| `wall_thickness` | 102 |
