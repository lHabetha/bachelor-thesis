# Optimization Statistics: random_sphere_coordinate_shrink_256_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 192 |
| Oracle confirmed | 195 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 8 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.5016 | 0.4683 |
| Oracle confirmed | 0.4888 | 0.4561 |
| False success | 0.0000 | — |
| Stuck / no crossing | 1.0625 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 66 | 66 | 0.2893 |
| nearest_splint_clearance | 134 | 126 | 129 | 0.6061 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 5.5750 |
| Median active coordinates | 5.0000 |
| Mean L1 distance | 0.9770 |
| Median L1 distance | 0.7453 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 142 |
| `overhang_span_y` | 134 |
| `leg_length` | 116 |
| `main_pin_radius` | 103 |
| `main_hole_offset_from_open_end` | 95 |
| `cross_hole_radius` | 83 |
| `splint_radius` | 80 |
| `cross_hole_distance_from_free_end` | 79 |
| `splint_length` | 78 |
| `main_hole_radius` | 72 |
| `depth` | 62 |
| `outer_span` | 51 |
| `wall_thickness` | 20 |
