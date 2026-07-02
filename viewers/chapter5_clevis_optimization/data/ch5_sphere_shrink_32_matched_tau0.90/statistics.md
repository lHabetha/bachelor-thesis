# Optimization Statistics: random_sphere_coordinate_shrink_32_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 148 |
| Oracle confirmed | 163 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 52 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.5718 | 0.5000 |
| Oracle confirmed | 0.5121 | 0.3940 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.8962 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 55 | 60 | 0.3877 |
| nearest_splint_clearance | 134 | 93 | 103 | 0.6626 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 6.9150 |
| Median active coordinates | 6.0000 |
| Mean L1 distance | 1.3005 |
| Median L1 distance | 0.9282 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 153 |
| `overhang_span_y` | 138 |
| `leg_length` | 125 |
| `main_pin_radius` | 123 |
| `main_hole_offset_from_open_end` | 116 |
| `cross_hole_radius` | 110 |
| `cross_hole_distance_from_free_end` | 104 |
| `splint_length` | 102 |
| `splint_radius` | 102 |
| `main_hole_radius` | 98 |
| `depth` | 83 |
| `outer_span` | 73 |
| `wall_thickness` | 56 |
