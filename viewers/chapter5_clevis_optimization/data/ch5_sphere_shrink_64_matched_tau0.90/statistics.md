# Optimization Statistics: random_sphere_coordinate_shrink_64_matched_seed_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 159 |
| Oracle confirmed | 173 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 41 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.5257 | 0.5000 |
| Oracle confirmed | 0.4765 | 0.4255 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.8585 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 59 | 63 | 0.3343 |
| nearest_splint_clearance | 134 | 100 | 110 | 0.6200 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 6.6500 |
| Median active coordinates | 6.0000 |
| Mean L1 distance | 1.1582 |
| Median L1 distance | 0.8093 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `overhang_span_y` | 146 |
| `main_pin_length` | 145 |
| `main_pin_radius` | 127 |
| `leg_length` | 117 |
| `cross_hole_distance_from_free_end` | 116 |
| `splint_radius` | 106 |
| `main_hole_offset_from_open_end` | 105 |
| `cross_hole_radius` | 104 |
| `splint_length` | 103 |
| `main_hole_radius` | 81 |
| `outer_span` | 67 |
| `depth` | 64 |
| `wall_thickness` | 49 |
