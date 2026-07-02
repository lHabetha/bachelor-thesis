# Optimization Statistics: random_sphere_expanding_256_v1

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
| All 200 starts | 0.8187 | 1.0000 |
| Oracle confirmed | 0.8141 | 1.0000 |
| False success | 0.0000 | — |
| Stuck / no crossing | 1.0625 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 66 | 66 | 0.4705 |
| nearest_splint_clearance | 134 | 126 | 129 | 0.9903 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 12.9750 |
| Median active coordinates | 13.0000 |
| Mean L1 distance | 2.3648 |
| Median L1 distance | 2.6172 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `cross_hole_distance_from_free_end` | 200 |
| `cross_hole_radius` | 200 |
| `depth` | 200 |
| `leg_length` | 200 |
| `main_hole_radius` | 200 |
| `main_pin_length` | 200 |
| `main_pin_radius` | 200 |
| `wall_thickness` | 200 |
| `main_hole_offset_from_open_end` | 199 |
| `outer_span` | 199 |
| `overhang_span_y` | 199 |
| `splint_length` | 199 |
| `splint_radius` | 199 |
