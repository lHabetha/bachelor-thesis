# Optimization Statistics: random_sphere_expanding_128_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 179 |
| Oracle confirmed | 185 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 21 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.8452 | 1.0000 |
| Oracle confirmed | 0.8057 | 1.0000 |
| False success | 0.0000 | — |
| Stuck / no crossing | 1.3095 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 64 | 64 | 0.5295 |
| nearest_splint_clearance | 134 | 115 | 121 | 1.0007 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 12.9950 |
| Median active coordinates | 13.0000 |
| Mean L1 distance | 2.4574 |
| Median L1 distance | 2.5555 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `cross_hole_distance_from_free_end` | 200 |
| `cross_hole_radius` | 200 |
| `depth` | 200 |
| `leg_length` | 200 |
| `main_hole_offset_from_open_end` | 200 |
| `main_hole_radius` | 200 |
| `main_pin_length` | 200 |
| `main_pin_radius` | 200 |
| `outer_span` | 200 |
| `overhang_span_y` | 200 |
| `splint_radius` | 200 |
| `wall_thickness` | 200 |
| `splint_length` | 199 |
