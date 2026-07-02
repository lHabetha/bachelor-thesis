# Optimization Statistics: random_sphere_expanding_32_v1

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
| All 200 starts | 0.9200 | 1.0000 |
| Oracle confirmed | 0.9393 | 1.0000 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.8962 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 55 | 60 | 0.6439 |
| nearest_splint_clearance | 134 | 93 | 103 | 1.0560 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 12.9800 |
| Median active coordinates | 13.0000 |
| Mean L1 distance | 2.6856 |
| Median L1 distance | 2.6684 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `cross_hole_distance_from_free_end` | 200 |
| `depth` | 200 |
| `leg_length` | 200 |
| `main_hole_radius` | 200 |
| `main_pin_length` | 200 |
| `outer_span` | 200 |
| `overhang_span_y` | 200 |
| `splint_length` | 200 |
| `splint_radius` | 200 |
| `cross_hole_radius` | 199 |
| `main_hole_offset_from_open_end` | 199 |
| `main_pin_radius` | 199 |
| `wall_thickness` | 199 |
