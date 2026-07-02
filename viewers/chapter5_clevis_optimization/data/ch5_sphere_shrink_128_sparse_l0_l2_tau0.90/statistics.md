# Optimization Statistics: random_sphere_coordinate_shrink_128_sparse_l0_l2_v1

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
| All 200 starts | 0.5654 | 0.5000 |
| Oracle confirmed | 0.5031 | 0.4648 |
| False success | 0.0000 | — |
| Stuck / no crossing | 1.3095 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 64 | 64 | 0.3388 |
| nearest_splint_clearance | 134 | 115 | 121 | 0.6769 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 6.1450 |
| Median active coordinates | 5.0000 |
| Mean L1 distance | 1.2006 |
| Median L1 distance | 0.8739 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 144 |
| `overhang_span_y` | 139 |
| `main_pin_radius` | 120 |
| `cross_hole_radius` | 112 |
| `leg_length` | 112 |
| `main_hole_offset_from_open_end` | 107 |
| `cross_hole_distance_from_free_end` | 95 |
| `splint_length` | 91 |
| `splint_radius` | 87 |
| `main_hole_radius` | 74 |
| `depth` | 59 |
| `outer_span` | 59 |
| `wall_thickness` | 30 |
