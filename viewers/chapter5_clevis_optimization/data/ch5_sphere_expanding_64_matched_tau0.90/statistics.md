# Optimization Statistics: random_sphere_expanding_v1

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
| All 200 starts | 0.8085 | 1.0000 |
| Oracle confirmed | 0.8035 | 0.5000 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.8585 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 59 | 63 | 0.5697 |
| nearest_splint_clearance | 134 | 100 | 110 | 0.9261 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 12.9900 |
| Median active coordinates | 13.0000 |
| Mean L1 distance | 2.3603 |
| Median L1 distance | 2.6466 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `depth` | 200 |
| `leg_length` | 200 |
| `main_hole_offset_from_open_end` | 200 |
| `main_hole_radius` | 200 |
| `main_pin_length` | 200 |
| `main_pin_radius` | 200 |
| `outer_span` | 200 |
| `overhang_span_y` | 200 |
| `splint_length` | 200 |
| `splint_radius` | 200 |
| `wall_thickness` | 200 |
| `cross_hole_distance_from_free_end` | 199 |
| `cross_hole_radius` | 199 |
