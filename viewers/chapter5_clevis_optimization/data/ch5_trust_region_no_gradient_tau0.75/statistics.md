# Optimization Statistics: trust_region_no_gradient_v1

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
| All 200 starts | 0.5857 | 0.5000 |
| Oracle confirmed | 0.5857 | 0.5000 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.0000 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 66 | 66 | 0.3642 |
| nearest_splint_clearance | 134 | 134 | 134 | 0.6948 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 2.4250 |
| Median active coordinates | 1.0000 |
| Mean L1 distance | 0.6579 |
| Median L1 distance | 0.5000 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 106 |
| `leg_length` | 92 |
| `overhang_span_y` | 41 |
| `main_hole_offset_from_open_end` | 33 |
| `cross_hole_distance_from_free_end` | 24 |
| `cross_hole_radius` | 24 |
| `depth` | 24 |
| `main_pin_radius` | 24 |
| `splint_length` | 24 |
| `splint_radius` | 24 |
| `wall_thickness` | 24 |
| `outer_span` | 23 |
| `main_hole_radius` | 22 |
