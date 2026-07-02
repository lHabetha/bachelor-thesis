# Optimization Statistics: trust_region_hybrid_v1

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
| All 200 starts | 0.5770 | 0.5000 |
| Oracle confirmed | 0.5770 | 0.5000 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.0000 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 66 | 66 | 0.3539 |
| nearest_splint_clearance | 134 | 134 | 134 | 0.6869 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 6.0150 |
| Median active coordinates | 1.0000 |
| Mean L1 distance | 0.8768 |
| Median L1 distance | 0.5000 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 140 |
| `leg_length` | 129 |
| `overhang_span_y` | 100 |
| `main_hole_offset_from_open_end` | 87 |
| `splint_radius` | 86 |
| `main_pin_radius` | 85 |
| `splint_length` | 84 |
| `wall_thickness` | 84 |
| `cross_hole_distance_from_free_end` | 83 |
| `cross_hole_radius` | 83 |
| `main_hole_radius` | 82 |
| `outer_span` | 81 |
| `depth` | 79 |
