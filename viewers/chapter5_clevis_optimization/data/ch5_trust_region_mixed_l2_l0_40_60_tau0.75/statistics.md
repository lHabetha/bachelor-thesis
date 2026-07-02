# Optimization Statistics: trust_region_mixed_l2_l0_40_60_v1

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
| Mean active coordinates | 2.4700 |
| Median active coordinates | 1.0000 |
| Mean L1 distance | 0.5863 |
| Median L1 distance | 0.5000 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 111 |
| `leg_length` | 93 |
| `overhang_span_y` | 40 |
| `main_hole_offset_from_open_end` | 32 |
| `splint_radius` | 26 |
| `main_hole_radius` | 25 |
| `main_pin_radius` | 25 |
| `splint_length` | 25 |
| `cross_hole_radius` | 24 |
| `wall_thickness` | 24 |
| `cross_hole_distance_from_free_end` | 23 |
| `depth` | 23 |
| `outer_span` | 23 |
