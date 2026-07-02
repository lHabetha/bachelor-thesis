# Optimization Statistics: multifixed_penalty_direction_frontier_lam0_10_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 160 |
| Oracle confirmed | 167 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 40 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.3650 | 0.3899 |
| Oracle confirmed | 0.4109 | 0.4399 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.1657 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 51 | 54 | 0.2147 |
| nearest_splint_clearance | 134 | 109 | 113 | 0.4391 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 12.0550 |
| Median active coordinates | 13.0000 |
| Mean L1 distance | 0.9756 |
| Median L1 distance | 1.0257 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 189 |
| `main_hole_offset_from_open_end` | 188 |
| `main_pin_radius` | 188 |
| `overhang_span_y` | 188 |
| `splint_radius` | 188 |
| `cross_hole_distance_from_free_end` | 186 |
| `leg_length` | 186 |
| `outer_span` | 186 |
| `cross_hole_radius` | 185 |
| `wall_thickness` | 184 |
| `main_hole_radius` | 183 |
| `splint_length` | 183 |
| `depth` | 177 |
