# Optimization Statistics: multifixed_penalty_direction_frontier_lam0_20_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 163 |
| Oracle confirmed | 166 |
| False successes | 1 |
| False success rate | 0.6% |
| No crossing (stuck) | 37 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.3399 | 0.3538 |
| Oracle confirmed | 0.3778 | 0.4065 |
| False success | 0.6979 | — |
| Stuck / no crossing | 0.1430 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 52 | 54 | 0.1914 |
| nearest_splint_clearance | 134 | 111 | 112 | 0.4131 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 11.9450 |
| Median active coordinates | 13.0000 |
| Mean L1 distance | 0.9128 |
| Median L1 distance | 0.9324 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 189 |
| `splint_radius` | 188 |
| `main_pin_radius` | 187 |
| `overhang_span_y` | 187 |
| `cross_hole_distance_from_free_end` | 186 |
| `main_hole_offset_from_open_end` | 186 |
| `leg_length` | 185 |
| `outer_span` | 183 |
| `splint_length` | 183 |
| `cross_hole_radius` | 182 |
| `main_hole_radius` | 182 |
| `wall_thickness` | 177 |
| `depth` | 174 |
