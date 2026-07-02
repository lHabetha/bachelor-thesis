# Optimization Statistics: multifixed_penalty_direction_frontier_lam0_50_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 162 |
| Oracle confirmed | 167 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 38 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.3397 | 0.3538 |
| Oracle confirmed | 0.3819 | 0.4087 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.1539 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 52 | 54 | 0.1903 |
| nearest_splint_clearance | 134 | 110 | 113 | 0.4133 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 11.9350 |
| Median active coordinates | 13.0000 |
| Mean L1 distance | 0.9144 |
| Median L1 distance | 0.9354 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 189 |
| `overhang_span_y` | 187 |
| `splint_radius` | 187 |
| `cross_hole_distance_from_free_end` | 186 |
| `main_hole_offset_from_open_end` | 186 |
| `main_pin_radius` | 186 |
| `leg_length` | 185 |
| `outer_span` | 183 |
| `splint_length` | 183 |
| `cross_hole_radius` | 182 |
| `main_hole_radius` | 182 |
| `wall_thickness` | 177 |
| `depth` | 174 |
