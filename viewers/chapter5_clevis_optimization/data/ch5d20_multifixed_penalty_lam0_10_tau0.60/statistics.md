# Optimization Statistics: multifixed_penalty_direction_frontier_lam0_10_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 165 |
| Oracle confirmed | 161 |
| False successes | 7 |
| False success rate | 4.2% |
| No crossing (stuck) | 35 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.3204 | 0.3316 |
| Oracle confirmed | 0.3692 | 0.3970 |
| False success | 0.0377 | — |
| Stuck / no crossing | 0.1679 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 54 | 50 | 0.1773 |
| nearest_splint_clearance | 134 | 111 | 111 | 0.3909 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 11.6600 |
| Median active coordinates | 13.0000 |
| Mean L1 distance | 0.8652 |
| Median L1 distance | 0.8993 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 185 |
| `splint_radius` | 185 |
| `main_pin_radius` | 184 |
| `overhang_span_y` | 183 |
| `leg_length` | 181 |
| `main_hole_radius` | 181 |
| `cross_hole_distance_from_free_end` | 180 |
| `main_hole_offset_from_open_end` | 180 |
| `outer_span` | 180 |
| `splint_length` | 177 |
| `cross_hole_radius` | 174 |
| `wall_thickness` | 174 |
| `depth` | 168 |
