# Optimization Statistics: one_shot_gradient_bracket_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 144 |
| Oracle confirmed | 151 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 49 |
| Validity failures | 7 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.3108 | 0.3095 |
| Oracle confirmed | 0.3486 | 0.3857 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.2069 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 45 | 49 | 0.1715 |
| nearest_splint_clearance | 134 | 99 | 102 | 0.3794 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 11.9400 |
| Median active coordinates | 13.0000 |
| Mean L1 distance | 0.8372 |
| Median L1 distance | 0.8064 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 193 |
| `overhang_span_y` | 190 |
| `main_pin_radius` | 188 |
| `cross_hole_distance_from_free_end` | 187 |
| `outer_span` | 187 |
| `leg_length` | 185 |
| `main_hole_offset_from_open_end` | 185 |
| `splint_radius` | 184 |
| `splint_length` | 183 |
| `main_hole_radius` | 179 |
| `cross_hole_radius` | 178 |
| `wall_thickness` | 177 |
| `depth` | 172 |
