# Optimization Statistics: one_shot_gradient_bracket_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 148 |
| Oracle confirmed | 146 |
| False successes | 7 |
| False success rate | 4.7% |
| No crossing (stuck) | 45 |
| Validity failures | 7 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.3010 | 0.3095 |
| Oracle confirmed | 0.3469 | 0.3857 |
| False success | 0.0031 | — |
| Stuck / no crossing | 0.2123 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 49 | 46 | 0.1643 |
| nearest_splint_clearance | 134 | 99 | 100 | 0.3683 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 11.6950 |
| Median active coordinates | 13.0000 |
| Mean L1 distance | 0.8129 |
| Median L1 distance | 0.8064 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 188 |
| `main_pin_radius` | 186 |
| `overhang_span_y` | 186 |
| `outer_span` | 183 |
| `leg_length` | 182 |
| `cross_hole_distance_from_free_end` | 181 |
| `main_hole_offset_from_open_end` | 181 |
| `splint_radius` | 181 |
| `splint_length` | 180 |
| `main_hole_radius` | 179 |
| `wall_thickness` | 175 |
| `cross_hole_radius` | 170 |
| `depth` | 167 |
