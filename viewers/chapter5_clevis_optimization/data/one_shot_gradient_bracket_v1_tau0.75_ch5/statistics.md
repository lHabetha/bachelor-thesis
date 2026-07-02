# Optimization Statistics: one_shot_gradient_bracket_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 143 |
| Oracle confirmed | 151 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 50 |
| Validity failures | 7 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.3164 | 0.3286 |
| Oracle confirmed | 0.3561 | 0.4048 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.2042 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 45 | 49 | 0.1774 |
| nearest_splint_clearance | 134 | 98 | 102 | 0.3849 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 11.9900 |
| Median active coordinates | 13.0000 |
| Mean L1 distance | 0.8512 |
| Median L1 distance | 0.8561 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 193 |
| `overhang_span_y` | 191 |
| `cross_hole_distance_from_free_end` | 189 |
| `main_pin_radius` | 188 |
| `outer_span` | 187 |
| `leg_length` | 186 |
| `main_hole_offset_from_open_end` | 186 |
| `splint_length` | 184 |
| `splint_radius` | 184 |
| `main_hole_radius` | 180 |
| `wall_thickness` | 179 |
| `cross_hole_radius` | 178 |
| `depth` | 173 |
