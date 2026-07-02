# Optimization Statistics: one_shot_gradient_bracket_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 150 |
| Oracle confirmed | 137 |
| False successes | 17 |
| False success rate | 11.3% |
| No crossing (stuck) | 43 |
| Validity failures | 7 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.2903 | 0.2905 |
| Oracle confirmed | 0.3434 | 0.3857 |
| False success | 0.0883 | — |
| Stuck / no crossing | 0.2204 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 50 | 39 | 0.1550 |
| nearest_splint_clearance | 134 | 100 | 98 | 0.3570 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 11.2950 |
| Median active coordinates | 13.0000 |
| Mean L1 distance | 0.7861 |
| Median L1 distance | 0.7568 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 181 |
| `overhang_span_y` | 180 |
| `main_pin_radius` | 178 |
| `outer_span` | 178 |
| `leg_length` | 177 |
| `splint_radius` | 176 |
| `cross_hole_distance_from_free_end` | 175 |
| `main_hole_radius` | 175 |
| `splint_length` | 175 |
| `main_hole_offset_from_open_end` | 174 |
| `wall_thickness` | 170 |
| `cross_hole_radius` | 162 |
| `depth` | 158 |
