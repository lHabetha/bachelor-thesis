# Optimization Statistics: one_shot_gradient_bracket_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 140 |
| Oracle confirmed | 151 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 53 |
| Validity failures | 7 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.3408 | 0.3476 |
| Oracle confirmed | 0.3883 | 0.4238 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.1990 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 45 | 49 | 0.1982 |
| nearest_splint_clearance | 134 | 95 | 102 | 0.4110 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 12.0900 |
| Median active coordinates | 13.0000 |
| Mean L1 distance | 0.9114 |
| Median L1 distance | 0.9057 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 193 |
| `overhang_span_y` | 192 |
| `outer_span` | 190 |
| `cross_hole_distance_from_free_end` | 189 |
| `main_pin_radius` | 189 |
| `main_hole_offset_from_open_end` | 188 |
| `leg_length` | 187 |
| `splint_length` | 184 |
| `splint_radius` | 184 |
| `wall_thickness` | 184 |
| `cross_hole_radius` | 181 |
| `main_hole_radius` | 181 |
| `depth` | 176 |
