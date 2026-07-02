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
| All 200 starts | 0.3234 | 0.3286 |
| Oracle confirmed | 0.3653 | 0.4048 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.2042 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 45 | 49 | 0.1831 |
| nearest_splint_clearance | 134 | 98 | 102 | 0.3925 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 12.0450 |
| Median active coordinates | 13.0000 |
| Mean L1 distance | 0.8684 |
| Median L1 distance | 0.8561 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 193 |
| `overhang_span_y` | 191 |
| `cross_hole_distance_from_free_end` | 189 |
| `main_hole_offset_from_open_end` | 188 |
| `main_pin_radius` | 188 |
| `outer_span` | 188 |
| `leg_length` | 187 |
| `splint_length` | 184 |
| `splint_radius` | 184 |
| `wall_thickness` | 183 |
| `main_hole_radius` | 180 |
| `cross_hole_radius` | 179 |
| `depth` | 175 |
