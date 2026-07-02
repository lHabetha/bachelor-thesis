# Optimization Statistics: adaptive_multistep_gradient_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 157 |
| Oracle confirmed | 165 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 43 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.3575 | 0.3834 |
| Oracle confirmed | 0.3981 | 0.4389 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.1836 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 50 | 54 | 0.2152 |
| nearest_splint_clearance | 134 | 107 | 111 | 0.4276 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 12.2250 |
| Median active coordinates | 13.0000 |
| Mean L1 distance | 0.9559 |
| Median L1 distance | 1.0254 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 193 |
| `overhang_span_y` | 192 |
| `main_pin_radius` | 191 |
| `cross_hole_distance_from_free_end` | 190 |
| `outer_span` | 190 |
| `splint_radius` | 190 |
| `main_hole_offset_from_open_end` | 189 |
| `cross_hole_radius` | 188 |
| `leg_length` | 187 |
| `main_hole_radius` | 186 |
| `splint_length` | 185 |
| `wall_thickness` | 185 |
| `depth` | 179 |
