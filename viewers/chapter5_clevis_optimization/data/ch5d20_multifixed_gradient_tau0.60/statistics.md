# Optimization Statistics: receding_gradient_multiscale_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 162 |
| Oracle confirmed | 158 |
| False successes | 7 |
| False success rate | 4.3% |
| No crossing (stuck) | 38 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.4196 | 0.5000 |
| Oracle confirmed | 0.4889 | 0.5000 |
| False success | 0.1025 | — |
| Stuck / no crossing | 0.1959 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 53 | 50 | 0.2496 |
| nearest_splint_clearance | 134 | 109 | 108 | 0.5033 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 11.8300 |
| Median active coordinates | 13.0000 |
| Mean L1 distance | 1.1308 |
| Median L1 distance | 1.3243 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 187 |
| `main_pin_radius` | 187 |
| `splint_radius` | 185 |
| `main_hole_offset_from_open_end` | 184 |
| `overhang_span_y` | 184 |
| `cross_hole_distance_from_free_end` | 183 |
| `leg_length` | 183 |
| `outer_span` | 182 |
| `main_hole_radius` | 181 |
| `cross_hole_radius` | 180 |
| `splint_length` | 180 |
| `wall_thickness` | 178 |
| `depth` | 172 |
