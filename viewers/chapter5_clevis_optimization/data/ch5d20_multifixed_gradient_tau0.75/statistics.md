# Optimization Statistics: receding_gradient_multiscale_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 159 |
| Oracle confirmed | 164 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 41 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.4504 | 0.5000 |
| Oracle confirmed | 0.5130 | 0.5000 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.1832 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 51 | 53 | 0.2576 |
| nearest_splint_clearance | 134 | 108 | 111 | 0.5454 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 12.0050 |
| Median active coordinates | 13.0000 |
| Mean L1 distance | 1.2092 |
| Median L1 distance | 1.3268 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 189 |
| `overhang_span_y` | 188 |
| `main_hole_offset_from_open_end` | 187 |
| `main_pin_radius` | 187 |
| `splint_radius` | 187 |
| `cross_hole_distance_from_free_end` | 186 |
| `leg_length` | 186 |
| `outer_span` | 184 |
| `cross_hole_radius` | 183 |
| `splint_length` | 183 |
| `wall_thickness` | 183 |
| `main_hole_radius` | 182 |
| `depth` | 176 |
