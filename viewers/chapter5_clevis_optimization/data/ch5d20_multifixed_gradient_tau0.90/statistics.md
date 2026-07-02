# Optimization Statistics: receding_gradient_multiscale_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 157 |
| Oracle confirmed | 164 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 43 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.4805 | 0.5000 |
| Oracle confirmed | 0.5498 | 0.5000 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.1903 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 50 | 53 | 0.2847 |
| nearest_splint_clearance | 134 | 107 | 111 | 0.5770 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 12.1150 |
| Median active coordinates | 13.0000 |
| Mean L1 distance | 1.2832 |
| Median L1 distance | 1.3281 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `leg_length` | 189 |
| `main_pin_length` | 189 |
| `main_hole_offset_from_open_end` | 188 |
| `main_pin_radius` | 188 |
| `outer_span` | 188 |
| `overhang_span_y` | 188 |
| `cross_hole_distance_from_free_end` | 187 |
| `splint_radius` | 187 |
| `cross_hole_radius` | 185 |
| `wall_thickness` | 185 |
| `main_hole_radius` | 184 |
| `splint_length` | 184 |
| `depth` | 181 |
