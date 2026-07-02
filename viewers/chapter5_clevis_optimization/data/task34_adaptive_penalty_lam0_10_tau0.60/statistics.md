# Optimization Statistics: adaptive_multistep_penalty_lam0_10_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 163 |
| Oracle confirmed | 158 |
| False successes | 8 |
| False success rate | 4.9% |
| No crossing (stuck) | 37 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.3130 | 0.3282 |
| Oracle confirmed | 0.3601 | 0.3951 |
| False success | 0.0331 | — |
| Stuck / no crossing | 0.1860 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 54 | 49 | 0.1767 |
| nearest_splint_clearance | 134 | 109 | 109 | 0.3801 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 11.7650 |
| Median active coordinates | 13.0000 |
| Mean L1 distance | 0.8454 |
| Median L1 distance | 0.8936 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 188 |
| `main_pin_radius` | 186 |
| `overhang_span_y` | 186 |
| `splint_radius` | 186 |
| `main_hole_radius` | 183 |
| `outer_span` | 183 |
| `cross_hole_distance_from_free_end` | 182 |
| `leg_length` | 182 |
| `main_hole_offset_from_open_end` | 181 |
| `splint_length` | 178 |
| `wall_thickness` | 175 |
| `cross_hole_radius` | 174 |
| `depth` | 169 |
