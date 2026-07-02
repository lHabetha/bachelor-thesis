# Optimization Statistics: adaptive_multistep_penalty_lam0_50_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 155 |
| Oracle confirmed | 161 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 45 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.3216 | 0.3473 |
| Oracle confirmed | 0.3541 | 0.3999 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.1925 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 51 | 54 | 0.1916 |
| nearest_splint_clearance | 134 | 104 | 107 | 0.3856 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 12.1100 |
| Median active coordinates | 13.0000 |
| Mean L1 distance | 0.8625 |
| Median L1 distance | 0.9341 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 193 |
| `overhang_span_y` | 191 |
| `cross_hole_distance_from_free_end` | 190 |
| `main_pin_radius` | 190 |
| `splint_radius` | 190 |
| `main_hole_offset_from_open_end` | 187 |
| `outer_span` | 187 |
| `leg_length` | 186 |
| `main_hole_radius` | 186 |
| `cross_hole_radius` | 185 |
| `splint_length` | 185 |
| `wall_thickness` | 178 |
| `depth` | 174 |
