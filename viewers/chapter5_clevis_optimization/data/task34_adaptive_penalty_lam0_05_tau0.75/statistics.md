# Optimization Statistics: adaptive_multistep_penalty_lam0_05_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 158 |
| Oracle confirmed | 165 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 42 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.3314 | 0.3473 |
| Oracle confirmed | 0.3671 | 0.4043 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.1833 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 51 | 54 | 0.1917 |
| nearest_splint_clearance | 134 | 107 | 111 | 0.4002 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 12.1150 |
| Median active coordinates | 13.0000 |
| Mean L1 distance | 0.8915 |
| Median L1 distance | 0.9316 |

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
| `splint_length` | 185 |
| `cross_hole_radius` | 184 |
| `wall_thickness` | 178 |
| `depth` | 176 |
