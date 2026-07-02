# Optimization Statistics: one_shot_gradient_bracket_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 143 |
| Oracle confirmed | 145 |
| False successes | 6 |
| False success rate | 4.2% |
| No crossing (stuck) | 51 |
| Validity failures | 6 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.3934 | 0.3476 |
| Oracle confirmed | 0.3992 | 0.3857 |
| False success | 0.0057 | — |
| Stuck / no crossing | 0.3969 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 55 | 54 | 0.2006 |
| nearest_splint_clearance | 134 | 88 | 91 | 0.4883 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 6.4700 |
| Median active coordinates | 7.0000 |
| Mean L1 distance | 0.6871 |
| Median L1 distance | 0.5866 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 191 |
| `cross_hole_distance_from_free_end` | 190 |
| `main_pin_radius` | 187 |
| `splint_radius` | 184 |
| `splint_length` | 182 |
| `cross_hole_radius` | 181 |
| `main_hole_radius` | 179 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `lock_main_bracket_user` |
| Scenario kind | `semantic_group` |
| Starts with locks | 200 |
| Mean locked parameter count | 6.00 |
| Mean locked gradient mass | 0.6086 |
| Mean locked delta mass | 0.5873 |
| Paired baseline | `one_shot_gradient_bracket_v1_tau0.60_labelblind` |
| Baseline oracle OK | 146 |
| Constrained oracle OK | 145 |
| Oracle success drop | 1 |
| Recoverability | 83.6% |
| Mean distance delta | 0.0924 |
| Median distance delta | 0.0212 |
| Same-or-better distance successes | 34 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `depth` | 200 |
| `leg_length` | 200 |
| `main_hole_offset_from_open_end` | 200 |
| `outer_span` | 200 |
| `overhang_span_y` | 200 |
| `wall_thickness` | 200 |
