# Optimization Statistics: one_shot_gradient_bracket_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 199 |
| Oracle confirmed | 190 |
| False successes | 9 |
| False success rate | 4.5% |
| No crossing (stuck) | 0 |
| Validity failures | 1 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.5189 | 0.5238 |
| Oracle confirmed | 0.5449 | 0.5714 |
| False success | 0.0281 | — |
| Stuck / no crossing | 0.0000 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 66 | 60 | 0.2743 |
| nearest_splint_clearance | 134 | 133 | 130 | 0.6394 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 1.9700 |
| Median active coordinates | 2.0000 |
| Mean L1 distance | 0.7325 |
| Median L1 distance | 0.7404 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 165 |
| `leg_length` | 157 |
| `overhang_span_y` | 40 |
| `main_hole_offset_from_open_end` | 32 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `grad_bottom_11` |
| Scenario kind | `gradient_ranked` |
| Starts with locks | 200 |
| Mean locked parameter count | 11.00 |
| Mean locked gradient mass | 0.5268 |
| Mean locked delta mass | 0.5083 |
| Paired baseline | `one_shot_gradient_bracket_v1_tau0.60_labelblind` |
| Baseline oracle OK | 146 |
| Constrained oracle OK | 190 |
| Oracle success drop | -44 |
| Recoverability | 99.3% |
| Mean distance delta | 0.2180 |
| Median distance delta | 0.1357 |
| Same-or-better distance successes | 6 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `cross_hole_distance_from_free_end` | 200 |
| `cross_hole_radius` | 200 |
| `depth` | 200 |
| `leg_length` | 42 |
| `main_hole_offset_from_open_end` | 167 |
| `main_hole_radius` | 200 |
| `main_pin_length` | 33 |
| `main_pin_radius` | 200 |
| `outer_span` | 200 |
| `overhang_span_y` | 158 |
| `splint_length` | 200 |
| `splint_radius` | 200 |
| `wall_thickness` | 200 |
