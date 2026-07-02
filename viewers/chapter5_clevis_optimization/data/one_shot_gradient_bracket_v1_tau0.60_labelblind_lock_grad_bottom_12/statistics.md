# Optimization Statistics: one_shot_gradient_bracket_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 200 |
| Oracle confirmed | 194 |
| False successes | 6 |
| False success rate | 3.0% |
| No crossing (stuck) | 0 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.4255 | 0.4333 |
| Oracle confirmed | 0.4386 | 0.4524 |
| False success | 0.0027 | — |
| Stuck / no crossing | 0.0000 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 66 | 62 | 0.2491 |
| nearest_splint_clearance | 134 | 134 | 132 | 0.5124 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 0.9900 |
| Median active coordinates | 1.0000 |
| Mean L1 distance | 0.4255 |
| Median L1 distance | 0.4333 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 111 |
| `leg_length` | 87 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `grad_bottom_12` |
| Scenario kind | `gradient_ranked` |
| Starts with locks | 200 |
| Mean locked parameter count | 12.00 |
| Mean locked gradient mass | 0.7406 |
| Mean locked delta mass | 0.7146 |
| Paired baseline | `one_shot_gradient_bracket_v1_tau0.60_labelblind` |
| Baseline oracle OK | 146 |
| Constrained oracle OK | 194 |
| Oracle success drop | -48 |
| Recoverability | 100.0% |
| Mean distance delta | 0.1245 |
| Median distance delta | 0.0145 |
| Same-or-better distance successes | 56 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `cross_hole_distance_from_free_end` | 200 |
| `cross_hole_radius` | 200 |
| `depth` | 200 |
| `leg_length` | 113 |
| `main_hole_offset_from_open_end` | 200 |
| `main_hole_radius` | 200 |
| `main_pin_length` | 87 |
| `main_pin_radius` | 200 |
| `outer_span` | 200 |
| `overhang_span_y` | 200 |
| `splint_length` | 200 |
| `splint_radius` | 200 |
| `wall_thickness` | 200 |
