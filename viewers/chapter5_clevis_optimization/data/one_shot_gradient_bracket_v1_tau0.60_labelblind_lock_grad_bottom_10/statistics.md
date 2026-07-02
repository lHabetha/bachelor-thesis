# Optimization Statistics: one_shot_gradient_bracket_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 184 |
| Oracle confirmed | 176 |
| False successes | 9 |
| False success rate | 4.9% |
| No crossing (stuck) | 14 |
| Validity failures | 2 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.3787 | 0.3667 |
| Oracle confirmed | 0.4147 | 0.4238 |
| False success | 0.0060 | — |
| Stuck / no crossing | 0.2001 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 63 | 58 | 0.2105 |
| nearest_splint_clearance | 134 | 121 | 118 | 0.4615 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 2.9400 |
| Median active coordinates | 3.0000 |
| Mean L1 distance | 0.6500 |
| Median L1 distance | 0.6299 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `leg_length` | 167 |
| `main_pin_length` | 164 |
| `overhang_span_y` | 121 |
| `main_hole_offset_from_open_end` | 75 |
| `splint_length` | 32 |
| `outer_span` | 29 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `grad_bottom_10` |
| Scenario kind | `gradient_ranked` |
| Starts with locks | 200 |
| Mean locked parameter count | 10.00 |
| Mean locked gradient mass | 0.3821 |
| Mean locked delta mass | 0.3688 |
| Paired baseline | `one_shot_gradient_bracket_v1_tau0.60_labelblind` |
| Baseline oracle OK | 146 |
| Constrained oracle OK | 176 |
| Oracle success drop | -30 |
| Recoverability | 100.0% |
| Mean distance delta | 0.0777 |
| Median distance delta | 0.0000 |
| Same-or-better distance successes | 90 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `cross_hole_distance_from_free_end` | 200 |
| `cross_hole_radius` | 200 |
| `depth` | 200 |
| `leg_length` | 31 |
| `main_hole_offset_from_open_end` | 123 |
| `main_hole_radius` | 200 |
| `main_pin_length` | 33 |
| `main_pin_radius` | 200 |
| `outer_span` | 169 |
| `overhang_span_y` | 77 |
| `splint_length` | 167 |
| `splint_radius` | 200 |
| `wall_thickness` | 200 |
