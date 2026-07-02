# Optimization Statistics: one_shot_gradient_bracket_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 95 |
| Oracle confirmed | 100 |
| False successes | 7 |
| False success rate | 7.4% |
| No crossing (stuck) | 98 |
| Validity failures | 7 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.4007 | 0.2619 |
| Oracle confirmed | 0.4462 | 0.3571 |
| False success | 0.2934 | — |
| Stuck / no crossing | 0.3296 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 22 | 28 | 0.2592 |
| nearest_splint_clearance | 134 | 73 | 72 | 0.4703 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 9.3000 |
| Median active coordinates | 10.0000 |
| Mean L1 distance | 0.9440 |
| Median L1 distance | 0.5358 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `cross_hole_distance_from_free_end` | 190 |
| `main_pin_radius` | 188 |
| `splint_radius` | 187 |
| `wall_thickness` | 187 |
| `main_hole_radius` | 185 |
| `cross_hole_radius` | 182 |
| `depth` | 182 |
| `outer_span` | 162 |
| `splint_length` | 154 |
| `main_hole_offset_from_open_end` | 113 |
| `overhang_span_y` | 73 |
| `main_pin_length` | 31 |
| `leg_length` | 26 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `grad_top_3` |
| Scenario kind | `gradient_ranked` |
| Starts with locks | 200 |
| Mean locked parameter count | 3.00 |
| Mean locked gradient mass | 0.6179 |
| Mean locked delta mass | 0.5962 |
| Paired baseline | `one_shot_gradient_bracket_v1_tau0.60_labelblind` |
| Baseline oracle OK | 146 |
| Constrained oracle OK | 100 |
| Oracle success drop | 46 |
| Recoverability | 64.4% |
| Mean distance delta | 0.0997 |
| Median distance delta | 0.0774 |
| Same-or-better distance successes | 0 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `leg_length` | 169 |
| `main_hole_offset_from_open_end` | 77 |
| `main_pin_length` | 167 |
| `outer_span` | 31 |
| `overhang_span_y` | 123 |
| `splint_length` | 33 |
