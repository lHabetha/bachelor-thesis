# Optimization Statistics: one_shot_gradient_bracket_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 138 |
| Oracle confirmed | 135 |
| False successes | 8 |
| False success rate | 5.8% |
| No crossing (stuck) | 55 |
| Validity failures | 7 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.3050 | 0.3857 |
| Oracle confirmed | 0.3500 | 0.4238 |
| False success | 0.0585 | — |
| Stuck / no crossing | 0.2255 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 38 | 36 | 0.2297 |
| nearest_splint_clearance | 134 | 100 | 99 | 0.3420 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 9.0850 |
| Median active coordinates | 10.0000 |
| Mean L1 distance | 0.7267 |
| Median L1 distance | 0.8936 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_hole_offset_from_open_end` | 186 |
| `overhang_span_y` | 186 |
| `leg_length` | 185 |
| `wall_thickness` | 184 |
| `outer_span` | 183 |
| `splint_length` | 183 |
| `cross_hole_distance_from_free_end` | 182 |
| `splint_radius` | 182 |
| `cross_hole_radius` | 173 |
| `depth` | 173 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `lock_pin_user` |
| Scenario kind | `semantic_group` |
| Starts with locks | 200 |
| Mean locked parameter count | 3.00 |
| Mean locked gradient mass | 0.2706 |
| Mean locked delta mass | 0.2615 |
| Paired baseline | `one_shot_gradient_bracket_v1_tau0.60_labelblind` |
| Baseline oracle OK | 146 |
| Constrained oracle OK | 135 |
| Oracle success drop | 11 |
| Recoverability | 91.8% |
| Mean distance delta | 0.0040 |
| Median distance delta | 0.0000 |
| Same-or-better distance successes | 72 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `main_hole_radius` | 200 |
| `main_pin_length` | 200 |
| `main_pin_radius` | 200 |
