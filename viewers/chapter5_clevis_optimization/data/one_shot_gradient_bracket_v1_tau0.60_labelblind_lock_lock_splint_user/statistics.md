# Optimization Statistics: one_shot_gradient_bracket_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 154 |
| Oracle confirmed | 150 |
| False successes | 8 |
| False success rate | 5.2% |
| No crossing (stuck) | 40 |
| Validity failures | 6 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.3085 | 0.3190 |
| Oracle confirmed | 0.3472 | 0.3762 |
| False success | 0.0046 | — |
| Stuck / no crossing | 0.2411 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 52 | 49 | 0.1683 |
| nearest_splint_clearance | 134 | 102 | 101 | 0.3776 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 9.1300 |
| Median active coordinates | 10.0000 |
| Mean L1 distance | 0.7808 |
| Median L1 distance | 0.7778 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 189 |
| `main_pin_radius` | 188 |
| `overhang_span_y` | 187 |
| `leg_length` | 184 |
| `outer_span` | 184 |
| `main_hole_offset_from_open_end` | 183 |
| `main_hole_radius` | 183 |
| `cross_hole_distance_from_free_end` | 182 |
| `wall_thickness` | 176 |
| `depth` | 170 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `lock_splint_user` |
| Scenario kind | `semantic_group` |
| Starts with locks | 200 |
| Mean locked parameter count | 3.00 |
| Mean locked gradient mass | 0.0824 |
| Mean locked delta mass | 0.0792 |
| Paired baseline | `one_shot_gradient_bracket_v1_tau0.60_labelblind` |
| Baseline oracle OK | 146 |
| Constrained oracle OK | 150 |
| Oracle success drop | -4 |
| Recoverability | 99.3% |
| Mean distance delta | 0.0075 |
| Median distance delta | 0.0000 |
| Same-or-better distance successes | 93 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `cross_hole_radius` | 200 |
| `splint_length` | 200 |
| `splint_radius` | 200 |
