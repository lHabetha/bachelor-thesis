# Optimization Statistics: adaptive_multistep_penalty_lam0_10_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 146 |
| Oracle confirmed | 154 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 54 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.3960 | 0.3686 |
| Oracle confirmed | 0.4116 | 0.3790 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.3122 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 54 | 57 | 0.2164 |
| nearest_splint_clearance | 134 | 92 | 97 | 0.4845 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 6.6500 |
| Median active coordinates | 7.0000 |
| Mean L1 distance | 0.6879 |
| Median L1 distance | 0.6162 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `cross_hole_distance_from_free_end` | 194 |
| `main_pin_length` | 194 |
| `splint_radius` | 192 |
| `main_pin_radius` | 191 |
| `cross_hole_radius` | 190 |
| `splint_length` | 186 |
| `main_hole_radius` | 183 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `lock_main_bracket_user` |
| Scenario kind | `semantic_group` |
| Starts with locks | 200 |
| Mean locked parameter count | 6.00 |
| Mean locked gradient mass | 0.6086 |
| Mean locked delta mass | 0.5868 |
| Paired baseline | `task34_adaptive_penalty_lam0_10_tau0.75` |
| Baseline oracle OK | 165 |
| Constrained oracle OK | 154 |
| Oracle success drop | 11 |
| Recoverability | 86.1% |
| Mean distance delta | 0.0661 |
| Median distance delta | 0.0197 |
| Same-or-better distance successes | 22 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `depth` | 200 |
| `leg_length` | 200 |
| `main_hole_offset_from_open_end` | 200 |
| `outer_span` | 200 |
| `overhang_span_y` | 200 |
| `wall_thickness` | 200 |
