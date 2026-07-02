# Optimization Statistics: adaptive_multistep_penalty_lam0_10_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 150 |
| Oracle confirmed | 156 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 50 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.3452 | 0.3982 |
| Oracle confirmed | 0.3899 | 0.4445 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.1892 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 46 | 49 | 0.2839 |
| nearest_splint_clearance | 134 | 104 | 107 | 0.3754 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 9.3600 |
| Median active coordinates | 10.0000 |
| Mean L1 distance | 0.8196 |
| Median L1 distance | 0.9315 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `overhang_span_y` | 191 |
| `cross_hole_distance_from_free_end` | 190 |
| `splint_radius` | 190 |
| `leg_length` | 189 |
| `main_hole_offset_from_open_end` | 189 |
| `outer_span` | 187 |
| `splint_length` | 187 |
| `wall_thickness` | 186 |
| `cross_hole_radius` | 185 |
| `depth` | 178 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `lock_pin_user` |
| Scenario kind | `semantic_group` |
| Starts with locks | 200 |
| Mean locked parameter count | 3.00 |
| Mean locked gradient mass | 0.2706 |
| Mean locked delta mass | 0.2611 |
| Paired baseline | `task34_adaptive_penalty_lam0_10_tau0.75` |
| Baseline oracle OK | 165 |
| Constrained oracle OK | 156 |
| Oracle success drop | 9 |
| Recoverability | 94.5% |
| Mean distance delta | 0.0152 |
| Median distance delta | 0.0007 |
| Same-or-better distance successes | 54 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `main_hole_radius` | 200 |
| `main_pin_length` | 200 |
| `main_pin_radius` | 200 |
