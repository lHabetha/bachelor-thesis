# Optimization Statistics: adaptive_multistep_penalty_lam0_10_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 111 |
| Oracle confirmed | 124 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 89 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.4519 | 0.3750 |
| Oracle confirmed | 0.5467 | 0.5658 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.3028 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 27 | 34 | 0.3092 |
| nearest_splint_clearance | 134 | 84 | 90 | 0.5222 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 9.4600 |
| Median active coordinates | 10.0000 |
| Mean L1 distance | 1.0484 |
| Median L1 distance | 0.8668 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `cross_hole_distance_from_free_end` | 193 |
| `main_pin_radius` | 191 |
| `splint_radius` | 190 |
| `cross_hole_radius` | 189 |
| `wall_thickness` | 189 |
| `main_hole_radius` | 188 |
| `depth` | 185 |
| `outer_span` | 162 |
| `splint_length` | 157 |
| `main_hole_offset_from_open_end` | 115 |
| `overhang_span_y` | 74 |
| `main_pin_length` | 32 |
| `leg_length` | 27 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `grad_top_3` |
| Scenario kind | `gradient_ranked` |
| Starts with locks | 200 |
| Mean locked parameter count | 3.00 |
| Mean locked gradient mass | 0.6179 |
| Mean locked delta mass | 0.5964 |
| Paired baseline | `task34_adaptive_penalty_lam0_10_tau0.75` |
| Baseline oracle OK | 165 |
| Constrained oracle OK | 124 |
| Oracle success drop | 41 |
| Recoverability | 72.7% |
| Mean distance delta | 0.1219 |
| Median distance delta | 0.1282 |
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
