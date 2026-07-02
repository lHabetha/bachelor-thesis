# Optimization Statistics: multifixed_penalty_direction_frontier_lam0_10_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 120 |
| Oracle confirmed | 132 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 80 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.5025 | 0.4013 |
| Oracle confirmed | 0.6131 | 0.5765 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.2883 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 29 | 36 | 0.3495 |
| nearest_splint_clearance | 134 | 91 | 96 | 0.5778 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 9.2350 |
| Median active coordinates | 10.0000 |
| Mean L1 distance | 1.1780 |
| Median L1 distance | 0.9459 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `cross_hole_distance_from_free_end` | 186 |
| `wall_thickness` | 186 |
| `main_pin_radius` | 185 |
| `depth` | 184 |
| `main_hole_radius` | 184 |
| `splint_radius` | 184 |
| `cross_hole_radius` | 183 |
| `outer_span` | 160 |
| `splint_length` | 153 |
| `main_hole_offset_from_open_end` | 113 |
| `overhang_span_y` | 72 |
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
| Mean locked delta mass | 0.5825 |
| Paired baseline | `ch5d20_multifixed_penalty_lam0_10_tau0.75` |
| Baseline oracle OK | 167 |
| Constrained oracle OK | 132 |
| Oracle success drop | 35 |
| Recoverability | 74.9% |
| Mean distance delta | 0.1644 |
| Median distance delta | 0.1383 |
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
