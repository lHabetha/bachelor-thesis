# Optimization Statistics: trust_region_hybrid_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 200 |
| Oracle confirmed | 200 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 0 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.7642 | 0.5000 |
| Oracle confirmed | 0.7642 | 0.5000 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.0000 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 66 | 66 | 0.3841 |
| nearest_splint_clearance | 134 | 134 | 134 | 0.9515 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 3.4200 |
| Median active coordinates | 1.0000 |
| Mean L1 distance | 0.9695 |
| Median L1 distance | 0.9381 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 187 |
| `splint_length` | 93 |
| `cross_hole_distance_from_free_end` | 81 |
| `cross_hole_radius` | 81 |
| `main_pin_radius` | 81 |
| `splint_radius` | 81 |
| `main_hole_radius` | 80 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `lock_main_bracket_user` |
| Scenario kind | `semantic_group` |
| Starts with locks | 200 |
| Mean locked parameter count | 6.00 |
| Mean locked gradient mass | 0.6086 |
| Mean locked delta mass | 0.5109 |
| Paired baseline | `trust_region_tau0.90_labelblind` |
| Baseline oracle OK | 200 |
| Constrained oracle OK | 200 |
| Oracle success drop | 0 |
| Recoverability | 100.0% |
| Mean distance delta | 0.1387 |
| Median distance delta | 0.0000 |
| Same-or-better distance successes | 143 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `depth` | 200 |
| `leg_length` | 200 |
| `main_hole_offset_from_open_end` | 200 |
| `outer_span` | 200 |
| `overhang_span_y` | 200 |
| `wall_thickness` | 200 |
