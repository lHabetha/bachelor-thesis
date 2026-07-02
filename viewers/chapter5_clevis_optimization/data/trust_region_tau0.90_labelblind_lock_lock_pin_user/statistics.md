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
| All 200 starts | 0.6810 | 1.0000 |
| Oracle confirmed | 0.6810 | 1.0000 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.0000 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 66 | 66 | 0.5136 |
| nearest_splint_clearance | 134 | 134 | 134 | 0.7634 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 5.2750 |
| Median active coordinates | 1.0000 |
| Mean L1 distance | 1.0096 |
| Median L1 distance | 1.0000 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `leg_length` | 148 |
| `overhang_span_y` | 147 |
| `main_hole_offset_from_open_end` | 97 |
| `cross_hole_distance_from_free_end` | 95 |
| `cross_hole_radius` | 95 |
| `outer_span` | 95 |
| `splint_radius` | 95 |
| `wall_thickness` | 95 |
| `depth` | 94 |
| `splint_length` | 94 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `lock_pin_user` |
| Scenario kind | `semantic_group` |
| Starts with locks | 200 |
| Mean locked parameter count | 3.00 |
| Mean locked gradient mass | 0.2706 |
| Mean locked delta mass | 0.4298 |
| Paired baseline | `trust_region_tau0.90_labelblind` |
| Baseline oracle OK | 200 |
| Constrained oracle OK | 200 |
| Oracle success drop | 0 |
| Recoverability | 100.0% |
| Mean distance delta | 0.0555 |
| Median distance delta | 0.0000 |
| Same-or-better distance successes | 156 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `main_hole_radius` | 200 |
| `main_pin_length` | 200 |
| `main_pin_radius` | 200 |
