# Optimization Statistics: random_sphere_coordinate_shrink_128_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 190 |
| Oracle confirmed | 193 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 10 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.5892 | 0.5784 |
| Oracle confirmed | 0.5614 | 0.5488 |
| False success | 0.0000 | — |
| Stuck / no crossing | 1.5500 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 65 | 65 | 0.4386 |
| nearest_splint_clearance | 134 | 125 | 128 | 0.6634 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 4.5150 |
| Median active coordinates | 4.0000 |
| Mean L1 distance | 1.0485 |
| Median L1 distance | 0.8748 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `leg_length` | 146 |
| `main_hole_offset_from_open_end` | 120 |
| `overhang_span_y` | 120 |
| `splint_length` | 101 |
| `splint_radius` | 90 |
| `cross_hole_radius` | 87 |
| `cross_hole_distance_from_free_end` | 81 |
| `depth` | 75 |
| `outer_span` | 59 |
| `wall_thickness` | 24 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `lock_pin_user` |
| Scenario kind | `semantic_group` |
| Starts with locks | 200 |
| Mean locked parameter count | 3.00 |
| Mean locked gradient mass | 0.2706 |
| Mean locked delta mass | 0.3173 |
| Paired baseline | `ch5_sphere_shrink_128_matched_tau0.90` |
| Baseline oracle OK | 185 |
| Constrained oracle OK | 193 |
| Oracle success drop | -8 |
| Recoverability | 98.9% |
| Mean distance delta | 0.0239 |
| Median distance delta | 0.0213 |
| Same-or-better distance successes | 66 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `main_hole_radius` | 200 |
| `main_pin_length` | 200 |
| `main_pin_radius` | 200 |
