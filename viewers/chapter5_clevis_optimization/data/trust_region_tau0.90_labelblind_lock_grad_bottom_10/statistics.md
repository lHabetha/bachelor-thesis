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
| All 200 starts | 0.5935 | 0.5000 |
| Oracle confirmed | 0.5935 | 0.5000 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.0000 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 66 | 66 | 0.3667 |
| nearest_splint_clearance | 134 | 134 | 134 | 0.7052 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 2.6200 |
| Median active coordinates | 3.0000 |
| Mean L1 distance | 0.8072 |
| Median L1 distance | 0.7300 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `leg_length` | 150 |
| `main_pin_length` | 149 |
| `overhang_span_y` | 103 |
| `main_hole_offset_from_open_end` | 61 |
| `splint_length` | 31 |
| `outer_span` | 30 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `grad_bottom_10` |
| Scenario kind | `gradient_ranked` |
| Starts with locks | 200 |
| Mean locked parameter count | 10.00 |
| Mean locked gradient mass | 0.3821 |
| Mean locked delta mass | 0.1815 |
| Paired baseline | `trust_region_tau0.90_labelblind` |
| Baseline oracle OK | 200 |
| Constrained oracle OK | 200 |
| Oracle success drop | 0 |
| Recoverability | 100.0% |
| Mean distance delta | -0.0320 |
| Median distance delta | -0.0000 |
| Same-or-better distance successes | 195 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `cross_hole_distance_from_free_end` | 200 |
| `cross_hole_radius` | 200 |
| `depth` | 200 |
| `leg_length` | 31 |
| `main_hole_offset_from_open_end` | 123 |
| `main_hole_radius` | 200 |
| `main_pin_length` | 33 |
| `main_pin_radius` | 200 |
| `outer_span` | 169 |
| `overhang_span_y` | 77 |
| `splint_length` | 167 |
| `splint_radius` | 200 |
| `wall_thickness` | 200 |
