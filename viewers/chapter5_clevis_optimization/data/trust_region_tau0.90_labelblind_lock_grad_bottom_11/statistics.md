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
| All 200 starts | 0.6265 | 0.5000 |
| Oracle confirmed | 0.6265 | 0.5000 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.0000 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 66 | 66 | 0.3742 |
| nearest_splint_clearance | 134 | 134 | 134 | 0.7507 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 1.9150 |
| Median active coordinates | 2.0000 |
| Mean L1 distance | 0.7578 |
| Median L1 distance | 0.6400 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 159 |
| `leg_length` | 151 |
| `overhang_span_y` | 42 |
| `main_hole_offset_from_open_end` | 31 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `grad_bottom_11` |
| Scenario kind | `gradient_ranked` |
| Starts with locks | 200 |
| Mean locked parameter count | 11.00 |
| Mean locked gradient mass | 0.5268 |
| Mean locked delta mass | 0.2922 |
| Paired baseline | `trust_region_tau0.90_labelblind` |
| Baseline oracle OK | 200 |
| Constrained oracle OK | 200 |
| Oracle success drop | 0 |
| Recoverability | 100.0% |
| Mean distance delta | 0.0010 |
| Median distance delta | -0.0000 |
| Same-or-better distance successes | 194 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `cross_hole_distance_from_free_end` | 200 |
| `cross_hole_radius` | 200 |
| `depth` | 200 |
| `leg_length` | 42 |
| `main_hole_offset_from_open_end` | 167 |
| `main_hole_radius` | 200 |
| `main_pin_length` | 33 |
| `main_pin_radius` | 200 |
| `outer_span` | 200 |
| `overhang_span_y` | 158 |
| `splint_length` | 200 |
| `splint_radius` | 200 |
| `wall_thickness` | 200 |
