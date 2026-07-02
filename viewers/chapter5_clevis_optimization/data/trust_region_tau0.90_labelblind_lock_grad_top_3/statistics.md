# Optimization Statistics: trust_region_hybrid_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 191 |
| Oracle confirmed | 195 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 9 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 1.4810 | 1.0000 |
| Oracle confirmed | 1.4062 | 1.0000 |
| False success | 0.0000 | — |
| Stuck / no crossing | 4.0000 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 65 | 65 | 1.1955 |
| nearest_splint_clearance | 134 | 126 | 130 | 1.6216 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 4.1950 |
| Median active coordinates | 1.0000 |
| Mean L1 distance | 2.0288 |
| Median L1 distance | 2.0000 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `outer_span` | 97 |
| `wall_thickness` | 91 |
| `cross_hole_distance_from_free_end` | 80 |
| `main_pin_radius` | 72 |
| `cross_hole_radius` | 71 |
| `depth` | 71 |
| `main_hole_radius` | 71 |
| `splint_radius` | 71 |
| `overhang_span_y` | 60 |
| `splint_length` | 60 |
| `main_hole_offset_from_open_end` | 47 |
| `main_pin_length` | 27 |
| `leg_length` | 21 |

## Parameter Locks

| Metric | Value |
|--------|-------|
| Constraint ID | `grad_top_3` |
| Scenario kind | `gradient_ranked` |
| Starts with locks | 200 |
| Mean locked parameter count | 3.00 |
| Mean locked gradient mass | 0.6179 |
| Mean locked delta mass | 0.8185 |
| Paired baseline | `trust_region_tau0.90_labelblind` |
| Baseline oracle OK | 200 |
| Constrained oracle OK | 195 |
| Oracle success drop | 5 |
| Recoverability | 97.5% |
| Mean distance delta | 0.8555 |
| Median distance delta | 0.5000 |
| Same-or-better distance successes | 41 |

### Locked Parameter Frequency

| Parameter | Locked Starts |
|-----------|---------------|
| `leg_length` | 169 |
| `main_hole_offset_from_open_end` | 77 |
| `main_pin_length` | 167 |
| `outer_span` | 31 |
| `overhang_span_y` | 123 |
| `splint_length` | 33 |
