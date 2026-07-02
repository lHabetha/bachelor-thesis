# Optimization Statistics: adaptive_multistep_penalty_process_l0_l2_k1_v1

Model artifact: `row1_uncertainty_disagreement_B1000_T2500_best`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 192 |
| Oracle confirmed | 192 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 8 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 0.4098 | 0.4349 |
| Oracle confirmed | 0.3977 | 0.4059 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.6986 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 66 | 66 | 0.2662 |
| nearest_splint_clearance | 134 | 126 | 126 | 0.4805 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 1.1050 |
| Median active coordinates | 1.0000 |
| Mean L1 distance | 0.4278 |
| Median L1 distance | 0.4482 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `main_pin_length` | 115 |
| `leg_length` | 91 |
| `overhang_span_y` | 6 |
| `main_hole_offset_from_open_end` | 4 |
| `main_pin_radius` | 2 |
| `outer_span` | 1 |
| `splint_length` | 1 |
| `wall_thickness` | 1 |
