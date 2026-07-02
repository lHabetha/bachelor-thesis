# Optimization Statistics: coordinate_axis_bracket_v1

Model artifact: `hist_gradient_boosting_best_task30_labelblind_rank1`

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
| All 200 starts | 1.4170 | 1.0000 |
| Oracle confirmed | 1.4170 | 1.0000 |
| False success | 0.0000 | — |
| Stuck / no crossing | 0.0000 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 66 | 66 | 1.0455 |
| nearest_splint_clearance | 134 | 134 | 134 | 1.6000 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 1.0000 |
| Median active coordinates | 1.0000 |
| Mean L1 distance | 1.4170 |
| Median L1 distance | 1.0000 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `leg_length` | 79 |
| `overhang_span_y` | 66 |
| `main_pin_length` | 41 |
| `main_hole_offset_from_open_end` | 13 |
| `splint_length` | 1 |
