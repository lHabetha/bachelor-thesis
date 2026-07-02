# Optimization Statistics: probability_guided_coordinate_v1

Model artifact: `xgboost_best_task30_labelblind_rank1`

## Overall

| Metric | Value |
|--------|-------|
| Total starts | 200 |
| Surrogate success | 172 |
| Oracle confirmed | 200 |
| False successes | 0 |
| False success rate | 0.0% |
| No crossing (stuck) | 28 |
| Validity failures | 0 |

## Normalized Distances

| Group | Mean | Median |
|-------|------|--------|
| All 200 starts | 2.4360 | 2.0000 |
| Oracle confirmed | 2.4360 | 2.0000 |
| False success | 0.0000 | — |
| Stuck / no crossing | 1.9464 | — |

## By Blocked Subgroup

| Subgroup | Count | Surr. Success | Oracle OK | Mean Dist |
|----------|-------|---------------|-----------|-----------|
| nearest_inward_movement | 66 | 63 | 66 | 1.9015 |
| nearest_splint_clearance | 134 | 109 | 134 | 2.6993 |

## Sparse Edit Metrics

| Metric | Value |
|--------|-------|
| Active-coordinate epsilon | 1.0e-04 |
| Mean active coordinates | 1.0000 |
| Median active coordinates | 1.0000 |
| Mean L1 distance | 2.4360 |
| Median L1 distance | 2.0000 |

### Moved Parameter Frequency

| Parameter | Active Finals |
|-----------|---------------|
| `overhang_span_y` | 137 |
| `leg_length` | 48 |
| `main_pin_length` | 8 |
| `main_hole_offset_from_open_end` | 5 |
| `splint_length` | 2 |
