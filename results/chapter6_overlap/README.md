# Chapter 6 Results — Overlap-Aware Extension

Frozen thesis tables, figures, probe summaries, and repair run aggregates.

Viewer run artifacts live under `../../viewers/chapter6_overlap_repair/data/`.

---

## Layout

```
results/chapter6_overlap/
├── checkpoints/          ← overlap surrogate weights (see checkpoints/README.md)
├── tables/               ← thesis CSV aggregates
├── figures/              ← thesis PNG copies
├── comparisons/          ← AL probe + architecture screen summaries
├── repair_ch64/          ← merged per-start results (8 thesis methods)
└── zigzag_ch65/          ← headline alternating-pipeline run
```

---

## Tables (`tables/`)

| File | Thesis anchor |
|------|---------------|
| `calibration_rows.csv` | `fig:task25-calibration` |
| `task36_al_strategy_probe_agg.csv` | `tab:overlap-regression-al-volume` |
| `multitask_al_probe_agg.csv` | `tab:overlap-regression-al-binary` |
| `ch64_repair_summary_v1.csv` | `tab:task25-strict-repair` |
| `ch64_repair_per_start_v1.csv` | `fig:task25-overlap-reduction`, `tab:overlap-stall-magnitude` |
| `strict_zigzag_ch65_summary.csv` | `tab:task25-strict-zigzag` |

---

## Figures (`figures/`)

| File | Thesis label |
|------|--------------|
| `calibration_error_speed.png` | `fig:task25-calibration` |
| `task25_regression_architecture_comparison_v3.png` | `fig:task25-regression-architectures` |
| `task25_al_strategy_probe_combined.png` | `fig:task25-regression-al-volume` |
| `task25_multitask_al_probe_combined.png` | `fig:task25-regression-al-binary` |
| `task25_ch64_overlap_reduction.png` | `fig:task25-overlap-reduction` |

These PNGs are the public frozen figure exports; the PDF is the authority.

---

## Repair aggregates

### `repair_ch64/` (8 thesis methods)

Merged `results.json` (50 starts each):

- `hybrid_lite__v3__tauvol0` (50/50)
- `hybrid_lite_reduced_calls__v3__tauvol0` (35/50)
- `finish_line__v3__tauvol0` (12/50)
- `mlp__v3__tauvol0` (12/50)
- `hybrid_lite__multitask__taubin0.1` (50/50)
- `mlp__multitask__taubin0.1` (11/50)
- `random_direct__none` (40/50)
- `axis_direct__none` (23/50)

### `zigzag_ch65/strict_zigzag_ch65_hybrid_lite_v3/`

Headline §6.5 run: 50/50 strict success. Includes `results.json`, `summary.json`,
`summary.csv`, `cycle_log.csv`.

---

## Comparisons (`comparisons/`)

Probe and architecture screen run summaries (JSON + architecture metrics CSV).
