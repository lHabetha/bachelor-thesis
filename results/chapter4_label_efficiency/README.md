# Chapter 4 Results — Label Efficiency And Architecture Comparison

Frozen summaries and regenerable figures for Chapter 4
(`ch:labeling-learning`, `sec:mlp-training`, `sec:arch-comparison`).

## Thesis figures

| Public file | Thesis anchor | Source |
|-------------|---------------|--------|
| `figures/task22_dense50k_bac_curves_combined.png` | `fig:task22-label-efficiency` | `plot_label_efficiency.py` |
| `figures/task22_al_advantage_heatmap.png` | `fig:task22-al-advantage` | `plot_al_advantage_heatmap.py` |
| `figures/task22_poolsize_comparison_6panel_log.png` | `fig:task22-pool-size-comparison` | `plot_combined_4panel_log.py` |
| `figures/task22_dense50k_paired_delta_heatmap.png` | `fig:task22-paired-delta` | `plot_label_efficiency.py` |
| `figures/task27_architecture_bac_compute.png` | `fig:task27-architecture` | frozen copy |
| `figures/task30_random_baseline_bac_linear.png` | `fig:task30-random-tree-comparison` | `tree_ensembles.plot_results` |

Additional plot outputs from rerunning scripts may appear alongside these
(e.g. `bac_curves_combined.png`, `combined_6panel_log.png`).

## Frozen aggregate CSVs

### `mlp_dense50k_labelblind/` (thesis-primary MLP study)

| File | Purpose |
|------|---------|
| `summary_stats.csv` | Mean BAC by row / base size / total labels |
| `trajectory_metrics.csv` | Per-trajectory milestones |
| `baseline_metrics.csv` | Random baseline replicates |
| `paired_deltas.csv` | Row1 vs Row2 paired comparison |

### `mlp_dense10k_labelblind/` (10k pool-size comparison)

Same four CSV types.

### `mlp_pool_22k/` (22k pool-size comparison)

| File | Purpose |
|------|---------|
| `plot_series_agg.csv` | Aggregated BAC series |
| `baseline_random_summary.csv` | Random baseline |
| `paired_deltas.csv` | Paired deltas |

### `task27_graphs/`

| File | Purpose |
|------|---------|
| `architecture_metrics.csv` | Per-fit BAC and timing (845 fits) |
| `architecture_summary_by_group.csv` | Grouped means for figure |
| `fixed_label_sets_manifest.json` | Exact training subsets (~8.4 MB) |

### `task30_trees/`

| File | Purpose |
|------|---------|
| `summary_by_group.csv` | Table / plot aggregation (`tab:task30-tree-surrogates`) |
| `all_metrics.csv` | Full metric dump |

## Writable rerun directories (empty until user reruns code)

- `mlp_runs/` — new AL trajectories and baselines
- `graph_runs/` — architecture-screen checkpoints (optional)
- `tree_runs/` — tree baseline / AL outputs (optional)

## Regenerate figures

See `code/chapter4_learning/README.md` for plot commands.
