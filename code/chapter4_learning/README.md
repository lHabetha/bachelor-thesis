# Chapter 4 Code — Label Efficiency And Model Comparison

Runnable learning code for Chapter 4 (`ch:labeling-learning`, `sec:mlp-training`,
`sec:arch-comparison`). Depends on released pools under
`datasets/chapter4_clevis_pools/` and formula labeling in
`code/chapter3_clevis_setup/`.

## Layout

```text
chapter4_learning/
  release_paths.py          # shared dataset/result paths
  mlp_active_learning/      # MLP + label-blind active learning
  graph_surrogates/         # MPNN/graph architecture screen
  tree_ensembles/           # RF/ET/HGB/XGB study
```

Frozen thesis summaries and figures live under
`results/chapter4_label_efficiency/`. New training runs write under
`results/chapter4_label_efficiency/mlp_runs/`, `graph_runs/`, or `tree_runs/`.

## Dependencies

Conda env `bachelor-thesis` (see [`docs/environment.md`](../../docs/environment.md)):
`torch`, `numpy`, `pandas`, `pyarrow`, `matplotlib`, `scikit-learn`, `xgboost`,
plus Chapter 3 modules on `PYTHONPATH`.

```bash
cd bachelor-thesis/code/chapter3_clevis_setup   # for Ch.3 smoke tests
cd bachelor-thesis/code/chapter4_learning/mlp_active_learning
```

## MLP + active learning

Primary experiment: `dense50k_v2_labelblind` (label-blind uncertainty /
diverse-uncertainty acquisition on the 50k pool).

### Reproduce thesis figures from frozen CSVs

```bash
cd code/chapter4_learning/mlp_active_learning
conda run -n bachelor-thesis python plot_label_efficiency.py
conda run -n bachelor-thesis python plot_al_advantage_heatmap.py
conda run -n bachelor-thesis python plot_combined_4panel_log.py
```

Outputs → `results/chapter4_label_efficiency/figures/` (BAC curves, heatmaps,
10k/22k/50k six-panel comparison).

### Rerun a single AL trajectory (optional)

Requires released pool + seed sets. Writes checkpoints under
`results/chapter4_label_efficiency/mlp_runs/data/trajectories/`.

```bash
conda run -n bachelor-thesis python run_trajectory.py \
  --row row1_uncertainty_disagreement --base-size 250 --seed-split 1
```

Full grid (112 trajectories): `run_grid.py --workers 12 --resume`
Random baseline: `run_baseline_random.py --replicates 5 --workers 12`
Aggregate trajectories → CSVs: `aggregate_results.py`

### Regenerate labeled pools (optional)

Uses Chapter 3 `smart_sampler` + formula oracle. **Default output** goes to
`datasets/chapter4_clevis_pools/` (use `--force` to overwrite).

```bash
conda run -n bachelor-thesis python build_pools_22k.py --workers 16 --force
conda run -n bachelor-thesis python build_pools_50k.py --workers 16 --force
conda run -n bachelor-thesis python build_pools_10k.py --workers 16 --force
conda run -n bachelor-thesis python build_pools_50k.py --write-seed-sets --seed-splits 8
```

Protocol JSON: `mlp_active_learning/configs/dense50k_v2_labelblind.json`.

## Graph surrogates

Architecture screen over three graph encoders. **Frozen** metrics and the
fixed-label manifest are in `results/chapter4_label_efficiency/task27_graphs/`.

Full architecture-screen replay additionally requires per-trajectory label
selection files that are not part of this release. Use the frozen CSVs for
thesis numbers; run `run_arch_screen.py` only after supplying trajectories
under `results/chapter4_label_efficiency/graph_runs/`.

```bash
cd code/chapter4_learning
conda run -n bachelor-thesis python -m graph_surrogates.aggregate_results   # if rerunning
```

## Tree ensembles

Label-blind tree baselines and active learning on the 50k pool.

```bash
cd code/chapter4_learning
conda run -n bachelor-thesis python -m tree_ensembles.plot_results
conda run -n bachelor-thesis python -m tree_ensembles.run_random_baseline --workers 8
conda run -n bachelor-thesis python -m tree_ensembles.aggregate_results
```

Frozen summaries: `results/chapter4_label_efficiency/task30_trees/`.
Protocol: `tree_ensembles/configs/protocol_v2_labelblind.json`.

## Data paths

- Pool inputs: `datasets/chapter4_clevis_pools/pool_*/candidate_pool.parquet`
- Holdout: `datasets/chapter4_clevis_pools/holdout/holdout.parquet`
