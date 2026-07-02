# Chapter 6 Code — Overlap-Aware Extension

Chapter 6 code for the clevis overlap-aware extension.

The package covers:

- analytic voxel overlap labels and relaxed-overlap sampling;
- overlap-regression training/evaluation utilities;
- strict overlap repair (`ch64`) with MLP, hybrid, and direct controls;
- strict overlap/assemblability alternating pipeline (`ch65`);
- thesis table/figure and viewer export scripts.

Run commands from the `bachelor-thesis/` repository root with `PYTHONPATH=code`:

```sh
PYTHONPATH=code python -m chapter6_overlap.<module>
```

---

## Paths

`release_paths.py` maps all inputs and outputs inside this repository:

| Artifact | Public path |
|----------|-------------|
| Pools / labels / benchmarks | `datasets/chapter6_overlap_clevis/` |
| Checkpoints | `results/chapter6_overlap/checkpoints/` |
| Tables / figures / summaries | `results/chapter6_overlap/` |
| Viewer exports | `viewers/chapter6_overlap_repair/data/` |
| Chapter 3 geometry / oracle | `code/chapter3_clevis_setup/` |
| Chapter 5 assemblability model / optimizer | `code/chapter5_optimization/`, `results/chapter5_optimization/checkpoints/` |

`paths.py` is retained as a compatibility shim; it points to the same folders.

---

## Included modules

### Labeling and data

- `sampler.py` — relaxed-validity parameter sampling.
- `analytic_overlap.py` — analytic voxel overlap labeler.
- `geometry.py` — optional CAD/mesh export for overlap fixtures.
- `label_cache.py` — on-demand labels, writing caches under `results/chapter6_overlap/labels_cache/`.
- `pool.py` — generate new relaxed pools under `datasets/chapter6_overlap_clevis/<name>/`.
- `calibrate_analytic.py`, `calibrate_labels.py` — calibration helpers for Fig. 6.1.

### Regression and active learning

- `regression_architectures.py`, `models.py`, `regression_metrics.py`.
- `task36_arch_screen.py` — architecture screen for Fig. 6.2.
- `task36_al_strategy_probe.py` — volume-head acquisition probe for Fig. 6.3 / Tab. 6.3.
- `multitask_overlap_model.py`, `multitask_al_probe.py` — binary-head multitask probe for Fig. 6.4 / Tab. 6.4.
- `task36_diagnostics.py` — holdout diagnostics cited in §6.3.
- `export_models.py` — lightweight checkpoint inspection helper.

### Repair and pipeline

- `strict_repair_ch64.py` — Chapter 6.4 strict overlap repair methods.
- `ch64_run_matrix.py` — command generator for the ch64 method matrix.
- `export_ch64_tables.py` — exports Tab. 6.6 and Fig. 6.5 source artifacts.
- `freeze_ch64_benchmark.py` — benchmark-freeze helper (the frozen benchmark is already copied).
- `strict_zigzag_ch65.py` — alternating overlap/assemblability controller for Tab. 6.7.
- `select_strict_blocked_starts.py` — ch65 benchmark selection helper.
- `export_ch64_viewer.py`, `export_ch65_zigzag_viewer.py` — viewer run exporters.

### Support files

- `_public_helpers.py` — small shared helper functions for the ch64/ch65 scripts.
- `CONTROLLED_OVERLAP_DESIGN_SPACE.md` — design-space relaxation notes.
- `configs/relaxed_sampler_v1.json`, `configs/sdf_label_v1.json` — frozen config references.

---

## Example commands

Use the `bachelor-thesis` environment (or equivalent Python with numpy, pandas,
torch, scikit-learn, matplotlib, cadquery, trimesh):

```sh
# Check final checkpoints
PYTHONPATH=code python -m chapter6_overlap.export_models

# Recompute the 12-assembly calibration figure/table
PYTHONPATH=code python -m chapter6_overlap.calibrate_analytic

# Architecture screen -> Fig. 6.2
PYTHONPATH=code python -m chapter6_overlap.task36_arch_screen

# Volume-head AL probe -> Fig. 6.3 / Tab. 6.3
PYTHONPATH=code python -m chapter6_overlap.task36_al_strategy_probe

# Binary-head AL probe -> Fig. 6.4 / Tab. 6.4
PYTHONPATH=code python -m chapter6_overlap.multitask_al_probe

# Generate ch64 repair commands (does not run them)
PYTHONPATH=code python -m chapter6_overlap.ch64_run_matrix --phase timing --n-starts 1

# Re-run one ch64 method on a small slice
PYTHONPATH=code python -m chapter6_overlap.strict_repair_ch64   --method mlp --model-kind v3 --run-id smoke_mlp_v3 --n-starts 1 --start-offset 0

# Re-export ch64 tables/figure from existing public repair summaries
PYTHONPATH=code python -m chapter6_overlap.export_ch64_tables

# Re-run the ch65 alternating pipeline
PYTHONPATH=code python -m chapter6_overlap.strict_zigzag_ch65 --variant hybrid_lite
```

Full AL and repair reruns can take hours; the release also ships frozen datasets,
checkpoints, tables, figures, and per-start results under `results/chapter6_overlap/`.

