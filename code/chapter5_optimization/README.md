# Chapter 5 Code — Surrogate-Based Optimization

Runnable optimizer workbench for Chapter 5 (`ch:optimization`, `sec:optimizer-results`,
`sec:parameter-locks`, `sec:distance-sparsity`).

Depends on:

- `datasets/chapter5_blocked200/` — 200 blocked benchmark starts
- `results/chapter5_optimization/checkpoints/` — MLP + tree surrogate weights
- `code/chapter3_clevis_setup/` — formula oracle for ground-truth verification
- `code/chapter4_learning/mlp_active_learning/` — MLP64 training/gradient code

---

## Layout

```text
chapter5_optimization/
  release_paths.py       # public dataset/checkpoint/result paths
  run_workbench.py       # main optimizer driver
  export_model.py        # re-export MLP checkpoint from Ch.4 trajectories
  generate_benchmark.py  # rebuild blocked_200_v1
  compare_runs.py        # aggregate run statistics → CSV
  parameter_lock_study.py
  analyze_task34_adaptive.py
  shared/                # interface, oracle, model loading, locks
  optimizers/            # 20 thesis-facing optimizer plugins + commons
  schemas/README.md      # run output schema notes
```

Runtime outputs write to `results/chapter5_optimization/runs/<run_id>/`.

---

## Dependencies

Conda env `bachelor-thesis`: `torch`, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`.

```bash
cd bachelor-thesis/code/chapter5_optimization
```

---

## Run a single optimizer (smoke test)

Full 200-start runs take minutes to hours depending on method. Example:

```bash
conda run -n bachelor-thesis python run_workbench.py \
  --optimizer one_shot_gradient_bracket_v1 \
  --tau 0.75 \
  --run-id smoke_one_shot_tau0.75
```

Outputs under `results/chapter5_optimization/runs/smoke_one_shot_tau0.75/`:

- `manifest.json`, `statistics.json`, `statistics.md`, `algorithm.md`
- `viewer_data.json`, `trajectories.json`

Default model checkpoint: `row1_uncertainty_disagreement_B1000_T2500_best`.
Override with `--model-dir ../../results/chapter5_optimization/checkpoints/<model_id>`.

Tree-control example:

```bash
conda run -n bachelor-thesis python run_workbench.py \
  --optimizer probability_guided_coordinate_v1 \
  --tau 0.90 \
  --model-dir ../../results/chapter5_optimization/checkpoints/xgboost_best_task30_labelblind_rank1 \
  --run-id smoke_xgb_probcoord_tau0.90
```

---

## Thesis-facing optimizer IDs

| Family | `--optimizer` values shipped |
|--------|------------------------------|
| One-shot gradient | `one_shot_gradient_bracket_v1` |
| Multi-fixed-step | `receding_gradient_multiscale_v1`, `multifixed_penalty_direction_frontier_lam0_{10,20,30,50}_v1` |
| Adaptive multistep | `adaptive_multistep_gradient_v1`, `adaptive_multistep_penalty_lam0_{05,10,20,30,50}_v1` |
| Coordinate-axis | `coordinate_axis_bracket_v1` |
| Trust region | `trust_region_hybrid_v1`, `trust_region_no_gradient_v1`, `trust_region_mixed_l2_l0_40_60_v1` |
| Random sphere | `random_sphere_expanding_v1`, `random_sphere_expanding_{32,128,256}_v1`, `random_sphere_coordinate_shrink_{32,64,128,256}_v1`, `random_sphere_coordinate_shrink_128_sparse_l0_l2_v1` |
| Tree control | `probability_guided_coordinate_v1` (+ coordinate/sphere with tree `--model-dir`) |
| Sparse proximity | `adaptive_multistep_penalty_process_l0_l2_k1_v1` |

Random-sphere 64-direction runs use `random_sphere_expanding_v1` with matched-seed CLI options.

---

## Aggregate and export scripts

```bash
# Summarize completed runs → results/chapter5_optimization/comparisons/
conda run -n bachelor-thesis python compare_runs.py --run-ids smoke_one_shot_tau0.75

# Chapter 5 adaptive table CSVs (reads runs/ + writes comparisons/)
conda run -n bachelor-thesis python analyze_task34_adaptive.py

# Rebuild benchmark (overwrites datasets/chapter5_blocked200/ with --force)
conda run -n bachelor-thesis python generate_benchmark.py --force
```

`export_model.py` retrains the fixed Ch5 MLP from Chapter 4 pool trajectories.
It requires active-learning trajectory selection files under
`results/chapter4_label_efficiency/mlp_runs/` that are not part of this
release — use the shipped checkpoint for thesis reproduction.

---

## Key paths

- Oracle: `shared/oracle.py` (uses `chapter3_clevis_setup`)
- Model loading: `shared/model_utils.py` (uses `chapter4_learning/mlp_active_learning/lib`)
- Checkpoints: `results/chapter5_optimization/checkpoints/`
- Benchmark: `datasets/chapter5_blocked200/blocked_200_v1.*`
- Viewer allowlist: `release_manifests/ch5_allowlist.json`
