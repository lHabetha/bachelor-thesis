# Schema Documentation

## Trajectory Schema v1

Each run produces `trajectories.json` — an array with one object per active benchmark start.

### Per-Start Record

| Field | Type | Description |
|-------|------|-------------|
| schema_version | string | Always "v1" |
| start_id | string | Sequential benchmark identifier such as "blocked_000" |
| optimizer_id | string | Algorithm identifier |
| model_id | string | Model artifact directory name |
| status | string | "surrogate_success" / "no_surrogate_crossing" / "no_valid_step" / "no_gradient" |
| start_params | object | 13 named parameters at start |
| final_params | object | 13 named parameters at selected step |
| start_probability | float | MLP P(assemblable) at start |
| final_probability | float | MLP P(assemblable) at final |
| threshold | float | tau used by this run |
| valid_final | bool | Whether final params pass validity |
| oracle_label | int | 0 or 1 from exact formula |
| oracle_reason | string | Formula reason at final params |
| normalized_distance | float | L2 distance in normalized space |
| n_steps | int | Total optimizer steps recorded |
| n_evaluations | int | Total model evaluations |
| stop_reason | string | Why the optimizer stopped |
| steps | array | Per-step records |
| constraint | object, optional | Parameter-lock metadata for constrained runs |

`no_valid_step` means the optimizer did not find any valid non-start candidate
worth reporting as the final design. Invalid probes may still appear in `steps`
for auditability, but the selected final record must be the valid start at
normalized distance `0.0`.

### Per-Step Record

| Field | Type | Description |
|-------|------|-------------|
| step_idx | int | Index in the step sequence |
| params | object | 13 named parameters |
| probability | float | MLP P(assemblable) |
| gradient_direction | array(13) | Unit direction in benchmark-normalized coordinates; for gradient methods this is the MLP ascent direction, for axis/random methods it is the sampled normalized-space search direction |
| step_magnitude | float | Step size in benchmark-normalized coordinates |
| valid | bool | Passes validity checks |
| oracle_label | int | Formula oracle result |
| oracle_reason | string | Formula reason |
| normalized_distance | float | Distance from start; for one-shot normalized-space steps this equals `step_magnitude` |
| is_selected | bool | This step is the final selection |
| is_terminated | bool | Optimization terminated here |
| stop_reason | string | Reason if terminated |

For constrained parameter-lock runs, every `params` object in every step must
keep each locked parameter exactly equal to the corresponding `start_params`
value. Any non-null `gradient_direction` must also have zero components for the
locked parameter indices.

### Optional Constraint Record

The same structure may appear in `trajectories.json`, `viewer_data.json`, and
lock-aware statistics:

| Field | Type | Description |
|-------|------|-------------|
| constraint_id | string | Lock scenario, e.g. `grad_top_1`, `delta_top_2`, `lock_roof_path`, `random_1_seed0` |
| scenario_kind | string | `gradient_ranked`, `baseline_delta_ranked`, `semantic_group`, `random_control`, or `manual` |
| locked_indices | array(int) | Frozen parameter indices in canonical feature order |
| locked_names | array(string) | Frozen parameter names |
| locked_count | int | Number of frozen parameters |
| locked_gradient_mass | float or null | Fraction of absolute normalized start-gradient mass locked |
| locked_delta_mass | float or null | Fraction of unconstrained normalized movement locked, if a baseline exists |
| baseline_run_id | string or null | Paired unconstrained baseline run |

## Viewer Data Schema v1

Each run produces `viewer_data.json` — an array with one object per active benchmark start for the browser viewer.

### Per-Assembly Viewer Record

| Field | Type | Description |
|-------|------|-------------|
| start_id | string | Benchmark start identifier |
| subgroup | string | Blocked subgroup classification |
| status | string | Optimizer outcome |
| oracle_label | int | Final oracle verdict |
| oracle_reason | string | Final oracle reason |
| normalized_distance | float | Final distance |
| start_probability | float | Starting P(assemblable) |
| final_probability | float | Final P(assemblable) |
| threshold | float | tau value |
| stop_reason | string | Optimizer stop reason |
| blocked_explanation | string | Human-readable blocked reason |
| frames | array(50) | Exactly 50 viewer frames |
| constraint | object, optional | Parameter-lock metadata for constrained runs |

### Per-Frame Record

| Field | Type | Description |
|-------|------|-------------|
| frame_idx | int | 0-49 |
| params | object | 13 named parameters |
| probability | float | MLP probability |
| gradient_direction | array(13) or null | Unit search direction in benchmark-normalized coordinates |
| step_magnitude | float or null | Step size in benchmark-normalized coordinates |
| normalized_distance | float | Distance from start |
| valid | bool | Validity status |
| oracle_label | int | Oracle verdict |
| oracle_reason | string | Oracle reason |
| terminated | bool | True if optimizer is done |
| termination_frame | int or null | Frame where termination happened |
| stop_reason | string or null | Reason for termination |

## Run Manifest Schema

Each run produces `manifest.json`:

| Field | Type | Description |
|-------|------|-------------|
| schema_version | string | "v1" |
| run_id | string | Unique run identifier |
| optimizer_id | string | Algorithm used |
| model_id | string | Model artifact |
| model_dir | string | Path to model |
| benchmark_set | string | Active benchmark identifier, e.g. "blocked_200_v1" |
| tau | float | Success threshold |
| n_starts | int | Number of starts in the active benchmark |
| wall_time_s | float | Total runtime |
| timestamp | string | ISO timestamp |
| results_summary | object | Headline metrics |
| constraint_id | string or null | Parameter-lock scenario ID for constrained runs |
| locked_params | array(string) | Manual lock list, if supplied |
| baseline_run_id | string or null | Paired unconstrained baseline run |
| constraint | object, optional | Run-level lock metadata |

## Parameter-Lock Study Artifacts

`constraint_studies/parameter_locks_v1/` contains suite-level robustness outputs:

| File | Description |
|------|-------------|
| scenario_registry.json | Deterministic definitions for every lock scenario |
| per_run_summary.csv | One row per constrained run with paired baseline deltas |
| per_start_results.csv | One row per constrained start result |
| parameter_criticality.csv | Parameter-level success-drop and distance-delta ranking |
| robustness_summary.md | Viewer-readable Markdown summary |

## Optional Sparse Metrics

Runs produced after the sparse-norm batch may include `sparse_metrics` in
`statistics.json`:

| Field | Type | Description |
|-------|------|-------------|
| active_epsilon | float | Normalized displacement threshold for counting active coordinates |
| active_coordinate_count_mean | float | Mean number of final parameters with `abs(delta_norm) > active_epsilon` |
| active_coordinate_count_median | float | Median active-coordinate count |
| l1_distance_mean | float | Mean normalized L1 distance from start to final |
| l1_distance_median | float | Median normalized L1 distance |
| moved_parameter_counts | object | Per-parameter count of active final changes |

The existing `normalized_distance` field remains the benchmark-normalized L2
distance for backwards compatibility and viewer coloring.

## Optimizer Documentation Template

Every optimizer must have `algorithm.md` with these sections:

1. **Intuition** — Plain-language explanation
2. **Configuration** — Table of parameters, tau, stopping rule
3. **Inputs And Outputs** — Exact file references
4. **Mathematical Description** — Objective, gradient, search rule
5. **Failure Modes** — Expected failure patterns
6. **Thesis Interpretation** — Scientific question answered
