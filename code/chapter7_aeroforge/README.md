# Chapter 7 Code — AeroForge Overlap Transfer

Thesis anchor: Chapter 7 (`ch:aeroforge-transfer`, §7.1–§7.6).

---

## Environment

Requires the **`bachelor-thesis` conda env** from the root `environment.yml` (CadQuery/OCP, PyTorch, scikit-learn) for overlap
labeling, CAD verification, and repair runs. Quick-sim labeling additionally needs
**AeroSandbox** (upstream AeroForge dependency).

### Upstream AeroForge (not bundled)

Clone the AeroForge repo as a sibling of `bachelor-thesis/`:

```sh
# From the parent directory of bachelor-thesis/
git clone https://github.com/IDEALLab/aeroforge.git aeroforge
```

Default path: `../aeroforge` relative to `bachelor-thesis/`. Override with:

```sh
export AEROFORGE_ROOT=/path/to/aeroforge
```

This repository does **not** bundle the upstream repo. The upstream source is
[`IDEALLab/aeroforge`](https://github.com/IDEALLab/aeroforge) (branch `main`;
the thesis work used commit `a88f0af`). Chapter 7 VLM quick-sim labels are
included for thesis reproducibility with attribution to the IDEAL Lab AeroForge
workflow and upstream AeroSandbox-derived tooling.

---

## Running from the public repo

```sh
cd bachelor-thesis
export PYTHONPATH=code
conda activate bachelor-thesis
```

All modules: `python -m chapter7_aeroforge.<subpackage>.<module>`

Path constants live in `release_paths.py` (datasets → `datasets/chapter7_aeroforge_adv/`,
checkpoints → `results/chapter7_aeroforge/checkpoints/`, repair runs →
`results/chapter7_aeroforge/runs/`).

---

## Package layout

| Subpackage | Modules | Role |
|------------|---------|------|
| `release_paths.py` | — | Public dataset/results/checkpoint paths + `subprocess_env()` |
| `overlap_search/` | `core`, `sampler`, `adv_dataset_generator`, `discovery_sampler`, `generate_adv_dataset_files`, `worker`, `label_forever` | 43-key ADV schema, pool generation, CadQuery overlap labeling |
| `quicksim/` | `label_quicksim_forever` | VLM quick-sim labeler for performance MLP |
| `ml/` | `features`, `models`, `sim_features`, `run_100k_model_grid`, `plot_100k_model_grid`, `export_optimizer_model`, `train_performance_mlp` | Surrogate training, CV grid, performance MLP |
| `optimizer/` | `model_io`, `optimizers`, `perf_surrogate`, `verify`, `workbench`, `run_sweep`, `run_aero_sweep`, `summarize_runs`, `summarize_aero_runs`, `build_normalization`, `preflight`, `quicksim_eval`, `benchmark_sets/select_overlap_benchmark` | Repair optimizers, table export, benchmark prep |

Interactive repair viewer: [`../../viewers/chapter7_aeroforge_repair/`](../../viewers/chapter7_aeroforge_repair/).

---

## Example commands

### Regenerate learning-curve figure (`fig:aeroforge-model-grid`)

```sh
python -m chapter7_aeroforge.ml.plot_100k_model_grid
```

Reads frozen `results/chapter7_aeroforge/comparisons/model_grid_100k_v1/results_agg.csv`.

### Export repair/grid tables (`tab:aeroforge-overlap-repair`, grid table)

Requires repair run JSON under `results/chapter7_aeroforge/runs/` (or use frozen
tables under `results/chapter7_aeroforge/tables/`; viewer JSON under
`viewers/chapter7_aeroforge_repair/data/`):

```sh
python -m chapter7_aeroforge.optimizer.summarize_runs --grid-id repair_grid_100k_v2
```

### Export aero-preserve tables (`tab:aeroforge-aero-preserve`)

```sh
python -m chapter7_aeroforge.optimizer.summarize_aero_runs --grid-id aero_preserve_v1
```

### Preflight before a full repair grid rerun

```sh
python -m chapter7_aeroforge.optimizer.preflight
```

Rebuilds normalization bounds + clip-checks the 100-start benchmark.

### Single repair subgroup (headline variant A, one optimizer)

```sh
conda run -n bachelor-thesis python -m chapter7_aeroforge.optimizer.workbench \
  --variant a --optimizer receding_multistep_gradient --workers 14 --seed 12345
```

### Overlap-label one sample (requires `aeroforge/` checkout)

```sh
AEROFORGE_OCCT_THREADS=1 conda run -n bachelor-thesis python -c "
from chapter7_aeroforge.overlap_search.core import label_sample, build_airframe
import json
from chapter7_aeroforge.release_paths import ADV_DIR
row = json.loads((ADV_DIR / '000000.json').read_text())
print(label_sample(row['adv']))
"
```

---

## What this package covers

- ADV schema, family-weighted sampling, 100k pool generator
- CadQuery wing–tail overlap labeler (1 mm³ threshold)
- Quick-sim / VLM feature extraction for performance MLP
- Four final overlap surrogate variants + 100k CV grid runner
- Performance MLP trainer (L/D, C_D0, C_L,VLM, C_m,VLM)
- Six repair optimizer families + `all` meta; six model configs A–F
- Three performance-aware optimizers + `all_aero` meta
- Table/figure export (`summarize_runs`, `summarize_aero_runs`, `plot_100k_model_grid`)

---

## Notes

- A full 42-subgroup repair grid plus 15 aero runs produces **large** output
  (~150–300 MB JSON without meshes). Rerunning sweeps writes to
  `results/chapter7_aeroforge/runs/`; the thesis runs are already available in
  the viewer under `viewers/chapter7_aeroforge_repair/data/`.
- `labels_sim/` quick-sim payloads are released with AeroForge/AeroSandbox
  attribution; keep that provenance visible if reorganizing the folder.
- Subprocess sweep runners set `PYTHONPATH=code` automatically when launched from
  `bachelor-thesis/` via `subprocess_env()`.
