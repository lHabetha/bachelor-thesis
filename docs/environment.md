# Environment Notes

How to run the release artifacts on a fresh machine. Development used
**Python 3.11** on macOS.

## Recommended setup

```bash
conda env create -f environment.yml
conda activate bachelor-thesis
```

The root [`../environment.yml`](../environment.yml) is the recommended full
environment for the public release. It mirrors the thesis development stack at
the package level and includes CadQuery, PyTorch, XGBoost, LightGBM, AeroSandbox,
and the geometry dependencies used by smoke tests and viewers.

[`../requirements.txt`](../requirements.txt) is provided as a pip reference, but
CadQuery and PyTorch are platform-sensitive. Prefer the conda environment file
for full CAD/render reproducibility.

Clone this repository, then always work from its root:

```bash
cd bachelor-thesis
export PYTHONPATH=code
```

## Per-chapter requirements

### Chapter 3 — clevis setup

| Package | Used for |
|---------|----------|
| `cadquery`, `numpy`, `scipy`, `trimesh`, `manifold3d` | Geometry build, mesh overlap |

Smoke test:

```bash
python -m chapter3_clevis_setup.build_and_label --skip-mesh --stream-demo 3
```

### Chapter 4 — learning

| Package | Used for |
|---------|----------|
| `torch`, `numpy`, `pandas`, `pyarrow`, `matplotlib`, `scikit-learn`, `xgboost` | MLP AL, graph screen, tree study |

Plot thesis figures from frozen CSVs (no GPU required):

```bash
cd code/chapter4_learning/mlp_active_learning
python plot_label_efficiency.py
```

See [`../code/chapter4_learning/README.md`](../code/chapter4_learning/README.md).

### Chapter 5 — optimization

Same stack as Chapter 4 plus Chapter 3 on `PYTHONPATH`. Checkpoints and the
blocked-200 benchmark ship under `datasets/chapter5_blocked200/` and
`results/chapter5_optimization/checkpoints/`.

```bash
python -m chapter5_optimization.run_workbench --help
```

### Chapter 6 — overlap

Chapter 3 + Chapter 5 checkpoints for repair methods that reuse the assemblability
MLP. Overlap pools and benchmarks under `datasets/chapter6_overlap_clevis/`.

```bash
python -m chapter6_overlap.export_ch64_viewer --help
```

### Chapter 7 — AeroForge

| Package / dependency | Used for |
|----------------------|----------|
| `cadquery`, `torch`, `scikit-learn` | ADV build, overlap labels, surrogates, repair |
| **AeroForge checkout** (not bundled) | Upstream aircraft parametric model + VLM quick-sim |

Clone AeroForge as a sibling of `bachelor-thesis/` (or set `AEROFORGE_ROOT`):

```bash
git clone https://github.com/IDEALLab/aeroforge.git ../aeroforge
export AEROFORGE_ROOT=/path/to/aeroforge   # optional override
python -m chapter7_aeroforge.ml.plot_100k_model_grid
```

Chapter 7 quick-sim/VLM payloads are released with attribution to the IDEAL Lab
AeroForge workflow and AeroSandbox-derived tooling. See
[`../code/chapter7_aeroforge/README.md`](../code/chapter7_aeroforge/README.md).

## Viewers (Chapters 5–7)

| Viewer | Default port | 3D render needs |
|--------|--------------|-----------------|
| Chapter 5 | 8090 | CadQuery + trimesh (Chapter 3 geometry) |
| Chapter 6 | 8091 | CadQuery + trimesh (Chapter 3 geometry) |
| Chapter 7 | 8092 | CadQuery + AeroForge import path (Chapter 7 `overlap_search.core`) |

Launch:

```bash
./scripts/view_ch5.sh
./scripts/view_ch6.sh
./scripts/view_ch7.sh
```

Or equivalently:

```bash
python -m viewers.chapter5_clevis_optimization.server
python -m viewers.chapter6_overlap_repair.server
python -m viewers.chapter7_aeroforge_repair.server
```

**Metrics-only mode:** all three viewers serve parameters, overlap statistics,
gradients, and markdown panels with plain Python 3. Only the on-demand mesh
render requires CadQuery (and Chapter 7 additionally needs the AeroForge tree).

**Browser:** WebGL-capable browser; Three.js loads from a CDN on first visit.

**Runtime caches:** viewers may write `cache/` or `data/*/meshes/` locally; safe
to delete, not part of the release.

## Platform assumptions

- macOS or Linux tested during development
- Viewers bind to `127.0.0.1` only (localhost)
- Large clone: Chapter 7 datasets plus compacted viewer data are roughly
  **810 MB** on disk; overall release size is larger because Chapter 5 viewer
  JSON is intentionally included.

## Known gaps

- `environment.yml` is a practical reproducibility environment, not an exact
  platform lockfile. Conda may solve slightly different builds on different
  platforms.
- AeroForge is an external clone — not vendored.
- The two Chapter 7 label JSONL files exceed GitHub's normal file-size limit
  uncompressed and therefore ship as plain-Git gzip archives; run
  `gunzip -k datasets/chapter7_aeroforge_adv/labels_{cloud,sim}/labels.jsonl.gz`
  once before using Chapter 7 code.
