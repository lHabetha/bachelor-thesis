# Chapter 3 Code — Clevis Setup

Public release of the modified-clevis geometry, validity checks, analytic
assemblability labels, and Appendix A.1 proposal streams from thesis
Chapter 3.

Thesis anchors: `ch:setup`, `sec:validity-labels`, `tab:clevis-parameters`,
Appendix A.1 (`sec:appendix-streams`).

Chapter 3 has **no separate dataset release**. Candidate pools generated from
this code belong under `datasets/chapter4_clevis_pools/`.

## Layout

| Module | Purpose |
|--------|---------|
| `generate_modified_clevis_dummy.py` | 13-parameter `DummyParams`, CadQuery `build_parts` |
| `clevis_generator.py` | `generate(params, out_dir)` → part `.obj` files + `params.json` |
| `design_space.py` | Validity constraints (`validate_params`), `sample_params` |
| `exact_assemblability.py` | Analytic roof / splint / inward mechanism checks |
| `smart_sampler.py` | Proposal streams + `sample_by_thesis_cycle` (Appendix A.1) |
| `overlap_check.py` | Mesh overlap test on generated parts |
| `labels.py` | Flat `label_params` record helper |
| `build_and_label.py` | CLI example script |

## Dependencies

Run in a Python 3.11 environment with:

- `cadquery` — CAD geometry export
- `numpy`, `scipy` — sampling (Latin hypercube)
- `trimesh`, `manifold3d` — mesh overlap booleans

See [`docs/environment.md`](../../docs/environment.md).

## Usage

From the repository root, change into `code/`:

```bash
cd bachelor-thesis/code
conda activate bachelor-thesis   # or your CadQuery-capable env

# Label five Appendix A.1 stream samples (no mesh export)
python -m chapter3_clevis_setup.build_and_label --skip-mesh --stream-demo 5 --seed 42

# Build CAD + overlap check + labels for the bundled demo fixture
python -m chapter3_clevis_setup.build_and_label \
  --params-json ../results/chapter3_clevis_setup/examples/demo01_params.json \
  --out-dir ../results/chapter3_clevis_setup/demo01_smoke
```

### Inputs

- `--params-json` — one design from a JSON file with the 13 thesis parameters
- `--stream-demo N` — draw `N` designs via the thesis five-mode cycle
  (`uniform → boundary → extreme → latin_hypercube → uniform`, repeating)
- `--seed` — starting seed for stream demo (default `42`)
- `--skip-mesh` — analytic labels only (no CadQuery / overlap)
- `--out-dir` — output root (default: `results/chapter3_clevis_setup/`)

### Outputs

Per sample under `--out-dir/<sample_name>/`:

- `bracket.obj`, `main_pin.obj`, `splint.obj`, `assembled.obj` (unless `--skip-mesh`)
- `params.json` (when meshes are generated)
- `label_summary.json` — validity, `label_reason`, assemblability, overlap status

Run-level `run_manifest.json` at the output root lists all samples.

Default output location: `bachelor-thesis/results/chapter3_clevis_setup/`.

## Appendix A.1 streams

Use `sample_by_thesis_cycle(seed)` from `smart_sampler.py`. The mode is
`seed % 5`:

| `seed % 5` | Stream |
|------------|--------|
| 0 | uniform |
| 1 | boundary-biased |
| 2 | extreme (Beta U-shaped quantiles) |
| 3 | Latin hypercube (batch size 1) |
| 4 | uniform |

Three parameters (main-hole radius, cross-hole distance, splint length) are drawn
conditionally inside `_derive_from_quantiles`, matching Appendix A.1.

`build_random_plan` and `stream_anchored_gaussian` are legacy helpers retained
for compatibility; the thesis documents the five-mode cycle above.
