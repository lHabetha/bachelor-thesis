# Chapter 5 Clevis Optimization Viewer

A local browser viewer for the surrogate-based optimizer trajectories reported
in Chapter 5 ("Surrogate-Based Optimization"). It replays each optimizer run
over the 200 blocked starts of `blocked_200_v1`, showing per-frame parameters,
the MLP-predicted assembly probability, the gradient direction, and an on-demand
3D render of the clevis assembly generated from the Chapter 3 CAD code.

## Scope (allowlist only)

The dropdown is driven entirely by
[`../../release_manifests/ch5_allowlist.json`](../../release_manifests/ch5_allowlist.json).
It exposes exactly the runs that back a Chapter 5 table row:

- **88 dropdown entries** — one per thesis table row, grouped by table anchor
  (5.2.1 one-shot gradient … 5.4 sparse-norm selection, plus the 5.2.7
  tree-surrogate controls).
- **80 unique run folders** under `data/`. A run folder shared by several table
  rows (for example a §5.2 baseline reused as the unconstrained reference in a
  §5.3 lock table) appears once per table row but is stored once on disk.

## Layout

```
chapter5_clevis_optimization/
├── server.py        # localhost HTTP server, allowlist-gated
├── index.html       # single-page Three.js UI
└── data/<run_id>/   # per-run: manifest.json, viewer_data.json,
                     #          statistics.json, statistics.md, algorithm.md
```

Each run folder holds the five files the viewer reads.

## Requirements

- Python 3.10+ with `cadquery` (2.7.x) and `trimesh` for the 3D geometry.
  The `bachelor-thesis` conda environment (root `environment.yml`) provides these.
- A WebGL-capable browser (Three.js is loaded from a CDN, so first launch needs
  internet access; the data panels and markdown work offline regardless).

The 3D render is generated on demand from the Chapter 3 geometry builder in
`../../code/chapter3_clevis_setup/generate_modified_clevis_dummy.py`. If
`cadquery` is unavailable the viewer still serves all parameter, probability,
gradient, statistics, and algorithm data; only the mesh render is skipped.

## Launch

Run from the repository root (`bachelor-thesis/`) so the Chapter 3 code and the
`datasets/chapter5_blocked200` benchmark metadata resolve:

```sh
python -m viewers.chapter5_clevis_optimization.server
```

Then open <http://127.0.0.1:8090>. Use a different port with `--port`:

```sh
python -m viewers.chapter5_clevis_optimization.server --port 8099
```

## Using the viewer

- **Thesis table / figure**: pick a Chapter 5 table to filter the run list, or
  keep "All Chapter 5 tables".
- **Optimizer Run**: choose a table row; the label matches the thesis alias and
  shows headline metrics (oracle-confirmed count, false successes, mean L2).
- **Assembly**: pick one of the 200 blocked starts. `✓` marks oracle-confirmed
  success, `~` a surrogate-only success.
- **Timeline**: scrub or play the 50 optimization frames; the parameter panel
  shows the start→current change and the gradient direction per parameter.
- **ℹ Info / 📊 Stats**: the run's `algorithm.md` and `statistics.md`.
- **Lock Study**: the parameter-lock robustness summary
  (`../../results/chapter5_optimization/constraint_studies/robustness_summary.md`).

A generated-geometry cache is written to `cache/` at runtime; it is safe to
delete and is not part of the release.
