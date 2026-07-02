# Chapter 6 Overlap Repair Viewer

A local browser viewer for the strict overlap-repair and alternating
overlap/assemblability trajectories reported in Chapter 6 ("Overlap-Aware
Extension"). It replays each thesis-table method over the frozen 50-start
benchmarks, showing per-frame analytic overlap, pair residuals, parameter
changes, and on-demand 3D renders of the clevis assembly generated from the
Chapter 3 CAD code.

## Scope (allowlist only)

The dropdown is driven entirely by
[`../../release_manifests/ch6_allowlist.json`](../../release_manifests/ch6_allowlist.json).
It exposes exactly the runs that back a Chapter 6 table row:

- **10 dropdown entries** — 8 × `tab:task25-strict-repair`, 1 ×
  `tab:task25-strict-zigzag`, plus 1 optional finish-line zigzag contrast.
- **10 unique run folders** under `data/`.

## Layout

```
chapter6_overlap_repair/
├── server.py        # localhost HTTP server, allowlist-gated
├── index.html       # single-page Three.js UI (overlap + zigzag modes)
└── data/<run_id>/   # per-run: manifest.json, viewer_data.json, algorithm.md
```

Each run folder holds only the three files the viewer reads. Ch6 exports do not
ship `statistics.json` / `statistics.md`; the stats panel is synthesized from
`manifest.results_summary`.

## Requirements

- Python 3.10+ with `cadquery` (2.7.x) and `trimesh` for the 3D geometry.
  The `bachelor-thesis` conda environment (root `environment.yml`) provides these.
- A WebGL-capable browser (Three.js is loaded from a CDN).

If `cadquery` is unavailable the viewer still serves all overlap metrics,
parameter trajectories, and markdown panels; only the mesh render is skipped.

## Launch

Run from the repository root (`bachelor-thesis/`):

```sh
python -m viewers.chapter6_overlap_repair.server
```

Then open <http://127.0.0.1:8091>. Use a different port with `--port`:

```sh
python -m viewers.chapter6_overlap_repair.server --port 8099
```

## Using the viewer

- **Thesis table / section**: filter by `tab:task25-strict-repair`,
  `tab:task25-strict-zigzag`, or show all Chapter 6 entries.
- **Repair / pipeline run**: choose a thesis table row; labels show strict-success
  counts and mean overlap reduction (§6.4) or pipeline success (§6.5).
- **Assembly**: pick one of the 50 benchmark starts. `✓` marks strict/pipeline
  success; `~` marks partial overlap reduction without strict clearance.
- **Timeline**: scrub or play frames. §6.4 runs morph between start and final
  geometry; §6.5 zigzag runs show per-stage stop-motion through each repair
  segment.
- **ℹ Info / 📊 Stats**: the run's `algorithm.md` and a summary derived from
  `manifest.json`.

A generated-geometry cache is written to `cache/` at runtime; it is safe to
delete and is not part of the release.

Run data can be regenerated from the frozen aggregates under
`results/chapter6_overlap/repair_ch64/` via
`chapter6_overlap.export_ch64_viewer`.
