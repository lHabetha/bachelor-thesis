# Chapter 7 AeroForge Overlap-Repair Viewer

Interactive localhost viewer for the AeroForge overlap-repair experiments in
Chapter 7 of the thesis. It shows, per optimizer start, the MLP-only repair
search, the CAD-verified final overlap, driver gradients, and the
performance-preserving quick-sim (VLM) aero deltas. Start and repaired airframes
can be rendered to 3D on demand.

The viewer serves the runs listed in
[`../../release_manifests/ch7_allowlist.json`](../../release_manifests/ch7_allowlist.json).

## Scope

- **49 allowlisted runs** (JSON only; meshes are rendered locally on demand):
  - **42 overlap-repair grid cells** — 6 surrogate-model columns × 7 optimizer
    rows (`tab:aeroforge-overlap-repair-grid`). Column A also anchors the
    headline table `tab:aeroforge-overlap-repair`.
  - **7 aero-preserve rows** — the thesis rows of `tab:aeroforge-aero-preserve`.
- Runs are grouped in the dropdown by surrogate model (columns A–F) plus a
  dedicated **Aero-preserving repair** group.
- Each run folder under `data/<run_id>/` contains `manifest.json`,
  `viewer_data.json`, `statistics.json`, `sim_eval.json`, and `algorithm.md`
  (thesis-aligned optimizer description for the **ℹ Algorithm info** panel).

## Launch

Run from the `bachelor-thesis/` repository root:

```bash
python -m viewers.chapter7_aeroforge_repair.server
# custom port:
python -m viewers.chapter7_aeroforge_repair.server --port 8092
```

Then open <http://127.0.0.1:8092>.

Metrics, gradients, and quick-sim panels work with a plain Python install.

## On-demand 3D rendering (optional)

The **Render start + final** button builds the two airframes with CadQuery and
serves them as STL. This requires the Chapter 7 CAD stack:

1. Launch inside the `bachelor-thesis` conda environment (CadQuery available).
2. A local `aeroforge/` checkout must be importable — see
   [`../../code/chapter7_aeroforge/README.md`](../../code/chapter7_aeroforge/README.md)
   (set `AEROFORGE_ROOT` if it is not a sibling of the repo).

Rendering writes STL files under `data/<run_id>/meshes/` on first use; these are
generated locally and are not part of the released bundle.

## Data size

The 49 JSON run folders total roughly 34 MB. Each `viewer_data.json` keeps
summary metrics, gradients, start/final ADVs for on-demand rendering, and
`sim_eval.json` quick-sim payloads. Meshes are generated locally by the
**Render start + final** button and are not shipped.

## Files

| Path | Purpose |
|------|---------|
| `server.py` | Allowlist-driven HTTP server + on-demand CAD render |
| `index.html` | Three.js frontend (metrics, gradients, quick-sim, geometry) |
| `data/<run_id>/` | Allowlisted run JSON (manifest / viewer_data / statistics / sim_eval) |
