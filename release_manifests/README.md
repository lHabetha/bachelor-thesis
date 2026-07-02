# Release Manifests — Viewer Allowlists

This folder holds the allowlist files that drive the three localhost viewers.
Each viewer server reads its allowlist at startup and serves exactly the runs
listed there — the runs reported in the thesis tables and figures.

| File | Viewer | Entries |
|------|--------|---------|
| `ch5_allowlist.json` | `viewers/chapter5_clevis_optimization/` | 88 entries (80 run folders) |
| `ch6_allowlist.json` | `viewers/chapter6_overlap_repair/` | 10 entries |
| `ch7_allowlist.json` | `viewers/chapter7_aeroforge_repair/` | 49 entries (42 repair-grid + 7 aero-preserve) |

Each entry records the run identifier, a display label for the viewer
dropdown, the thesis table/figure anchor it corresponds to, and the run-data
folder under `viewers/<viewer>/data/`.

To browse the runs, launch a viewer via `scripts/view_ch5.sh`,
`scripts/view_ch6.sh`, or `scripts/view_ch7.sh` (see the root `README.md`).
