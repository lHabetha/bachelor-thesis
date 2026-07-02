# Thesis Viewers

Three localhost viewers for inspecting the optimizer and repair runs reported
in the thesis tables. Each serves the entries listed in its allowlist.

| Viewer | Folder | Allowlist | Default port |
|--------|--------|-----------|--------------|
| Chapter 5 clevis optimization | [`chapter5_clevis_optimization/`](chapter5_clevis_optimization/) | [`../release_manifests/ch5_allowlist.json`](../release_manifests/ch5_allowlist.json) | 8090 |
| Chapter 6 overlap repair | [`chapter6_overlap_repair/`](chapter6_overlap_repair/) | [`../release_manifests/ch6_allowlist.json`](../release_manifests/ch6_allowlist.json) | 8091 |
| Chapter 7 AeroForge repair | [`chapter7_aeroforge_repair/`](chapter7_aeroforge_repair/) | [`../release_manifests/ch7_allowlist.json`](../release_manifests/ch7_allowlist.json) | 8092 |

## Launch

From the repository root:

```bash
./scripts/view_ch5.sh
./scripts/view_ch6.sh
./scripts/view_ch7.sh
```

Pass `--port <n>` to any script to override the default. See
[`../scripts/README.md`](../scripts/README.md) and
[`../docs/environment.md`](../docs/environment.md) for dependencies.

## Scope summary

- **Chapter 5:** 88 dropdown entries backed by 80 unique run folders under
  `chapter5_clevis_optimization/data/`. Groups match thesis tables (§5.2–§5.4).
- **Chapter 6:** 10 entries (8 strict-repair + 1 zigzag + 1 optional contrast).
  Run folders under `chapter6_overlap_repair/data/`.
- **Chapter 7:** 49 entries (42 repair grid cells + 7 aero-preserve rows).
  JSON-only under `chapter7_aeroforge_repair/data/` (~34 MB after compaction); STL meshes are
  built on demand.

## Per-viewer READMEs

- [`chapter5_clevis_optimization/README.md`](chapter5_clevis_optimization/README.md)
- [`chapter6_overlap_repair/README.md`](chapter6_overlap_repair/README.md)
- [`chapter7_aeroforge_repair/README.md`](chapter7_aeroforge_repair/README.md)
