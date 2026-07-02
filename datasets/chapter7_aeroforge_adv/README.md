# Chapter 7 Dataset — AeroForge ADV

Thesis anchor: Chapter 7 (`ch:aeroforge-transfer`, §7.2–§7.5).

---

## Contents

| Artifact | Public path | Size | Thesis anchor |
|----------|-------------|------|---------------|
| **100k ADV pool** | `advs/000000.json` … `099999.json` | **391 MB** | `sec:aeroforge-sampling` |
| Pool manifest | `manifest.json` | 4 KB | Family weights, schema |
| **Overlap labels** | `labels_cloud/labels.jsonl.gz` | **19 MB** (141 MB unpacked) | Surrogate training + repair benchmark |
| Overlap summary | `labels_cloud/summary.json` | 2 KB | 41,169/100k overlap rows |
| **Quick-sim / VLM labels** | `labels_sim/labels.jsonl.gz` | **42 MB** (244 MB unpacked) | Performance MLP, aero-preserve eval |
| Quick-sim summary | `labels_sim/summary.json` | 2 KB | 99,998 VLM-ok rows |
| **Repair benchmark** | `benchmark_overlap_ranked100_v2/overlap_ranked100_v2.json` | 173 KB | All Ch.7 repair tables (100 starts) |
| Optimizer normalization | `normalization_bounds.json` | ~4 KB | Empirical [min,max] per numeric driver |
| ADV schema | `adv_schema.json`, `adv_schema.md` | ~3 KB | Appendix ADV tables |

**Total:** ~776 MB (100,000 ADV JSON files).

---

## Decompressing the label files

The two label JSONL files ship gzip-compressed so the repository stays within
GitHub's normal file-size limits without Git LFS. Before running any Chapter 7
code that reads them (`code/chapter7_aeroforge/release_paths.py` expects the
uncompressed `labels.jsonl` paths), decompress once, keeping the archives:

```bash
gunzip -k datasets/chapter7_aeroforge_adv/labels_cloud/labels.jsonl.gz
gunzip -k datasets/chapter7_aeroforge_adv/labels_sim/labels.jsonl.gz
```

---

## SHA-256 (key blobs)

| File | SHA-256 |
|------|---------|
| `labels_cloud/labels.jsonl.gz` (shipped) | `6cce88df6f2ee48e4cbece47de76d4c8693ccc82df3eba121ed9c430146f32b5` |
| `labels_sim/labels.jsonl.gz` (shipped) | `3c0c70cff27a06cd374dd0bca588cdc5ecc8b99c92bc3fdf96cf784426fc0263` |
| `labels_cloud/labels.jsonl` (after gunzip) | `6204e15565758787aa5fa8b03699d61b838712bb268d78ddcc371ee3b7d1e002` |
| `labels_sim/labels.jsonl` (after gunzip) | `047443a9b35140705ceb6ce91336f738298ab1d4e128a82351206c68293821fa` |

---

## Holdout / evaluation splits

No frozen holdout CSV. Thesis metrics use:

| Split | Mechanism |
|-------|-----------|
| Overlap surrogate CV | 10-fold × 3 seeds in model grid (`results/.../comparisons/model_grid_100k_v1/`) |
| Performance MLP | 90/10 train/val on 90,630 filtered sim rows |
| Repair benchmark | Fixed 100 worst-overlap starts (`overlap_ranked100_v2.json`) |

---

## Large files

The two label JSONL files would exceed GitHub's normal 100 MB file-size limit
uncompressed, so they ship as gzip archives (`labels.jsonl.gz`, 19 MB + 42 MB)
in plain Git — no Git LFS required. Decompress them once as described above.

The 100k ADV JSON files remain normal Git files because each individual file is
small and the public Chapter 7 code reads the directory directly.

**Redistribution:** `labels_sim/` contains VLM quick-report payloads derived from
the IDEAL Lab AeroForge workflow and AeroSandbox-derived quick-sim tooling. They
are included for thesis reproducibility with attribution. Upstream AeroForge:
[`IDEALLab/aeroforge`](https://github.com/IDEALLab/aeroforge).

---

## Related folders

- Runnable code using this dataset: `code/chapter7_aeroforge/`
- Viewer allowlist and run data: `viewers/chapter7_aeroforge_repair/` with
  `release_manifests/ch7_allowlist.json`
