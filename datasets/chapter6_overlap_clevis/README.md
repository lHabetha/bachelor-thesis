# Chapter 6 Dataset — Overlap Clevis

Thesis anchor: Chapter 6 (`ch:overlap-aware-extension`, §6.2–§6.5).

Relaxed-validity clevis candidate pools, labeled subsets, calibration fixtures,
and strict repair benchmarks for the overlap-aware extension.

---

## Files

| Path | Size | SHA-256 | Purpose |
|------|------|---------|---------|
| `pool_100k/pool.csv` | 38.4 MB | `e3f588fd…5669a` | 100,000 params-only rows (seed 252520) |
| `pool_100k/manifest.json` | 1.6 KB | — | Pool metadata, stream counts, schema |
| `holdout_5k/pool.csv` | 1.9 MB | `7d927007…0fae2` | 5,000 fixed holdout rows (seed 252521) |
| `holdout_5k/manifest.json` | 1.4 KB | — | Holdout metadata |
| `labeled/acquired_random_15k.csv` | 7.9 MB | — | Deployed regressor training labels |
| `labeled/holdout_labeled_5k.csv` | 2.6 MB | — | Fixed holdout overlap labels (§6.3 metrics) |
| `calibration/calibration_rows.csv` | 4 KB | — | 12-assembly voxel calibration (Fig 6.1) |
| `benchmark_strict_repair_ch64_v1/` | 68 KB | — | 50 overlap-only repair starts (§6.4) |
| `benchmark_strict_blocked_v2/` | 75 KB | — | 50 overlap+blocked starts (§6.5) |

**Total:** ~51 MB — normal Git (no LFS).

---

## Pool policy

- **100k pool:** parameter-only; overlap labels acquired on demand during active
  learning (not stored in the pool CSV).
- **5k holdout:** fixed evaluation set; labeled copy also under `labeled/`.
- **Relaxed sampler:** `code/chapter6_overlap/configs/relaxed_sampler_v1.json`.

---

## Benchmarks

### `benchmark_strict_repair_ch64_v1` (§6.4)

50 curated starts, strict threshold normalized overlap ≤ 10⁻⁶. Ten per contact
family; 33 large / 7 moderate / 10 near-threshold starts.

Thesis anchor: `tab:task25-strict-benchmark`.

### `benchmark_strict_blocked_v2` (§6.5)

Same 50-start prefix as ch64_v1, filtered to designs that are both strictly
overlapping and kinematically blocked at start.

Thesis anchor: `tab:task25-strict-zigzag`.

---

## Related paths

| Artifact | Location |
|----------|----------|
| Overlap checkpoints | `results/chapter6_overlap/checkpoints/` |
| Frozen thesis tables/figures | `results/chapter6_overlap/tables/`, `figures/` |
| Overlap code | `code/chapter6_overlap/` |
| Viewer run data | `viewers/chapter6_overlap_repair/data/` |
