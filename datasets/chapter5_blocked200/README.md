# Chapter 5 Dataset — Blocked 200 Benchmark

Thesis anchor: Chapter 5 (`ch:optimization`, `sec:optimization-objective`);
appendix `tab:appendix-optimizer-benchmark`.

200 blocked-start clevis designs used by the Chapter 5 optimizer workbench.

---

## Files

| File | Size | SHA-256 (parquet) | Purpose |
|------|------|-------------------|---------|
| `blocked_200_v1.parquet` | 69 KB | `cbf97ee4…dcd25e` | Primary release format (200 rows, 35 columns) |
| `blocked_200_v1.jsonl` | 282 KB | — | Same rows, one JSON object per line |
| `normalization.json` | 1 KB | — | Per-parameter median/std for normalized L2 distance |
| `generation_audit.json` | 1.5 KB | — | Seed 6202001, quotas, leakage-exclusion proof |
| `README.md` | 2 KB | — | Bucket quotas |

**Total:** ~368 KB — normal Git (no LFS).

---

## Row composition

| Bucket (`benchmark_bucket`) | Count |
|-----------------------------|-------|
| near_boundary | 70 |
| typical_blocked | 50 |
| extreme_roof_cage | 50 |
| stress_extreme_corner | 30 |

Starts are **not** sampled from the Chapter 4 dense pool. Generation excluded
50k pool IDs, 5k holdout IDs, legacy `blocked_100_v1`, and default MLP training IDs
(see `generation_audit.json`).

Formula oracle: Chapter 3 exact assemblability (`label=0`, `formula_reason=blocked`).

---

## Related artifacts (not in this folder)

| Artifact | Public path |
|----------|-------------|
| MLP input scaler + weights | `results/chapter5_optimization/checkpoints/row1_uncertainty_disagreement_B1000_T2500_best/` |
| Surrogate threshold τ | Runtime `--tau` (not a file) |
| Tree control checkpoints | `results/chapter5_optimization/checkpoints/*_task30_labelblind_rank1/` |

---

## Regeneration

Use `code/chapter5_optimization/generate_benchmark.py`.
