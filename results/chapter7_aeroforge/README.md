# Chapter 7 Results — AeroForge Overlap Transfer

Thesis anchor: Chapter 7 (`ch:aeroforge-transfer`).

Viewer run data lives under `viewers/chapter7_aeroforge_repair/data/`.

---

## Checkpoints (`checkpoints/`)

| File | Size | Thesis use |
|------|------|------------|
| `eng_multitask_gate_strong_100k.pt` | 229 KB | Headline overlap brain (variant A); `tab:aeroforge-overlap-repair` |
| `eng_log_huber_100k.pt` | 228 KB | Grid column B |
| `eng_gate_aug_strong_100k.pt` | 228 KB | Best gate F1 in `tab:aeroforge-surrogate-results` |
| `raw_log_huber_100k.pt` | 221 KB | Grid column E |
| `performance_mlp_90k.pt` | 222 KB | `sec:aeroforge-performance-preserving` |
| `registry.json` | 3 KB | Model index |

SHA-256 (`eng_multitask_gate_strong_100k.pt`): `c57807018f41e1409c2315dab350a14524cb240813757804cc72c02b0b05fef7`

---

## Surrogate evaluation (`tab:aeroforge-surrogate-results`)

| Artifact | Path |
|----------|------|
| CV aggregate CSV | `comparisons/model_grid_100k_v1/results_agg.csv` |
| Leaderboard JSON | `comparisons/model_grid_100k_v1/leaderboard.json` |
| Learning-curve figure | `figures/model_grid_100k.png` (`fig:aeroforge-model-grid`) |

SHA-256 (`model_grid_100k.png`): `b9fb0b12cb0ecf6957b2e1bd03b4a30718e7d92bfed7ea9b1150272751491885`

---

## Repair tables

| Table | Path |
|-------|------|
| Headline + grid (`tab:aeroforge-overlap-repair`, `tab:aeroforge-overlap-repair-grid`) | `tables/repair_grid_100k_v2/` (P1–P7, T1–T6, thesis_ready_summary.csv) |
| Regenerable via `summarize_runs --grid-id grid_100k_v2` | |

Headline variant A / `all` row: **94/100 verified clean, 6 false OK** — matches the thesis.

---

## Aero-preserve (`tab:aeroforge-aero-preserve`)

| File | Notes |
|------|-------|
| `tables/aero_preserve_v1/T1_repair.csv` … `T6_fidelity.csv` | Full 15-run sweep export |
| `tables/aero_preserve_v1/T1_thesis_rows.csv` | 7 thesis table rows (+ all_aero meta) |
| `tables/aero_preserve_v1/README.md` | Methodology notes |

---

## Performance MLP

| File | Path |
|------|------|
| Training metrics | `performance_mlp_v1/metrics.json` |

SHA-256 (`performance_mlp_90k.pt`): `0ef560b20c61d95f9b889bf0f638d26ca816a2191b5027d2049cf7934310ce15`

---

## Size summary

| Component | Size |
|-----------|------|
| Checkpoints + registry | ~1.1 MB |
| Model grid + figure | ~220 KB figure + CSVs |
| Repair grid tables | ~28 KB |
| Aero-preserve tables | ~28 KB |
| **Subtotal (`results/`)** | **~1.5 MB** |
| Datasets (sibling folder) | ~776 MB |
