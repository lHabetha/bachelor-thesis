# Strict Overlap Repair Benchmark (Chapter 6.4, ch64_v1)

Canonical 50-start overlap-only repair benchmark for Chapter 6.4. These are the
first 50 rows of the strict start set, in the same ordering Chapter 6.5
consumes, so the two chapters share a benchmark prefix. Success is strict
analytic non-overlap (`total_overlap_norm <= 1e-6`); the goal is overlap
*removal* only, not kinematic assemblability.

- Source: `datasets/chapter6_overlap_clevis/benchmark_strict_repair_ch64_v1/starts.csv`
- Starts: `50`
- Strict success threshold (normalized): `1e-06`
- Acquisition / clean threshold tau0 (normalized): `5e-05`
- Start overlap range (normalized): `4.3e-05` .. `0.0199`
- Near-zero-band starts (1e-6 < O < 5e-5, vanishing-gradient regime): `10`

## Category counts

| Category | Count |
|---|---:|
| bracket_splint_other | 10 |
| other_overlap | 10 |
| pin_head_or_shaft_wall_candidate | 10 |
| pin_splint_cross_overlap | 10 |
| splint_head_wall_candidate | 10 |

## Magnitude-bin counts

| Magnitude bin | Count |
|---|---:|
| large | 33 |
| moderate | 7 |
| tiny | 10 |
