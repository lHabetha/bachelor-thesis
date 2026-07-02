# Chapter 4 Datasets — Clevis Labeled Pools

Formula-labeled clevis candidate pools and the fixed holdout used in Chapter 4
(`ch:labeling-learning`, `sec:mlp-training`, `sec:arch-comparison`).

All rows use the corrected clevis geometry and labels from the analytic oracle
in `code/chapter3_clevis_setup/`. Pools were generated with the thesis
Appendix A.1 five-mode proposal cycle, then filtered to valid designs only.

## Layout

| Path | Rows | Role |
|------|------|------|
| `pool_50k/candidate_pool.parquet` | 50,000 | Primary dense pool (89.3% assemblable) |
| `pool_22k/candidate_pool.parquet` | 22,000 | Pool-size comparison |
| `pool_10k/candidate_pool.parquet` | 10,000 | Pool-size comparison |
| `holdout/holdout.parquet` | 5,000 | Fixed evaluation set (never for training/AL) |
| `seed_sets/dense50k_v1/` | — | 8 seed splits for 50k active-learning reruns |
| `seed_sets/dense10k_v1/` | — | Seed splits for 10k comparison runs |
| `seed_sets/pool_22k/` | — | Seed splits for original 22k run |

Each pool folder includes `pool_manifest.json` with SHA-256 checksums and label
distributions. The holdout has `holdout_manifest.json`.

**Total release size:** ~30 MB (normal Git; no LFS required).

## Schema

### Parameters (13-D input to surrogates)

`wall_thickness`, `outer_span`, `leg_length`, `depth`,
`main_hole_offset_from_open_end`, `main_hole_radius`, `main_pin_length`,
`main_pin_radius`, `cross_hole_radius`, `cross_hole_distance_from_free_end`,
`splint_radius`, `splint_length`, `overhang_span_y`

### Key label columns

| Column | Meaning |
|--------|---------|
| `param_id` | Stable row identifier (hash of parameters) |
| `label` | Binary assemblability: 1 = assemblable, 0 = blocked |
| `label_class` | `assemb` or `blocked` or `overlap` |
| `label_subclass` | Mechanism subclass when assemblable |
| `formula_reason` | Fine-grained reason (`assemb-roof_clearance`, etc.) |
| `validity_ok` | Passed geometric validity checks |

50k/10k pools also store derived oracle terms (margins, mechanism flags) used
during generation. Training code typically uses the 13 parameters + `label`.

### Normalization

There is no global normalization file. MLP training applies a per-split
`Standardizer` (mean/std fit on the labeled training subset only).

## Loading example

```python
from pathlib import Path
import pandas as pd

ROOT = Path("datasets/chapter4_clevis_pools")

pool = pd.read_parquet(ROOT / "pool_50k" / "candidate_pool.parquet")
holdout = pd.read_parquet(ROOT / "holdout" / "holdout.parquet")

FEATURES = [
    "wall_thickness", "outer_span", "leg_length", "depth",
    "main_hole_offset_from_open_end", "main_hole_radius",
    "main_pin_length", "main_pin_radius", "cross_hole_radius",
    "cross_hole_distance_from_free_end", "splint_radius",
    "splint_length", "overhang_span_y",
]

X_pool = pool[FEATURES].to_numpy()
y_pool = pool["label"].to_numpy()
X_holdout = holdout[FEATURES].to_numpy()
y_holdout = holdout["label"].to_numpy()
```

## Integrity (SHA-256)

| File | SHA-256 |
|------|---------|
| `pool_50k/candidate_pool.parquet` | `902244bd5ba9432eb56112ade3ecf40ceda262888efd796db78fd3cabee09f22` |
| `pool_22k/candidate_pool.parquet` | `321a237c7d4d142533d2c916485952ceade8cd46c305f105611e3e6f8b782868` |
| `pool_10k/candidate_pool.parquet` | `445fe02ad74b01b58e090f1ecff93048a14361b42ef788a85263714d1687f513` |
| `holdout/holdout.parquet` | `5b9030e1852cce459e16fc8c758436a44b86221cb5df9d60875d422d52cf1ac5` |

Holdout rows are disjoint from all candidate pools (`holdout_exclusion: true`).

Training and plot code: `code/chapter4_learning/`.
