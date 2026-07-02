# `data/seed_sets/` — Base seed prefixes per seed split

Ensures Row 1 and Row 2 start from **identical** base training sets for each `(B, R)`.

## Layout

```text
seed_sets/
└── R001/
    ├── master_2500.json       # full stratified list (length 2500 or special case)
    ├── base_prefix_0.json     # empty (B=0 cold start)
    ├── base_prefix_250.json   # first 250 param_ids from master
    ├── base_prefix_500.json
    …
    └── base_prefix_2500.json
```

Naming: `R` zero-padded to 3 digits (`R001`, `R002`, …).

## How generated

```bash
python build_pools.py --seed-splits 8 --write-seed-sets
```

For each seed split `R`:

1. Draw one **master stratified** list `S_R` of length 2500 (or max base).
2. For each `B`, write `base_prefix_<B>.json` = first `B` entries of `S_R`.

Nested prefix property: `base_prefix_250 ⊂ base_prefix_500 ⊂ … ⊂ base_prefix_2500`.

## Special case B = 2500

The clean corpus behind this pool has only **2353** rows. For `B=2500`:

- all 2353 corpus param_ids are used,
- plus **147** additional IDs drawn once per `R` from the candidate pool
  (deterministic).

## JSON format

```json
{
  "seed_split": 1,
  "base_size": 250,
  "param_ids": ["...", "..."]
}
```

## Pairing rule

Row 1 and Row 2 for the same `(B, R)` must load the same `base_prefix_<B>.json`.
When rerunning, do not regenerate base seeds per row (this breaks the pairing)
and do not share acquisition RNG state across rows (only the base is shared).
