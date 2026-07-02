# Active-Learning Seed Splits

JSON files listing `param_id` values drawn from the matching candidate pool.
Used to reproduce the Chapter 4 label-efficiency grid (base sizes
`B ∈ {0, 250, 500, 750, 1000, 1500, 2000, 2500}`, eight seed splits `R001`–`R008`).

| Folder | Pool |
|--------|------|
| `dense50k_v1/` | 50k primary pool |
| `dense10k_v1/` | 10k comparison pool |
| `pool_22k/` | 22k comparison pool |

Each split folder contains `base_prefix_<B>.json` files and a `master_2500.json`
full prefix for the largest base size.
