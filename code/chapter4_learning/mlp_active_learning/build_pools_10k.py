"""Build 10k candidate pool and seed sets for the label-blind pool-size rerun.

Uses the same generator and holdout exclusion as ``build_pools_50k.py``.
"""
from __future__ import annotations

import argparse

from build_pools_50k import build_candidate_pool, write_manifest, write_seed_sets

POOL_ID = "dense10k_v1"
POOL_FILE = "candidate_pool.parquet"
MANIFEST_FILE = "pool_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--n-candidates", type=int, default=10000)
    parser.add_argument("--candidate-seed", type=int, default=550001)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--write-seed-sets", action="store_true")
    parser.add_argument("--seed-splits", type=int, default=8)
    args = parser.parse_args()

    if args.write_seed_sets:
        write_seed_sets(
            n_splits=args.seed_splits,
            pool_id=POOL_ID,
            pool_file=POOL_FILE,
        )
    else:
        build_candidate_pool(
            n_candidates=args.n_candidates,
            workers=args.workers,
            seed=args.candidate_seed,
            force=args.force,
            pool_id=POOL_ID,
            pool_file=POOL_FILE,
        )
        write_manifest(
            force=args.force,
            pool_id=POOL_ID,
            pool_file=POOL_FILE,
            manifest_file=MANIFEST_FILE,
            pool_seed_base=args.candidate_seed,
        )


if __name__ == "__main__":
    main()
