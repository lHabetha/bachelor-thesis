"""Build nested stratified Chapter 4 base seed sets."""
from __future__ import annotations

import argparse

from .lib.data_utils import build_seed_sets, load_bundle
from .lib.paths import ensure_dirs, load_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    ensure_dirs()
    protocol = load_protocol()
    seed = args.seed if args.seed is not None else int(protocol["seeds"]["base_seed"])
    bundle = load_bundle()
    build_seed_sets(
        bundle,
        base_sizes=[int(v) for v in protocol["labels"]["base_sizes"]],
        n_splits=int(protocol["seeds"]["n_splits"]),
        seed=seed,
    )
    print("[task30] Wrote seed sets")


if __name__ == "__main__":
    main()
