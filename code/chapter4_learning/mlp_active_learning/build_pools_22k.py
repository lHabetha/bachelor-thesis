"""Build candidate pool, holdout, and seed sets for Chapter 4 (Clean Rerun — Post-Chapter 3).

Hardened: loops in deterministic chunks until exact target counts are reached,
regardless of validity rejection rate from corrected Chapter 3 geometry rules.

Usage:
    python build_pools.py --workers 16 --force
    python build_pools.py --seed-splits 8 --write-seed-sets
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.formula_oracle import (
    FEATURE_NAMES,
    REASON_OVERLAP,
    REASON_ORDER,
    label_record,
    params_from_dict,
)
from lib.io import (
    atomic_json_write,
    file_sha256,
    param_hash,
    records_to_parquet,
    read_parquet,
)

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT.parent))
from release_paths import (  # noqa: E402
    DATASETS_ROOT,
    HOLDOUT_PATH,
    SEED_SETS_ROOT,
    ensure_chapter3_importable,
)

ensure_chapter3_importable()
from chapter3_clevis_setup.design_space import DummyParams, sample_params, validate_params  # noqa: E402
from chapter3_clevis_setup.smart_sampler import (  # noqa: E402
    build_random_plan,
    stream_anchored_gaussian,
    stream_boundary_pushed,
    stream_extreme,
    stream_lhs_batch,
    stream_uniform,
)

_POOL_ID = "pool_22k"
_POOLS_DIR = DATASETS_ROOT / _POOL_ID
_SEEDS_DIR = SEED_SETS_ROOT / _POOL_ID
_HOLDOUT_DIR = DATASETS_ROOT / "holdout"
_HOLDOUT_PATH = HOLDOUT_PATH


# ─── Pool generation ─────────────────────────────────────────────────────


def _generate_one_candidate(seed: int) -> dict[str, Any] | None:
    """Generate and formula-label a single candidate. Returns None if invalid."""
    rng = np.random.default_rng(seed)
    mode = seed % 5
    try:
        if mode == 0:
            p = stream_uniform(rng)
        elif mode == 1:
            p = stream_boundary_pushed(rng)
        elif mode == 2:
            p = stream_extreme(rng)
        elif mode == 3:
            batch = stream_lhs_batch(1, rng)
            p = batch[0] if batch else stream_uniform(rng)
        else:
            p = stream_uniform(rng)
    except Exception:
        return None

    rec = label_record("", "", p)
    if not rec.get("validity_ok", False):
        return None
    if rec.get("formula_reason") == REASON_OVERLAP:
        return None
    return rec


def _generate_pool_worker(args: tuple[int, int]) -> list[dict[str, Any]]:
    """Worker: generate a chunk of candidates."""
    start_seed, count = args
    results = []
    for i in range(count):
        rec = _generate_one_candidate(start_seed + i)
        if rec is not None:
            results.append(rec)
    return results


def build_candidate_pool(
    n_candidates: int = 22000,
    workers: int = 16,
    seed: int = 220001,
    force: bool = False,
) -> Path:
    """Generate and save candidate_pool.parquet.

    Loops in batches until exactly n_candidates unique valid rows are collected.
    """
    out_path = _POOLS_DIR / "candidate_pool.parquet"
    if out_path.exists() and not force:
        print(f"[build_pools] candidate pool exists: {out_path}")
        return out_path

    print(f"[build_pools] Generating {n_candidates} candidates with {workers} workers...")
    t0 = time.perf_counter()

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    batch_seed = seed
    batch_size = n_candidates * 2
    attempt = 0

    while len(deduped) < n_candidates:
        attempt += 1
        chunk_size = batch_size // workers
        chunks = [(batch_seed + i * chunk_size, chunk_size) for i in range(workers)]

        with Pool(processes=workers) as pool:
            chunk_results = pool.map(_generate_pool_worker, chunks)

        for chunk in chunk_results:
            for rec in chunk:
                h = param_hash(rec)
                if h in seen:
                    continue
                seen.add(h)
                rec["param_id"] = h
                rec["source"] = f"pool_gen_seed{seed}_batch{attempt}"
                deduped.append(rec)
                if len(deduped) >= n_candidates:
                    break
            if len(deduped) >= n_candidates:
                break

        batch_seed += batch_size
        print(f"  batch {attempt}: {len(deduped)}/{n_candidates} collected", flush=True)

    deduped = deduped[:n_candidates]
    elapsed = time.perf_counter() - t0
    print(f"[build_pools] Pool complete: {len(deduped)} rows in {elapsed:.1f}s")

    records_to_parquet(deduped, out_path)
    return out_path


def build_holdout(
    n_holdout: int = 5000,
    workers: int = 16,
    seed: int = 500001,
    force: bool = False,
) -> Path:
    """Generate and save holdout.parquet (disjoint from candidate pool).

    Loops in batches until exactly n_holdout unique valid rows are collected.
    """
    out_path = _HOLDOUT_DIR / "holdout.parquet"
    if out_path.exists() and not force:
        print(f"[build_pools] holdout exists: {out_path}")
        return out_path

    # Load candidate pool param_ids to exclude
    cand_path = _POOLS_DIR / "candidate_pool.parquet"
    if cand_path.exists():
        cand_records = read_parquet(cand_path)
        excluded = {str(r["param_id"]) for r in cand_records}
    else:
        excluded = set()

    print(f"[build_pools] Generating {n_holdout} holdout rows (excl {len(excluded)} pool IDs)...")
    t0 = time.perf_counter()

    seen: set[str] = set(excluded)
    deduped: list[dict[str, Any]] = []
    batch_seed = seed
    batch_size = n_holdout * 3
    attempt = 0

    while len(deduped) < n_holdout:
        attempt += 1
        chunk_size = batch_size // workers
        chunks = [(batch_seed + i * chunk_size, chunk_size) for i in range(workers)]

        with Pool(processes=workers) as pool:
            chunk_results = pool.map(_generate_pool_worker, chunks)

        for chunk in chunk_results:
            for rec in chunk:
                h = param_hash(rec)
                if h in seen:
                    continue
                seen.add(h)
                rec["param_id"] = h
                rec["source"] = f"holdout_gen_seed{seed}_batch{attempt}"
                deduped.append(rec)
                if len(deduped) >= n_holdout:
                    break
            if len(deduped) >= n_holdout:
                break

        batch_seed += batch_size
        print(f"  batch {attempt}: {len(deduped)}/{n_holdout} collected", flush=True)

    deduped = deduped[:n_holdout]
    elapsed = time.perf_counter() - t0
    print(f"[build_pools] Holdout complete: {len(deduped)} rows in {elapsed:.1f}s")
    records_to_parquet(deduped, out_path)
    return out_path


def write_manifest(force: bool = False) -> None:
    """Write pool_manifest.json with file hashes, row counts, and version markers."""
    manifest_path = _POOLS_DIR / "pool_manifest.json"
    if manifest_path.exists() and not force:
        return

    cand_path = _POOLS_DIR / "candidate_pool.parquet"
    hold_path = _HOLDOUT_PATH

    manifest: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "geometry_version": "Chapter 3 corrected (D5 outer wall, D7, slack-aware D4)",
        "oracle_module": "chapter3_clevis_setup.exact_assemblability",
        "design_space_module": "chapter3_clevis_setup.design_space",
        "label_taxonomy": list(REASON_ORDER),
        "head_length_rule": "5% of shaft length",
    }
    if cand_path.exists():
        cand = read_parquet(cand_path)
        reasons = {}
        for r in cand:
            reason = str(r.get("formula_reason", "unknown"))
            reasons[reason] = reasons.get(reason, 0) + 1
        manifest["candidate_pool"] = {
            "path": "candidate_pool.parquet",
            "n_rows": len(cand),
            "sha256": file_sha256(cand_path),
            "label_dist": {
                "assemblable": sum(1 for r in cand if int(r.get("label", 0)) == 1),
                "blocked": sum(1 for r in cand if int(r.get("label", 0)) == 0),
            },
            "formula_reason_dist": reasons,
        }
    if hold_path.exists():
        hold = read_parquet(hold_path)
        reasons = {}
        for r in hold:
            reason = str(r.get("formula_reason", "unknown"))
            reasons[reason] = reasons.get(reason, 0) + 1
        manifest["holdout"] = {
            "path": "holdout.parquet",
            "n_rows": len(hold),
            "sha256": file_sha256(hold_path),
            "label_dist": {
                "assemblable": sum(1 for r in hold if int(r.get("label", 0)) == 1),
                "blocked": sum(1 for r in hold if int(r.get("label", 0)) == 0),
            },
            "formula_reason_dist": reasons,
        }
    atomic_json_write(manifest_path, manifest)
    print(f"[build_pools] Manifest written: {manifest_path}")


# ─── Seed set generation ─────────────────────────────────────────────────


def write_seed_sets(n_splits: int = 8, master_size: int = 2500) -> None:
    """Generate nested base prefix seed sets from the candidate pool.

    For each split R, draw a stratified master list of `master_size` param_ids,
    then write prefix files for each base size B.
    """
    cand_path = _POOLS_DIR / "candidate_pool.parquet"
    if not cand_path.exists():
        raise FileNotFoundError("Run pool generation first")

    records = read_parquet(cand_path)
    pos_ids = [str(r["param_id"]) for r in records if int(r.get("label", 0)) == 1]
    neg_ids = [str(r["param_id"]) for r in records if int(r.get("label", 0)) == 0]

    base_sizes = [0, 250, 500, 750, 1000, 1500, 2000, 2500]

    for split_r in range(1, n_splits + 1):
        rng = np.random.default_rng(22000 + split_r)
        rng.shuffle(pos_ids)
        rng.shuffle(neg_ids)

        # Stratified: maintain ~same pos/neg ratio as pool
        neg_rate = len(neg_ids) / max(len(records), 1)
        n_neg = min(len(neg_ids), max(1, int(round(master_size * neg_rate))))
        n_pos = min(len(pos_ids), master_size - n_neg)

        master = list(neg_ids[:n_neg]) + list(pos_ids[:n_pos])
        rng.shuffle(master)
        master = master[:master_size]

        split_dir = _SEEDS_DIR / f"R{split_r:03d}"
        split_dir.mkdir(parents=True, exist_ok=True)

        # Write master
        atomic_json_write(
            split_dir / "master_2500.json",
            {"seed_split": split_r, "base_size": master_size, "param_ids": master},
        )

        # Write prefixes for each B
        for B in base_sizes:
            prefix = master[:B] if B <= len(master) else master
            atomic_json_write(
                split_dir / f"base_prefix_{B}.json",
                {"seed_split": split_r, "base_size": B, "param_ids": prefix},
            )

        print(f"[build_pools] Seed split R{split_r:03d}: master={len(master)}, "
              f"prefixes for B={base_sizes}")

    print(f"[build_pools] Wrote {n_splits} seed splits to {_SEEDS_DIR}")


# ─── CLI ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--n-candidates", type=int, default=22000)
    parser.add_argument("--n-holdout", type=int, default=5000)
    parser.add_argument("--candidate-seed", type=int, default=220001)
    parser.add_argument("--holdout-seed", type=int, default=500001)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--write-seed-sets", action="store_true")
    parser.add_argument("--seed-splits", type=int, default=8)
    args = parser.parse_args()

    if args.write_seed_sets:
        write_seed_sets(n_splits=args.seed_splits)
    else:
        build_candidate_pool(
            n_candidates=args.n_candidates,
            workers=args.workers,
            seed=args.candidate_seed,
            force=args.force,
        )
        build_holdout(
            n_holdout=args.n_holdout,
            workers=args.workers,
            seed=args.holdout_seed,
            force=args.force,
        )
        write_manifest(force=args.force)


if __name__ == "__main__":
    main()
