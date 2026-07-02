"""Build 50k candidate pool and seed sets for Chapter 4 Dense-Pool Rerun.

Uses the SAME holdout from the 22k run for comparability.
Generates 50,000 unique valid candidates excluding holdout IDs.

Usage:
    python build_pools_50k.py --workers 16 --force
    python build_pools_50k.py --seed-splits 8 --write-seed-sets
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
    pool_key,
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

_DEFAULT_POOL_ID = "dense50k_v1"
_DEFAULT_POOL_FILE = "candidate_pool.parquet"
_DEFAULT_MANIFEST_FILE = "pool_manifest.json"
_HOLDOUT_SRC = HOLDOUT_PATH


def _pool_dir(pool_id: str) -> Path:
    return DATASETS_ROOT / pool_key(pool_id)


def _seeds_dir(pool_id: str) -> Path:
    from release_paths import seed_sets_root_for_pool

    return seed_sets_root_for_pool(pool_id)


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


def _load_holdout_ids() -> set[str]:
    """Load param_ids from existing holdout to exclude from pool."""
    if not _HOLDOUT_SRC.exists():
        print(f"[build_pools_50k] WARNING: holdout not found at {_HOLDOUT_SRC}")
        return set()
    records = read_parquet(_HOLDOUT_SRC)
    ids = set()
    for r in records:
        if "param_id" in r:
            ids.add(str(r["param_id"]))
        h = param_hash(r)
        ids.add(h)
    print(f"[build_pools_50k] Loaded {len(ids)} holdout IDs for exclusion")
    return ids


def build_candidate_pool(
    n_candidates: int = 50000,
    workers: int = 16,
    seed: int = 550001,
    force: bool = False,
    pool_id: str = _DEFAULT_POOL_ID,
    pool_file: str = _DEFAULT_POOL_FILE,
) -> Path:
    """Generate and save a candidate-pool parquet file.

    Loops in batches until exactly n_candidates unique valid rows are collected.
    Excludes all holdout IDs by param_hash.
    """
    pools_dir = _pool_dir(pool_id)
    out_path = pools_dir / pool_file
    if out_path.exists() and not force:
        print(f"[build_pools_50k] candidate pool exists: {out_path}")
        return out_path

    pools_dir.mkdir(parents=True, exist_ok=True)

    holdout_ids = _load_holdout_ids()
    print(f"[build_pools_50k] Generating {n_candidates} candidates with {workers} workers...")
    print(f"[build_pools_50k] Seed base: {seed}, excluding {len(holdout_ids)} holdout hashes")
    t0 = time.perf_counter()

    seen: set[str] = set(holdout_ids)
    deduped: list[dict[str, Any]] = []
    batch_seed = seed
    batch_size = n_candidates * 2
    attempt = 0
    total_generated = 0
    total_rejected_invalid = 0
    total_rejected_dup = 0
    total_rejected_holdout = 0
    stream_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    while len(deduped) < n_candidates:
        attempt += 1
        chunk_size = batch_size // workers
        chunks = [(batch_seed + i * chunk_size, chunk_size) for i in range(workers)]

        with Pool(processes=workers) as pool:
            chunk_results = pool.map(_generate_pool_worker, chunks)

        for chunk in chunk_results:
            for rec in chunk:
                total_generated += 1
                h = param_hash(rec)
                if h in holdout_ids:
                    total_rejected_holdout += 1
                    continue
                if h in seen:
                    total_rejected_dup += 1
                    continue
                seen.add(h)
                rec["param_id"] = h
                rec["source"] = f"{pool_id}_gen_seed{seed}_batch{attempt}"
                mode = (batch_seed + total_generated) % 5
                stream_counts[mode] = stream_counts.get(mode, 0) + 1
                deduped.append(rec)
                if len(deduped) >= n_candidates:
                    break
            if len(deduped) >= n_candidates:
                break

        batch_seed += batch_size
        elapsed = time.perf_counter() - t0
        print(f"  batch {attempt}: {len(deduped)}/{n_candidates} collected "
              f"({elapsed:.1f}s elapsed)", flush=True)

    deduped = deduped[:n_candidates]
    elapsed = time.perf_counter() - t0
    print(f"[build_pools_50k] Pool complete: {len(deduped)} rows in {elapsed:.1f}s")
    print(f"  Total attempts: {total_generated + (batch_size * attempt - total_generated)}")
    print(f"  Rejected (holdout): {total_rejected_holdout}")
    print(f"  Rejected (dup): {total_rejected_dup}")

    records_to_parquet(deduped, out_path)
    return out_path


def write_manifest(
    force: bool = False,
    pool_id: str = _DEFAULT_POOL_ID,
    pool_file: str = _DEFAULT_POOL_FILE,
    manifest_file: str = _DEFAULT_MANIFEST_FILE,
    pool_seed_base: int = 550001,
) -> None:
    """Write a pool manifest with file hashes, row counts, and version markers."""
    pools_dir = _pool_dir(pool_id)
    manifest_path = pools_dir / manifest_file
    if manifest_path.exists() and not force:
        print(f"[build_pools_50k] manifest exists: {manifest_path}")
        return

    cand_path = pools_dir / pool_file

    manifest: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": pool_id,
        "geometry_version": "Chapter 3 corrected (D5 outer wall, D7, slack-aware D4, 5% head lengths)",
        "oracle_module": "chapter3_clevis_setup.exact_assemblability",
        "design_space_module": "chapter3_clevis_setup.design_space",
        "label_taxonomy": list(REASON_ORDER),
        "head_length_rule": "5% of shaft length",
        "holdout_source": str(_HOLDOUT_SRC),
        "holdout_exclusion": True,
        "pool_seed_base": pool_seed_base,
        "proposal_distribution": "rejection-conditioned smart_sampler mix (uniform/boundary/extreme/lhs/uniform)",
    }

    if cand_path.exists():
        cand = read_parquet(cand_path)
        reasons = {}
        for r in cand:
            reason = str(r.get("formula_reason", "unknown"))
            reasons[reason] = reasons.get(reason, 0) + 1
        labels_pos = sum(1 for r in cand if int(r.get("label", 0)) == 1)
        labels_neg = sum(1 for r in cand if int(r.get("label", 0)) == 0)
        manifest["candidate_pool"] = {
            "path": pool_file,
            "n_rows": len(cand),
            "sha256": file_sha256(cand_path),
            "label_dist": {
                "assemblable": labels_pos,
                "blocked": labels_neg,
                "assemblable_rate": round(labels_pos / max(len(cand), 1), 4),
                "blocked_rate": round(labels_neg / max(len(cand), 1), 4),
            },
            "formula_reason_dist": reasons,
        }

    holdout_path = _HOLDOUT_SRC
    if holdout_path.exists():
        hold = read_parquet(holdout_path)
        reasons_h = {}
        for r in hold:
            reason = str(r.get("formula_reason", "unknown"))
            reasons_h[reason] = reasons_h.get(reason, 0) + 1
        manifest["holdout"] = {
            "source_path": str(holdout_path),
            "n_rows": len(hold),
            "sha256": file_sha256(holdout_path),
            "label_dist": {
                "assemblable": sum(1 for r in hold if int(r.get("label", 0)) == 1),
                "blocked": sum(1 for r in hold if int(r.get("label", 0)) == 0),
            },
            "formula_reason_dist": reasons_h,
        }

    atomic_json_write(manifest_path, manifest)
    print(f"[build_pools_50k] Manifest written: {manifest_path}")


# ─── Seed set generation ─────────────────────────────────────────────────


def write_seed_sets(
    n_splits: int = 8,
    master_size: int = 2500,
    pool_id: str = _DEFAULT_POOL_ID,
    pool_file: str = _DEFAULT_POOL_FILE,
) -> None:
    """Generate nested base prefix seed sets from a candidate pool.

    For each split R, draw a stratified master list of `master_size` param_ids,
    then write prefix files for each base size B.
    """
    pools_dir = _pool_dir(pool_id)
    seeds_dir = _seeds_dir(pool_id)
    cand_path = pools_dir / pool_file
    if not cand_path.exists():
        raise FileNotFoundError(f"Run pool generation first: {cand_path}")

    records = read_parquet(cand_path)
    pos_ids = [str(r["param_id"]) for r in records if int(r.get("label", 0)) == 1]
    neg_ids = [str(r["param_id"]) for r in records if int(r.get("label", 0)) == 0]

    base_sizes = [0, 250, 500, 750, 1000, 1500, 2000, 2500]

    for split_r in range(1, n_splits + 1):
        rng = np.random.default_rng(55000 + split_r)
        rng.shuffle(pos_ids)
        rng.shuffle(neg_ids)

        neg_rate = len(neg_ids) / max(len(records), 1)
        n_neg = min(len(neg_ids), max(1, int(round(master_size * neg_rate))))
        n_pos = min(len(pos_ids), master_size - n_neg)

        master = list(neg_ids[:n_neg]) + list(pos_ids[:n_pos])
        rng.shuffle(master)
        master = master[:master_size]

        split_dir = seeds_dir / f"R{split_r:03d}"
        split_dir.mkdir(parents=True, exist_ok=True)

        atomic_json_write(
            split_dir / "master_2500.json",
            {"seed_split": split_r, "base_size": master_size, "param_ids": master},
        )

        for B in base_sizes:
            prefix = master[:B] if B <= len(master) else master
            atomic_json_write(
                split_dir / f"base_prefix_{B}.json",
                {"seed_split": split_r, "base_size": B, "param_ids": prefix},
            )

        print(f"[build_pools_50k] Seed split R{split_r:03d}: master={len(master)}, "
              f"prefixes for B={base_sizes}")

    print(f"[build_pools_50k] Wrote {n_splits} seed splits to {seeds_dir}")


# ─── CLI ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--n-candidates", type=int, default=50000)
    parser.add_argument("--candidate-seed", type=int, default=550001)
    parser.add_argument("--pool-id", default=_DEFAULT_POOL_ID)
    parser.add_argument("--pool-file", default=_DEFAULT_POOL_FILE)
    parser.add_argument("--manifest-file", default=_DEFAULT_MANIFEST_FILE)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--write-seed-sets", action="store_true")
    parser.add_argument("--seed-splits", type=int, default=8)
    args = parser.parse_args()

    if args.write_seed_sets:
        write_seed_sets(
            n_splits=args.seed_splits,
            pool_id=args.pool_id,
            pool_file=args.pool_file,
        )
    else:
        build_candidate_pool(
            n_candidates=args.n_candidates,
            workers=args.workers,
            seed=args.candidate_seed,
            force=args.force,
            pool_id=args.pool_id,
            pool_file=args.pool_file,
        )
        write_manifest(
            force=args.force,
            pool_id=args.pool_id,
            pool_file=args.pool_file,
            manifest_file=args.manifest_file,
            pool_seed_base=args.candidate_seed,
        )


if __name__ == "__main__":
    main()
