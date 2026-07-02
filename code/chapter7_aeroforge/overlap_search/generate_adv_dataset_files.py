#!/usr/bin/env python3
"""Generate the fixed 100k ADV JSON dataset used by the resumable labeler."""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

import argparse
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


from chapter7_aeroforge.overlap_search.adv_dataset_generator import (  # noqa: E402
    family_weights,
    sample_adv_dataset_with_family,
)
from chapter7_aeroforge.overlap_search.core import ADV_KEYS  # noqa: E402

from chapter7_aeroforge.release_paths import DATA_ROOT

DEFAULT_OUT_DIR = DATA_ROOT
DEFAULT_N = 100_000
DEFAULT_SEED_BASE = 20260620


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate fixed ADV JSON files.")
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing ADV JSON files if present.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    adv_dir = out_dir / "advs"
    labels_dir = out_dir / "labels"
    adv_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    existing = list(adv_dir.glob("*.json"))
    if existing and not args.overwrite:
        raise SystemExit(
            f"{adv_dir} already contains {len(existing)} JSON files. "
            "Pass --overwrite to regenerate them."
        )
    if args.overwrite:
        for path in existing:
            path.unlink()

    family_counts: Counter[str] = Counter()
    t0 = time.perf_counter()
    for sample_idx in range(args.n):
        seed = args.seed_base + sample_idx
        rng = random.Random(seed)
        adv, family = sample_adv_dataset_with_family(rng)
        missing = [key for key in ADV_KEYS if key not in adv]
        if missing:
            raise RuntimeError(f"sample {sample_idx} missing ADV keys: {missing}")

        family_counts[family] += 1
        row = {
            "sample_idx": sample_idx,
            "seed": seed,
            "family": family,
            "adv": adv,
        }
        path = adv_dir / f"{sample_idx:06d}.json"
        path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")

        done = sample_idx + 1
        if done % 10_000 == 0 or done == args.n:
            elapsed = time.perf_counter() - t0
            print(f"generated {done:6d}/{args.n} ADV JSONs ({elapsed:.1f}s)", flush=True)

    manifest = {
        "created_at_utc": _utc_now(),
        "n": args.n,
        "seed_base": args.seed_base,
        "adv_dir": str(adv_dir),
        "file_pattern": "advs/{sample_idx:06d}.json",
        "schema": {
            "sample_idx": "integer 0-based dataset index",
            "seed": "seed_base + sample_idx",
            "family": "weighted generator family name",
            "adv": f"exactly {len(ADV_KEYS)} AeroForge ADV keys",
        },
        "generator": "chapter7_aeroforge.overlap_search.adv_dataset_generator.sample_adv_dataset_with_family",
        "family_weights": family_weights(),
        "family_counts": dict(sorted(family_counts.items())),
        "labels": {
            "path": str(labels_dir / "labels.jsonl"),
            "script": "chapter7_aeroforge/overlap_search/label_forever.py",
            "metric": "cadquery_cut_each_overlap_fast",
            "definition": "(wings - fuselage) intersect (tail - fuselage) > 1 mm^3",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    readme = f"""# Chapter 7 ADV Dataset

This folder contains the fixed 100,000-sample ADV pool for overlap-label training.

- `advs/000000.json` ... `advs/{args.n - 1:06d}.json`: one pre-generated 43-key ADV per file.
- `manifest.json`: generator seed, family weights, and realized family counts.
- `labels/labels.jsonl`: append-only labels written by `label_forever.py`.
- `labels/summary.json`: cumulative labeling progress.

Each ADV file stores:

```json
{{
  "sample_idx": 0,
  "seed": {args.seed_base},
  "family": "...",
  "adv": {{ "...": "43 ADV keys" }}
}}
```

Recreate the dataset:

```bash
conda run -n bachelor-thesis python3 -u \\
  chapter7_aeroforge/overlap_search/generate_adv_dataset_files.py --n {args.n} --overwrite
```

Resume labeling:

```bash
AEROFORGE_OCCT_THREADS=1 caffeinate -dimsu conda run -n bachelor-thesis python3 -u \\
  chapter7_aeroforge/overlap_search/label_forever.py --18
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    elapsed = time.perf_counter() - t0
    print(f"Done: generated {args.n} files in {elapsed:.1f}s at {adv_dir}", flush=True)


if __name__ == "__main__":
    main()
