"""Params-only candidate pool generation for Chapter 6."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .paths import DATA_DIR
from .sampler import rows_from_samples, sample_relaxed_batch


def write_pool(*, n: int, seed: int, out: Path, prefix: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    samples = sample_relaxed_batch(n, seed=seed, prefix=prefix)
    rows = rows_from_samples(samples)
    csv_path = out / "pool.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "n": n,
        "seed": seed,
        "prefix": prefix,
        "pool_csv": str(csv_path),
        "contains_overlap_labels": False,
        "sampler_version": "relaxed_sampler_v1",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# Chapter 6 Params-Only Candidate Pool\n\n"
        f"- Rows: `{n}`\n"
        f"- Seed: `{seed}`\n"
        f"- Prefix: `{prefix}`\n"
        "- Contains overlap labels: `false`\n\n"
        "Labels must be acquired on demand through the label cache.\n",
        encoding="utf-8",
    )
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=252500)
    parser.add_argument("--name", default="generated_pool")
    parser.add_argument("--prefix", default="t25_pool")
    args = parser.parse_args()
    csv_path = write_pool(
        n=args.n,
        seed=args.seed,
        out=DATA_DIR / args.name,
        prefix=args.prefix,
    )
    print(csv_path)


if __name__ == "__main__":
    main()
