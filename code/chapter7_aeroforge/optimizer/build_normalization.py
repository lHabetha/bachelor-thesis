#!/usr/bin/env python3
"""Compute data-driven normalization bounds for the optimizer's numeric drivers.

The 100k ADV dataset was generated with a much wider, multi-family distribution
than the legacy `sampler.py` ranges, so normalizing against the sampler bounds
clips essentially every design. The correct normalized space is the *actual MLP
training domain*: the per-driver empirical [min, max] over the labeled rows the
surrogate saw. This writes `normalization_bounds.json`, which `model_io.py` loads
so that every start maps to u in [0, 1] and the optimizer stays in-distribution.
"""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

import argparse
import json
import sys
from pathlib import Path


from chapter7_aeroforge.overlap_search.sampler import (  # noqa: E402
    BACKGROUND_UNIFORM,
    DRIVER_SPECS,
)

from chapter7_aeroforge.release_paths import LABELS_CLOUD, NORMALIZATION_JSON

DEFAULT_LABELS = LABELS_CLOUD
DEFAULT_OUT = NORMALIZATION_JSON
NUMERIC_KEYS = sorted({*DRIVER_SPECS.keys(), *BACKGROUND_UNIFORM.keys()})


def main() -> None:
    ap = argparse.ArgumentParser(description="Build data-driven normalization bounds")
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--ok-only", action="store_true", default=True)
    args = ap.parse_args()

    lo: dict[str, float] = {k: float("inf") for k in NUMERIC_KEYS}
    hi: dict[str, float] = {k: float("-inf") for k in NUMERIC_KEYS}
    n = 0
    with args.labels.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if args.ok_only and not row.get("ok", True):
                continue
            adv = row.get("adv") or {}
            n += 1
            for k in NUMERIC_KEYS:
                v = adv.get(k)
                if isinstance(v, (int, float)):
                    lo[k] = min(lo[k], float(v))
                    hi[k] = max(hi[k], float(v))

    bounds = {}
    for k in NUMERIC_KEYS:
        if lo[k] == float("inf") or hi[k] <= lo[k]:
            # constant or unseen driver: fall back to a tiny window around it.
            base = lo[k] if lo[k] != float("inf") else 0.0
            bounds[k] = [base, base + 1.0]
        else:
            bounds[k] = [lo[k], hi[k]]

    payload = {
        "source_labels": str(args.labels.resolve()),
        "n_rows": n,
        "ok_only": bool(args.ok_only),
        "note": "Per-driver empirical [min, max] over labeled rows = MLP training domain.",
        "bounds": bounds,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} from {n} rows")
    for k in NUMERIC_KEYS:
        print(f"  {k:24s} [{bounds[k][0]:10.3f}, {bounds[k][1]:10.3f}]")


if __name__ == "__main__":
    main()
