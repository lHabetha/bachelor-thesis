"""Emit the Chapter 6.4 repair matrix as shell commands (Chapter 6 #5f).

The Chapter 6.4 benchmark is a matrix of ``method x surrogate x tau`` cells, each
run over the frozen 50-start set. This script defines that matrix once and prints
one fully-quoted command per (cell x start-shard) job so a launcher (``xargs -P``,
GNU ``parallel``) can saturate the GCP box. Aggregation is shard-agnostic: every
per-start row carries its ``method`` label, so :mod:`export_ch64_tables` just globs
all ``summary.csv`` files and groups by method.

Phases:

- ``full``   -- the complete matrix (all metrics), sharded for parallelism.
- ``timing`` -- the wall-clock comparison subset, **one shard per cell** so it can be
  run uncontended (one job at a time) for faithful timing (Chapter 6).

Usage::

    python -m ...ch64_run_matrix --phase full --shards 5 --python "$PY" > jobs.full.sh
    python -m ...ch64_run_matrix --phase timing --python "$PY" > jobs.timing.sh
"""

from __future__ import annotations

import argparse
import shlex
from .release_paths import CH64_STARTS_CSV, POOL_100K_CSV, REPAIR_CH64_DIR, RESULTS_ROOT

MODULE = "chapter6_overlap.strict_repair_ch64"

# Each cell: method, model kind, and either tau_vol (single-head volume stop) or
# tau_bin (multitask binary stop). label is the run-id / grouping key.
FULL_MATRIX = [
    # A. v3 single-head volume-stop sweep tau_vol in {0, 1e-6, 1e-5}.
    {"label": "mlp__v3__tauvol0", "method": "mlp", "model_kind": "v3", "tau_vol": 0.0},
    {"label": "mlp__v3__tauvol1e-06", "method": "mlp", "model_kind": "v3", "tau_vol": 1e-6},
    {"label": "mlp__v3__tauvol1e-05", "method": "mlp", "model_kind": "v3", "tau_vol": 1e-5},
    # Candidate-primary surrogate methods (v3, analytic-verified stop).
    {"label": "hybrid_lite__v3__tauvol0", "method": "hybrid_lite", "model_kind": "v3", "tau_vol": 0.0},
    {"label": "hybrid_lite_reduced_calls__v3__tauvol0", "method": "hybrid_lite_reduced_calls", "model_kind": "v3", "tau_vol": 0.0},
    {"label": "finish_line__v3__tauvol0", "method": "finish_line", "model_kind": "v3", "tau_vol": 0.0},
    # Upper-bound reference.
    {"label": "full_hybrid__v3__tauvol0", "method": "full_hybrid", "model_kind": "v3", "tau_vol": 0.0},
    # Negative controls (no surrogate).
    {"label": "random_direct__none", "method": "random_direct", "model_kind": "none"},
    {"label": "axis_direct__none", "method": "axis_direct", "model_kind": "none"},
    # B. multitask: volume gradient + binary-head stop tau_bin in {0.05, 0.1, 0.2};
    #    plus an analytic-stop multitask baseline.
    {"label": "mlp__multitask__tauvol0", "method": "mlp", "model_kind": "multitask", "tau_vol": 0.0},
    {"label": "mlp__multitask__taubin0.05", "method": "mlp", "model_kind": "multitask", "tau_bin": 0.05},
    {"label": "mlp__multitask__taubin0.1", "method": "mlp", "model_kind": "multitask", "tau_bin": 0.1},
    {"label": "mlp__multitask__taubin0.2", "method": "mlp", "model_kind": "multitask", "tau_bin": 0.2},
    {"label": "hybrid_lite__multitask__taubin0.1", "method": "hybrid_lite", "model_kind": "multitask", "tau_bin": 0.1},
]

# Wall-clock comparison subset (matched-budget story): run uncontended for timing.
TIMING_LABELS = {
    "mlp__v3__tauvol0",
    "hybrid_lite__v3__tauvol0",
    "hybrid_lite_reduced_calls__v3__tauvol0",
    "finish_line__v3__tauvol0",
    "full_hybrid__v3__tauvol0",
    "random_direct__none",
    "axis_direct__none",
}


def _shard_bounds(n_starts: int, shards: int) -> list[tuple[int, int]]:
    base = n_starts // shards
    rem = n_starts % shards
    bounds = []
    offset = 0
    for i in range(shards):
        count = base + (1 if i < rem else 0)
        if count == 0:
            continue
        bounds.append((offset, count))
        offset += count
    return bounds


def _cell_command(cell: dict, args: argparse.Namespace, *, offset: int, count: int, run_id: str, out_root: str) -> str:
    parts = [
        args.python, "-m", MODULE,
        "--method", cell["method"],
        "--model-kind", cell["model_kind"],
        "--run-id", run_id,
        "--out-root", out_root,
        "--starts-csv", args.starts_csv,
        "--pool-csv", args.pool_csv,
        "--n-starts", str(count),
        "--start-offset", str(offset),
        "--method-label", cell["label"],
    ]
    if "tau_bin" in cell:
        parts += ["--tau-bin", str(cell["tau_bin"])]
    if "tau_vol" in cell:
        parts += ["--tau-vol", repr(float(cell["tau_vol"]))]
    return " ".join(shlex.quote(p) for p in parts)


def run(args: argparse.Namespace) -> None:
    if args.phase == "timing":
        for cell in FULL_MATRIX:
            if cell["label"] not in TIMING_LABELS:
                continue
            print(_cell_command(cell, args, offset=0, count=args.n_starts, run_id=cell["label"], out_root=args.timing_out_root))
        return
    for cell in FULL_MATRIX:
        for offset, count in _shard_bounds(args.n_starts, args.shards):
            run_id = f"{cell['label']}/shard_{offset:03d}_{offset + count:03d}"
            print(_cell_command(cell, args, offset=offset, count=count, run_id=run_id, out_root=args.out_root))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("full", "timing"), default="full")
    parser.add_argument("--shards", type=int, default=5, help="start-shards per cell (full phase)")
    parser.add_argument("--n-starts", type=int, default=50)
    parser.add_argument("--python", default="python")
    parser.add_argument(
        "--starts-csv",
        default=str(CH64_STARTS_CSV),
    )
    parser.add_argument(
        "--pool-csv",
        default=str(POOL_100K_CSV),
    )
    parser.add_argument("--out-root", default=str(REPAIR_CH64_DIR))
    parser.add_argument("--timing-out-root", default=str(RESULTS_ROOT / "repair_ch64_timed"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
