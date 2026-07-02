"""Multiprocessing worker for one-sample overlap labeling."""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from chapter7_aeroforge.release_paths import AEROFORGE_ROOT


def worker_init() -> None:
    # Pin numerics to 1 thread BEFORE heavy libs load (belt-and-suspenders;
    # OCCT itself is pinned separately via limit_occt_threads below).
    for _var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(_var, "1")
    sys.path.insert(0, str(AEROFORGE_ROOT))
    os.chdir(AEROFORGE_ROOT)
    from chapter7_aeroforge.overlap_search.core import (  # noqa: PLC0415
        ensure_aeroforge_imports,
        limit_occt_threads,
    )

    ensure_aeroforge_imports()
    # The decisive knob: cap each worker's OCCT thread pool so W workers occupy
    # ~W cores instead of oversubscribing W*cores threads. AEROFORGE_OCCT_THREADS=0
    # leaves OCCT at its default (all logical cores) for comparison.
    _n = int(os.environ.get("AEROFORGE_OCCT_THREADS", "1"))
    if _n > 0:
        limit_occt_threads(_n)


def process_sample(task: dict) -> dict:
    """Label one ADV sample; stdout/stderr silenced for AeroForge DEBUG spam."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from chapter7_aeroforge.overlap_search.core import label_sample  # noqa: PLC0415

    params = task["params"]
    meta = {
        "iteration": task["iteration"],
        "sample_idx": task["sample_idx"],
        "seed": task["seed"],
    }
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        row = label_sample(params)
    row.update(meta)
    return row
