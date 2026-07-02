#!/usr/bin/env python3
"""Single-design CadQuery overlap verification (the one expensive call per run).

This is the *only* ground-truth step in an optimizer run: the workbench calls it
once, on the final selected design, to check whether the MLP's "clean" verdict is
real. It builds the airframe, computes the extrinsic cut-each wings-tail overlap
with the fast early-exit path, and returns a small dict. No meshes are exported
here (rendering is on-demand only, in `render_worker.py`).
"""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

import contextlib
import io
import signal
import sys
import threading
import time

# Hard cap on a single CadQuery verification before the soft watchdog fires.
# A healthy cut-each overlap on a UAV airframe is ~0.5-2 s; anything beyond this
# is a pathological geometry and is treated as a build failure for that start so
# the subgroup keeps going (see run_sweep.py for the hard subprocess-level cap).
DEFAULT_VERIFY_TIMEOUT_S = 180.0  # worst-overlap finals build in ~70s unloaded; headroom under 14-way load


class VerifyTimeout(Exception):
    """Raised by the SIGALRM watchdog when a single verification runs too long."""


def _can_arm_alarm() -> bool:
    # setitimer/SIGALRM only deliver to the main thread of a process; pool
    # workers run their task in the main thread so this holds there.
    return hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()


def verify_overlap(adv: dict, *, timeout_s: float | None = DEFAULT_VERIFY_TIMEOUT_S) -> dict:
    """Build `adv` and return ground-truth extrinsic wing-tail overlap (mm^3).

    A soft per-call watchdog (SIGALRM) aborts a verification that exceeds
    `timeout_s`; the result is returned as a build failure with an explanatory
    error instead of wedging the worker. Pass ``timeout_s=None`` to disable.
    """
    from chapter7_aeroforge.overlap_search.core import (  # noqa: PLC0415
        build_airframe,
        cadquery_cut_each_overlap_fast,
        is_overlap,
    )

    t0 = time.perf_counter()
    out = {
        "build_ok": False,
        "verified_overlap_mm3": None,
        "verified_raw_overlap_mm3": None,
        "is_overlap": None,
        "error": None,
        "wall_s": None,
        "timed_out": False,
    }

    armed = False
    previous_handler = None
    if timeout_s and timeout_s > 0 and _can_arm_alarm():
        def _on_alarm(signum, frame):  # noqa: ANN001, ARG001
            raise VerifyTimeout(f"verification exceeded {timeout_s:.0f}s")

        previous_handler = signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
        armed = True

    try:
        # AeroForge prints verbose DEBUG lines on every build; keep logs readable.
        with contextlib.redirect_stdout(io.StringIO()):
            uav, _timings = build_airframe(adv)
            wings = uav.components["wings"]
            tail = uav.components["tail"]
            fuselage = uav.components["fuselage_outer"]
            metrics = cadquery_cut_each_overlap_fast(wings, tail, fuselage)
        overlap = float(metrics["overlap_mm3_cut_each"])
        out.update(
            {
                "build_ok": True,
                "verified_overlap_mm3": overlap,
                "verified_raw_overlap_mm3": float(metrics["overlap_mm3_raw"]),
                "is_overlap": bool(is_overlap(overlap)),
            }
        )
    except VerifyTimeout as exc:
        out["error"] = f"VerifyTimeout: {exc}"
        out["timed_out"] = True
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if armed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)

    out["wall_s"] = time.perf_counter() - t0
    return out
