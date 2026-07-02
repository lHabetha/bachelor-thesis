#!/usr/bin/env python3
"""Deterministic full-VLM quick-sim performance pass for the Chapter 7 grid.

Purpose:
  * Run the AeroForge **quick sim** (``generate_quick_report`` internals:
    ``collect_aero_metrics`` + ``maybe_add_asb_vlm_metrics`` — the *full VLM* path,
    the highest-definition deterministic sim the quick tier offers) ONCE on each
    of the 100 benchmark start assemblies (theta_0), and again on every optimized
    final (theta*) of every optimizer run. From the before/after pairs we get
    per-start and aggregate Delta L/D, Delta CD0, Delta CD (drag), etc.

Why a separate pass (not inside ``workbench``):
  * The optimization workers build CadQuery geometry (~1.8 GB RSS each); the VLM
    quick sim needs aerosandbox/casadi instead (~0.3-0.4 GB). Running them in the
    *same* worker would double RAM at 14-way fan-out. So the grid runs lean
    (CAD only), and this pass runs afterwards lean (VLM only).
  * It is **resumable** at two levels: a global ADV-hash sim cache
    (``runs/_sim_cache/<hash>.json`` — the 100 shared baselines are simmed once,
    identical/again-seen finals are deduped) and a per-run ``sim_eval.json``
    completion check. A dead laptop only costs the in-flight run.

Determinism: the VLM is a closed-form panel solve (no RNG); identical ADV ->
identical metrics. The cache is therefore exact, not approximate.

Weight/mass caveat: the quick sim deliberately omits the mass block
(``include_mass_block=False``); weight needs ``compute_uav_mass_cog`` which
builds CadQuery solids (the slow full-report path, minutes/design). It is
therefore NOT produced here. Every aero/VLM scalar the quick report does expose
is tracked, so L/D, CD0, CD, CDi, e, Re, Cm, NP/CoP, root bending/shear all get
deltas; ``weight`` is reported as unavailable with a note.

Run (standalone, after the grid):
    conda run --no-capture-output -n bachelor-thesis python3 -m \
        chapter7_aeroforge.optimizer.quicksim_eval --runs-dir <runs> --workers 14

It is also invoked automatically at the end of ``run_sweep`` (``--sim`` on by default).
"""

from __future__ import annotations

from chapter7_aeroforge.release_paths import ensure_code_importable

ensure_code_importable()

import argparse
import hashlib
import io
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The batch labeler already solves every hard part of running the quick sim under
# a process pool: BLAS/OMP thread pinning at import, a RAM-saving import that
# blocks the optional CadQuery geom hook, a per-worker VLM warmup, and a clean
# (curated_json, vlm_error) contract. Reuse it verbatim so this pass and the 100k
# labeler stay byte-for-byte consistent in *how* a design is simmed.
from chapter7_aeroforge.quicksim import label_quicksim_forever as L  # noqa: E402

from chapter7_aeroforge.release_paths import BENCHMARK_JSON, RUNS_DIR

DEFAULT_RUNS_DIR = RUNS_DIR
DEFAULT_BENCHMARK = BENCHMARK_JSON
DEFAULT_WORKERS = 14
SIM_SCHEMA_VERSION = "sim_eval/1.0"

# Curated headline scalars diffed start->final. L_over_D/CD0 use the quick
# report's own VLM-preferring fallback (label_quicksim_forever._headline); the
# rest are read straight from the asb_vlm / aero_simple / summary blocks.
DELTA_KEYS = [
    "L_over_D",
    "CD0",
    "CD_from_vlm",
    "CDi_from_vlm",
    "CL_from_vlm",
    "L_over_D_from_vlm",
    "Cm_from_vlm",
    "e",
    "e_fit_from_vlm",
    "Re_c_ref",
    "CD0_wing",
    "CD0_tail",
    "CD0_fuselage",
    "neutral_point_x_m_from_vlm",
    "CoP_x_m_from_vlm",
    "bending_moment_root_Nm_from_vlm",
    "shear_max_N_from_vlm",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _full_adv_hash(adv: dict) -> str:
    """Hash the FULL ADV (numeric rounded to 6 dp + string keys verbatim).

    Unlike ``workbench._adv_hash`` (numeric-only, fine for verify dedup because
    optimizers never move airfoil-path strings), the sim depends on every ADV key
    merged onto the defaults, so string keys (e.g. NACA CSV paths) must be in the
    key to avoid collisions across designs that differ only there.
    """
    items = sorted(
        (k, (round(float(v), 6) if isinstance(v, (int, float)) else v)) for k, v in adv.items()
    )
    return hashlib.sha1(repr(items).encode("utf-8")).hexdigest()[:16]


def _finite(v: Any) -> float | None:
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def _headline_metrics(quick: dict) -> dict:
    """Pull the curated aero/VLM scalars we delta from one quick-report JSON."""
    perf = quick.get("uav_performance", {}) if isinstance(quick, dict) else {}
    aero = perf.get("aero_simple", {}) or {}
    asb = perf.get("asb_vlm", {}) or {}
    summ = perf.get("summary", {}) or {}
    ld, cd0, status = L._headline(quick)
    return {
        "L_over_D": ld,
        "CD0": cd0,
        "L_over_D_from_vlm": _finite(asb.get("L_over_D_from_vlm")),
        "CD_from_vlm": _finite(asb.get("CD_from_vlm")),
        "CDi_from_vlm": _finite(asb.get("CDi_from_vlm")),
        "CL_from_vlm": _finite(asb.get("CL_from_vlm")),
        "Cm_from_vlm": _finite(asb.get("Cm_from_vlm")),
        "e": _finite(aero.get("e")),
        "e_fit_from_vlm": _finite(asb.get("e_fit_from_vlm")),
        "Re_c_ref": _finite(aero.get("Re_c_ref")),
        "CD0_wing": _finite(aero.get("CD0_wing")),
        "CD0_tail": _finite(aero.get("CD0_tail")),
        "CD0_fuselage": _finite(aero.get("CD0_fuselage")),
        "neutral_point_x_m_from_vlm": _finite(asb.get("neutral_point_x_m_from_vlm")),
        "CoP_x_m_from_vlm": _finite(asb.get("CoP_x_m_from_vlm")),
        "bending_moment_root_Nm_from_vlm": _finite(asb.get("bending_moment_root_Nm_from_vlm")),
        "shear_max_N_from_vlm": _finite(asb.get("shear_max_N_from_vlm")),
        "min_drag_coefficient_CD0": _finite(summ.get("min_drag_coefficient_CD0")),
        "vlm_status": status,
    }


def _sim_one(adv: dict) -> dict:
    """Worker task: full VLM quick sim on one ADV merged onto the defaults."""
    sim = L._load_sim()
    params = {**sim["DEFAULT_PARAMS"], **adv}
    t0 = time.perf_counter()
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            quick, vlm_error = L._run_quick_sim(params)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "sim_s": time.perf_counter() - t0}
    hm = _headline_metrics(quick)
    return {
        "ok": True,
        "headline": hm,
        "quick_report": quick,
        "vlm_error": vlm_error,
        "vlm_status": hm["vlm_status"],
        "vlm_ok": hm["vlm_status"] == "ok",
        "sim_s": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# Global ADV-hash sim cache (one file per unique geometry; atomic; resumable)
# ---------------------------------------------------------------------------


def _cache_path(cache_dir: Path, h: str) -> Path:
    return cache_dir / f"{h}.json"


def _cache_get(cache_dir: Path, h: str, mem: dict) -> dict | None:
    if h in mem:
        return mem[h]
    p = _cache_path(cache_dir, h)
    if p.exists():
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            mem[h] = rec
            return rec
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _cache_put(cache_dir: Path, h: str, rec: dict, mem: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = _cache_path(cache_dir, h)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, separators=(",", ":"), default=str), encoding="utf-8")
    os.replace(tmp, p)
    mem[h] = rec


def _sim_missing(
    advs_by_hash: dict[str, dict],
    cache_dir: Path,
    mem: dict,
    executor: ProcessPoolExecutor,
) -> None:
    """Ensure every (hash -> adv) has a cache entry; sim the misses in parallel."""
    missing = {h: adv for h, adv in advs_by_hash.items() if _cache_get(cache_dir, h, mem) is None}
    if not missing:
        return
    hashes = list(missing)
    futs = {executor.submit(_sim_one, missing[h]): h for h in hashes}
    for fut in futs:
        h = futs[fut]
        try:
            res = fut.result()
        except Exception as exc:  # noqa: BLE001
            res = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        rec = {
            "adv_hash": h,
            "ok": bool(res.get("ok")),
            "vlm_status": res.get("vlm_status"),
            "vlm_ok": bool(res.get("vlm_ok")),
            "vlm_error": res.get("vlm_error"),
            "error": res.get("error"),
            "sim_s": res.get("sim_s"),
            "headline": res.get("headline"),
            "quick_report": res.get("quick_report"),
            "computed_at": _now(),
        }
        _cache_put(cache_dir, h, rec, mem)


# ---------------------------------------------------------------------------
# Per-run assembly: before/after headline + deltas + aggregate
# ---------------------------------------------------------------------------


def _headline_of(cache_rec: dict | None) -> dict | None:
    if cache_rec and cache_rec.get("ok") and isinstance(cache_rec.get("headline"), dict):
        return cache_rec["headline"]
    return None


def _delta(final: dict | None, start: dict | None) -> dict:
    out: dict[str, float] = {}
    if not final or not start:
        return out
    for k in DELTA_KEYS:
        a, b = _finite(final.get(k)), _finite(start.get(k))
        if a is not None and b is not None:
            out[k] = a - b
    return out


def _mean(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def assemble_run_sim(vd: dict, cache_dir: Path, mem: dict) -> dict:
    """Build a run's sim_eval payload from its viewer_data + the (populated) cache."""
    # CAD was really run iff at least one start carries a verified_clean bool.
    # (Do NOT trust stats["verification"]: it is the string "none"/"cad", and the
    # All meta-run hardcodes verify=True in its stats even on a --no-verify grid;
    # also bool("none") is truthy. The per-start signal is the only reliable one,
    # and it is correct for the All run too — its rows are copied from the siblings,
    # so they carry real verified_clean only when the grid actually verified.)
    verify_on = any(s.get("verified_clean") is not None for s in vd.get("starts", []))
    basis = "verified_clean" if verify_on else "mlp_clean"

    per_start: list[dict] = []
    d_ld, d_cd0, d_cd, d_cdi, rel_ld = [], [], [], [], []
    base_ld, fin_ld = [], []
    n_changed = n_success = n_both_vlm_ok = 0
    start_vlm_ok = final_vlm_ok = 0

    for s in vd.get("starts", []):
        adv0, advf = s.get("adv_start"), s.get("adv_final")
        if not isinstance(adv0, dict) or not isinstance(advf, dict):
            continue
        h0, hf = _full_adv_hash(adv0), _full_adv_hash(advf)
        rec0, recf = _cache_get(cache_dir, h0, mem), _cache_get(cache_dir, hf, mem)
        hs, hf_m = _headline_of(rec0), _headline_of(recf)
        changed = bool(s.get("changed"))
        verified_clean = s.get("verified_clean")
        mlp_clean = bool(s.get("mlp_clean"))
        success = bool(verified_clean) if verify_on else mlp_clean
        s0_ok = bool(rec0 and rec0.get("vlm_ok"))
        sf_ok = bool(recf and recf.get("vlm_ok"))
        delta = _delta(hf_m, hs)

        per_start.append(
            {
                "rank": s.get("rank"),
                "sample_idx": s.get("sample_idx"),
                "status": s.get("status"),
                "changed": changed,
                "success": success,
                "adv_start_hash": h0,
                "adv_final_hash": hf,
                "start_vlm_ok": s0_ok,
                "final_vlm_ok": sf_ok,
                "start": hs,
                "final": hf_m,
                "delta": delta,
            }
        )

        if changed:
            n_changed += 1
        if s0_ok:
            start_vlm_ok += 1
        if sf_ok:
            final_vlm_ok += 1
        # Aggregate Delta only over SUCCESSFUL repairs with both sims VLM-ok and
        # finite L/D — the thesis "over the repairs they find" convention, and a
        # guard against the pathological max-overlap starts whose VLM is degenerate.
        if success and s0_ok and sf_ok and hs and hf_m:
            n_both_vlm_ok += 1
            n_success += 1
            if hs.get("L_over_D") is not None:
                base_ld.append(hs["L_over_D"])
            if hf_m.get("L_over_D") is not None:
                fin_ld.append(hf_m["L_over_D"])
            if "L_over_D" in delta:
                d_ld.append(delta["L_over_D"])
                b = hs.get("L_over_D")
                if b not in (None, 0):
                    rel_ld.append(delta["L_over_D"] / abs(b))
            if "CD0" in delta:
                d_cd0.append(delta["CD0"])
            if "CD_from_vlm" in delta:
                d_cd.append(delta["CD_from_vlm"])
            if "CDi_from_vlm" in delta:
                d_cdi.append(delta["CDi_from_vlm"])

    aggregate = {
        "success_basis": basis,
        "n_starts": len(per_start),
        "n_changed": n_changed,
        "n_success": n_success,
        "n_success_with_both_vlm_ok": n_both_vlm_ok,
        "start_vlm_ok_rate": (start_vlm_ok / len(per_start)) if per_start else None,
        "final_vlm_ok_rate": (final_vlm_ok / len(per_start)) if per_start else None,
        "baseline_mean_L_over_D": _mean(base_ld),
        "baseline_median_L_over_D": _median(base_ld),
        "final_mean_L_over_D": _mean(fin_ld),
        "final_median_L_over_D": _median(fin_ld),
        "mean_delta_L_over_D": _mean(d_ld),
        "median_delta_L_over_D": _median(d_ld),
        "mean_rel_delta_L_over_D": _mean(rel_ld),
        "mean_delta_CD0": _mean(d_cd0),
        "median_delta_CD0": _median(d_cd0),
        "mean_delta_CD": _mean(d_cd),
        "median_delta_CD": _median(d_cd),
        "mean_delta_CDi": _mean(d_cdi),
        "mean_delta_weight": None,
        "weight_note": "unavailable in the VLM quick sim (needs CAD compute_uav_mass_cog / full report)",
    }
    return {
        "schema": SIM_SCHEMA_VERSION,
        "run_id": vd.get("run_id"),
        "group_id": vd.get("group_id"),
        "subgroup_id": vd.get("subgroup_id"),
        "model_id": vd.get("model_id"),
        "optimizer_id": vd.get("optimizer_id"),
        "optimizer_label": vd.get("optimizer_label"),
        "benchmark_id": vd.get("benchmark_id"),
        "generated_at": _now(),
        "aggregate": aggregate,
        "starts": per_start,
    }


# ---------------------------------------------------------------------------
# Pass driver
# ---------------------------------------------------------------------------


def _sim_meta() -> dict:
    sim = L._load_sim()
    sweep = None
    try:
        from sim_params import VLM_ALPHA_SWEEP_DEG  # noqa: PLC0415

        sweep = list(VLM_ALPHA_SWEEP_DEG)
    except Exception:  # noqa: BLE001
        pass
    return {
        "definition": "AeroForge generate_quick_report (collect_aero_metrics + maybe_add_asb_vlm_metrics); full VLM",
        "deterministic": True,
        "speed_mps": sim.get("speed"),
        "altitude_m": sim.get("alt"),
        "alpha_target_deg": sim.get("alpha"),
        "vlm_alpha_sweep_deg": sweep,
        "disabled_sections": ["mass", "energy", "takeoff", "stability"],
        "weight_available": False,
    }


def _discover_run_dirs(runs_dir: Path, only: set[str] | None) -> list[Path]:
    out: list[Path] = []
    for vd_path in sorted(runs_dir.glob("*/viewer_data.json")):
        rd = vd_path.parent
        if only is not None and rd.name not in only:
            continue
        out.append(rd)
    return out


def run_pass(
    *,
    runs_dir: Path,
    benchmark_path: Path,
    workers: int,
    only: set[str] | None,
    force: bool,
    no_warmup: bool,
    max_starts: int | None,
    max_runs: int | None,
) -> int:
    cache_dir = runs_dir / "_sim_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mem: dict[str, dict] = {}

    # Parent warmup (build the casadi/NeuralFoil on-disk caches once so 14 fresh
    # workers don't all race a cold cache). Restore CWD: _load_sim chdir's to aeroforge.
    cwd0 = os.getcwd()
    meta = _sim_meta()
    if not no_warmup:
        t = time.perf_counter()
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                sim = L._load_sim()
                L._run_quick_sim(dict(sim["DEFAULT_PARAMS"]))
            print(f"[quicksim_eval] parent warmup {time.perf_counter() - t:.1f}s (caches built)")
        except Exception as exc:  # noqa: BLE001
            print(f"[quicksim_eval] warmup failed (continuing): {type(exc).__name__}: {exc}")
    os.chdir(cwd0)

    executor = ProcessPoolExecutor(
        max_workers=workers, initializer=L.worker_init, mp_context=L._mp()
    )
    t0 = time.perf_counter()
    try:
        # --- baseline: the 100 overlapping starts, simmed once and cached -------
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        bstarts = benchmark.get("starts", [])
        if max_starts is not None:
            bstarts = bstarts[:max_starts]
        base_by_hash: dict[str, dict] = {}
        rank_for_hash: dict[str, dict] = {}
        for s in bstarts:
            adv = s.get("adv")
            if not isinstance(adv, dict):
                continue
            h = _full_adv_hash(adv)
            base_by_hash[h] = adv
            rank_for_hash.setdefault(h, {"rank": s.get("rank"), "sample_idx": s.get("sample_idx")})
        print(f"[quicksim_eval] baseline: {len(base_by_hash)} unique start designs ({benchmark.get('benchmark_id')})")
        _sim_missing(base_by_hash, cache_dir, mem, executor)
        baseline_rows = []
        for h, info in sorted(rank_for_hash.items(), key=lambda kv: (kv[1]["rank"] is None, kv[1]["rank"])):
            rec = _cache_get(cache_dir, h, mem)
            baseline_rows.append(
                {
                    "rank": info["rank"],
                    "sample_idx": info["sample_idx"],
                    "adv_hash": h,
                    "vlm_ok": bool(rec and rec.get("vlm_ok")),
                    "headline": _headline_of(rec),
                }
            )
        base_ld = [r["headline"]["L_over_D"] for r in baseline_rows if r["headline"] and r["headline"].get("L_over_D") is not None and r["vlm_ok"]]
        baseline_doc = {
            "schema": SIM_SCHEMA_VERSION,
            "benchmark_id": benchmark.get("benchmark_id"),
            "generated_at": _now(),
            "sim": meta,
            "n_starts": len(baseline_rows),
            "vlm_ok_count": sum(1 for r in baseline_rows if r["vlm_ok"]),
            "mean_L_over_D_vlm_ok": _mean(base_ld),
            "median_L_over_D_vlm_ok": _median(base_ld),
            "starts": baseline_rows,
        }
        bpath = cache_dir / f"baseline_{benchmark.get('benchmark_id')}.json"
        tmp = bpath.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(baseline_doc, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, bpath)
        print(f"[quicksim_eval] baseline -> {bpath.name}  (mean L/D vlm-ok = {_fmt(_mean(base_ld))})")

        # --- finals: every optimizer run's selected theta* ----------------------
        run_dirs = _discover_run_dirs(runs_dir, only)
        if max_runs is not None:
            run_dirs = run_dirs[:max_runs]
        print(f"[quicksim_eval] {len(run_dirs)} run dir(s) to evaluate")
        done = skipped = 0
        for i, rd in enumerate(run_dirs, 1):
            sim_eval_path = rd / "sim_eval.json"
            vd = json.loads((rd / "viewer_data.json").read_text(encoding="utf-8"))
            starts = vd.get("starts", [])
            if max_starts is not None:
                starts = starts[:max_starts]
                vd = {**vd, "starts": starts}
            if sim_eval_path.exists() and not force:
                try:
                    prev = json.loads(sim_eval_path.read_text(encoding="utf-8"))
                    if prev.get("schema") == SIM_SCHEMA_VERSION and prev.get("aggregate", {}).get("n_starts") == len(starts):
                        skipped += 1
                        print(f"  [{i:02d}/{len(run_dirs)}] {rd.name}  sim_eval present -> skip")
                        continue
                except (json.JSONDecodeError, OSError):
                    pass
            # collect this run's unique final designs (starts already cached)
            advs: dict[str, dict] = {}
            for s in starts:
                advf = s.get("adv_final")
                adv0 = s.get("adv_start")
                if isinstance(advf, dict):
                    advs[_full_adv_hash(advf)] = advf
                if isinstance(adv0, dict):
                    advs[_full_adv_hash(adv0)] = adv0
            _sim_missing(advs, cache_dir, mem, executor)
            payload = assemble_run_sim(vd, cache_dir, mem)
            payload["sim"] = meta
            tmp = sim_eval_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            os.replace(tmp, sim_eval_path)
            agg = payload["aggregate"]
            done += 1
            elapsed = time.perf_counter() - t0
            eta = (elapsed / (done + skipped)) * (len(run_dirs) - done - skipped) if (done + skipped) else 0.0
            print(
                f"  [{i:02d}/{len(run_dirs)}] {rd.name}  n_succ={agg['n_success']:3d}  "
                f"meanDLD={_fmt(agg['mean_delta_L_over_D'])} medDLD={_fmt(agg['median_delta_L_over_D'])} "
                f"meanDCD0={_fmt(agg['mean_delta_CD0'])} | elapsed {elapsed/60:.1f}m ETA {eta/60:.1f}m"
            )
        print(f"[quicksim_eval] FINISHED: evaluated={done} skipped={skipped} of {len(run_dirs)} in {(time.perf_counter()-t0)/60:.1f} min")
        print(f"[quicksim_eval] cache -> {cache_dir} ({len(list(cache_dir.glob('*.json')))} entries)")
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return 0


def _fmt(x: Any) -> str:
    if x is None:
        return "  n/a"
    try:
        return f"{float(x):+.4f}"
    except (TypeError, ValueError):
        return str(x)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deterministic full-VLM quick-sim performance pass over the optimization grid")
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--runs", type=str, default=None, help="Comma list of run_ids to (re)evaluate; default = all complete runs")
    ap.add_argument("--force", action="store_true", help="Recompute sim_eval.json even if present (cache still reused)")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--max-starts", type=int, default=None, help="Cap starts per run (smoke tests)")
    ap.add_argument("--max-runs", type=int, default=None, help="Cap number of runs (smoke tests)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    only = {r.strip() for r in args.runs.split(",")} if args.runs else None
    return run_pass(
        runs_dir=args.runs_dir.resolve(),
        benchmark_path=args.benchmark.resolve(),
        workers=args.workers,
        only=only,
        force=args.force,
        no_warmup=args.no_warmup,
        max_starts=args.max_starts,
        max_runs=args.max_runs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
