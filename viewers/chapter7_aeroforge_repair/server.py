"""Chapter 7 AeroForge Repair Viewer Server (public release).

Serves a browser UI for inspecting the AeroForge overlap-repair grid
(``tab:aeroforge-overlap-repair`` / ``tab:aeroforge-overlap-repair-grid``) and the
performance-preserving repair sweep (``tab:aeroforge-aero-preserve``) reported in
Chapter 7 of the thesis. Shows per-start metrics, driver gradients, quick-sim
aero deltas, and on-demand CadQuery mesh export of the start and repaired final
airframes.

Only the runs listed in
``bachelor-thesis/release_manifests/ch7_allowlist.json`` are served.
Run JSON lives under ``data/<run_id>/`` next to this file (meshes are rendered
on demand).

Usage:
    python -m viewers.chapter7_aeroforge_repair.server
    python -m viewers.chapter7_aeroforge_repair.server --port 8092

Run from the ``bachelor-thesis/`` repository root inside the ``bachelor-thesis``
conda env so the Chapter 7 CAD code resolves for on-demand rendering. Requires a
local ``aeroforge/`` checkout (see ``code/chapter7_aeroforge/README.md``);
metrics-only inspection works without it.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CODE_ROOT = REPO_ROOT / "code"

DATA_DIR = SCRIPT_DIR / "data"
ALLOWLIST_PATH = REPO_ROOT / "release_manifests" / "ch7_allowlist.json"

for _path in (str(CODE_ROOT),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

PARTS = ("fuselage", "wings", "tail", "overlap")
_MIME = {".html": "text/html; charset=utf-8", ".json": "application/json", ".stl": "model/stl"}


def _load_allowlist() -> dict:
    if not ALLOWLIST_PATH.exists():
        return {"entries": []}
    return json.loads(ALLOWLIST_PATH.read_text())


_ALLOWLIST = _load_allowlist()
_ALLOWLIST_ENTRIES = _ALLOWLIST.get("entries", [])
_ALLOWED_RUN_IDS = {e["run_id"] for e in _ALLOWLIST_ENTRIES}
_ENTRY_TO_RUN = {e["entry_id"]: e["run_id"] for e in _ALLOWLIST_ENTRIES}
_ENTRY_BY_ID = {e["entry_id"]: e for e in _ALLOWLIST_ENTRIES}


def _resolve_run_id(key: str) -> str | None:
    key = unquote(key)
    if key in _ENTRY_TO_RUN:
        return _ENTRY_TO_RUN[key]
    if key in _ALLOWED_RUN_IDS:
        return key
    return None


def _run_dir(key: str) -> Path | None:
    run_id = _resolve_run_id(key)
    if run_id is None:
        return None
    candidate = (DATA_DIR / run_id).resolve()
    if not str(candidate).startswith(str(DATA_DIR.resolve())):
        return None
    if not (candidate / "viewer_data.json").exists():
        return None
    return candidate


def _discover_runs() -> list[dict]:
    """One record per allowlist entry, grouped for the dropdown."""
    runs: list[dict] = []
    for entry in _ALLOWLIST_ENTRIES:
        run_id = entry["run_id"]
        run_dir = DATA_DIR / run_id
        vd_path = run_dir / "viewer_data.json"
        if not vd_path.exists():
            continue
        try:
            vd = json.loads(vd_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        category = entry.get("category", "overlap_repair")
        if category == "aero_preserve":
            group_id = "aero_preserve"
            group_label = "Aero-preserving repair (tab:aeroforge-aero-preserve)"
        else:
            group_id = entry.get("group_id", vd.get("group_id", "ungrouped"))
            group_label = vd.get("group_label", group_id)
        runs.append({
            "entry_id": entry["entry_id"],
            "run_id": run_id,
            "title": vd.get("title") or run_id,
            "group_id": group_id,
            "group_label": group_label,
            "subgroup_id": vd.get("subgroup_id", run_id),
            "subgroup_label": entry.get("display_label") or vd.get("subgroup_label") or run_id,
            "model_id": vd.get("model_id"),
            "optimizer_id": vd.get("optimizer_id"),
            "n_starts": len(vd.get("starts", [])),
            "statistics": vd.get("statistics"),
            "thesis_anchors": entry.get("thesis_anchors", []),
            "category": category,
            "has_sim_eval": (run_dir / "sim_eval.json").exists(),
            "has_algorithm_md": (run_dir / "algorithm.md").exists(),
        })
    return runs


def _render_adv(adv: dict, out_dir: Path) -> dict:
    import cadquery as cq  # noqa: PLC0415

    from chapter7_aeroforge.overlap_search.core import (  # noqa: PLC0415
        _solid_volume,
        build_airframe,
        cut_fuselage_from_parts,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with contextlib.redirect_stdout(io.StringIO()):
        uav, _timings = build_airframe(adv)
        wings = uav.components["wings"]
        tail = uav.components["tail"]
        fuselage = uav.components["fuselage_outer"]
        wings_ext, tail_ext = cut_fuselage_from_parts(wings, tail, fuselage)
        overlap = wings_ext.intersect(tail_ext)
        overlap_vol = _solid_volume(overlap)
        parts = {"fuselage": fuselage, "wings": wings_ext, "tail": tail_ext}
        if overlap_vol > 1e-6:
            parts["overlap"] = overlap
        for name, solid in parts.items():
            try:
                cq.exporters.export(solid, str(out_dir / f"{name}.stl"))
                written.append(name)
            except Exception:  # noqa: BLE001, PERF203
                pass
    return {"parts": written, "overlap_mm3": float(overlap_vol)}


def _mesh_status(run_dir: Path, start_id: str) -> dict:
    out: dict[str, list[str]] = {}
    for which in ("start", "final"):
        d = run_dir / "meshes" / start_id / which
        out[which] = sorted(p.stem for p in d.glob("*.stl")) if d.is_dir() else []
    return out


class ViewerHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: A002
        pass

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data, status: int = 200) -> None:
        self._send_bytes(json.dumps(data).encode(), "application/json", status)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._send_bytes((SCRIPT_DIR / "index.html").read_bytes(), _MIME[".html"])
            return
        if path == "/api/runs":
            self._send_json({"runs": _discover_runs()})
            return
        if path == "/api/allowlist":
            self._send_json(_ALLOWLIST)
            return
        if path.startswith("/api/viewer_data/"):
            run_dir = _run_dir(path[len("/api/viewer_data/"):])
            if run_dir is None:
                self._send_json({"error": "run not allowlisted or missing"}, 404)
                return
            self._send_bytes((run_dir / "viewer_data.json").read_bytes(), _MIME[".json"])
            return
        if path.startswith("/api/statistics/"):
            run_dir = _run_dir(path[len("/api/statistics/"):])
            if run_dir is None or not (run_dir / "statistics.json").exists():
                self._send_json({"error": "not found"}, 404)
                return
            self._send_bytes((run_dir / "statistics.json").read_bytes(), _MIME[".json"])
            return
        if path.startswith("/api/sim_eval/"):
            run_dir = _run_dir(path[len("/api/sim_eval/"):])
            sim_path = run_dir / "sim_eval.json" if run_dir else None
            if sim_path is None or not sim_path.exists():
                self._send_json({"error": "sim_eval not found"}, 404)
                return
            self._send_bytes(sim_path.read_bytes(), _MIME[".json"])
            return
        if path.startswith("/api/algorithm_md/"):
            run_dir = _run_dir(path[len("/api/algorithm_md/"):])
            md_path = (run_dir / "algorithm.md") if run_dir else None
            if md_path is None or not md_path.exists():
                self._send_json({"error": "algorithm.md not found"}, 404)
                return
            self._send_bytes(md_path.read_bytes(), "text/markdown; charset=utf-8")
            return
        if path.startswith("/api/mesh_status/"):
            rest = unquote(path[len("/api/mesh_status/"):]).rsplit("/", 1)
            run_dir = _run_dir(rest[0]) if len(rest) == 2 else None
            if run_dir is None:
                self._send_json({"error": "not found"}, 404)
                return
            self._send_json({"status": _mesh_status(run_dir, rest[1])})
            return
        if path.startswith("/mesh/"):
            rest = unquote(path[len("/mesh/"):]).split("/", 1)
            if len(rest) != 2:
                self.send_error(404)
                return
            run_dir = _run_dir(rest[0])
            if run_dir is None:
                self.send_error(404)
                return
            target = (run_dir / rest[1]).resolve()
            if not str(target).startswith(str(run_dir.resolve())) or not target.is_file():
                self.send_error(404)
                return
            self._send_bytes(target.read_bytes(), _MIME.get(target.suffix, "application/octet-stream"))
            return
        self.send_error(404)

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if not path.startswith("/api/render/"):
            self.send_error(404)
            return
        rest = unquote(path[len("/api/render/"):]).rsplit("/", 1)
        if len(rest) != 2:
            self._send_json({"ok": False, "error": "bad request"}, 400)
            return
        run_key, start_id = rest
        run_dir = _run_dir(run_key)
        if run_dir is None:
            self._send_json({"ok": False, "error": "run not allowlisted"}, 404)
            return

        vd = json.loads((run_dir / "viewer_data.json").read_text())
        start = next(
            (s for s in vd.get("starts", []) if str(s.get("start_id")) == start_id), None
        )
        if start is None:
            self._send_json({"ok": False, "error": f"start_id not found: {start_id}"}, 404)
            return

        what = (parse_qs(parsed.query).get("what", [""])[0]) or ""
        status = _mesh_status(run_dir, start_id)
        if not what.strip():
            what = "final" if status.get("start") else "start,final"
        wanted = {w.strip() for w in what.split(",") if w.strip()}
        adv_by_what = {"start": start.get("adv_start"), "final": start.get("adv_final")}

        t0 = time.perf_counter()
        try:
            for which in ("start", "final"):
                if which not in wanted:
                    continue
                adv = adv_by_what.get(which)
                if not adv:
                    continue
                _render_adv(adv, run_dir / "meshes" / start_id / which)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "error": f"render failed: {exc}"}, 500)
            return

        status = _mesh_status(run_dir, start_id)
        self._send_json({
            "ok": any(status.values()),
            "start_id": start_id,
            "status": status,
            "wall_s": time.perf_counter() - t0,
        })


def main() -> None:
    ap = argparse.ArgumentParser(description="Chapter 7 AeroForge Repair Viewer (public release)")
    ap.add_argument("--port", type=int, default=8092)
    args = ap.parse_args()

    runs = _discover_runs()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), ViewerHandler)
    print("Chapter 7 AeroForge Repair Viewer")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Allowlist entries served: {len(runs)}")
    print(f"  Open: http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
