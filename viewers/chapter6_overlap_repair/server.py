"""Chapter 6 Overlap Repair Viewer Server (public release).

Serves a browser UI for inspecting the strict overlap-repair and alternating
overlap/assemblability trajectories reported in Chapter 6 of the thesis.
Supports algorithm info (markdown), summary statistics, overlap-aware frame
visualization, and on-demand CadQuery OBJ generation of the clevis assembly.

Only the runs listed in
``bachelor-thesis/release_manifests/ch6_allowlist.json`` are served.
Run data lives under ``data/<run_id>/``
next to this file.

Usage:
    python -m viewers.chapter6_overlap_repair.server
    python -m viewers.chapter6_overlap_repair.server --port 8091

Run from the ``bachelor-thesis/`` repository root so the Chapter 3 geometry code
and Chapter 6 benchmark metadata resolve correctly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CODE_ROOT = REPO_ROOT / "code"
CHAPTER3_ROOT = CODE_ROOT / "chapter3_clevis_setup"

DATA_DIR = SCRIPT_DIR / "data"
CACHE_DIR = SCRIPT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

ALLOWLIST_PATH = REPO_ROOT / "release_manifests" / "ch6_allowlist.json"
CH64_BENCHMARK_DIR = REPO_ROOT / "datasets" / "chapter6_overlap_clevis" / "benchmark_strict_repair_ch64_v1"
CH65_BENCHMARK_DIR = REPO_ROOT / "datasets" / "chapter6_overlap_clevis" / "benchmark_strict_blocked_v2"

for _path in (str(CODE_ROOT), str(CHAPTER3_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

PARAM_NAMES = [
    "wall_thickness", "outer_span", "leg_length", "depth",
    "main_hole_offset_from_open_end", "main_hole_radius",
    "main_pin_length", "main_pin_radius",
    "cross_hole_radius", "cross_hole_distance_from_free_end",
    "splint_radius", "splint_length", "overhang_span_y",
]

PARAM_RANGES = {
    "wall_thickness": (2.0, 12.0),
    "outer_span": (30.0, 100.0),
    "leg_length": (20.0, 80.0),
    "depth": (12.0, 60.0),
    "main_hole_offset_from_open_end": (2.0, 62.0),
    "main_hole_radius": (2.0, 10.0),
    "main_pin_length": (39.0, 182.0),
    "main_pin_radius": (1.5, 10.0),
    "cross_hole_radius": (1.0, 5.5),
    "cross_hole_distance_from_free_end": (1.5, 30.0),
    "splint_radius": (0.8, 4.0),
    "splint_length": (7.0, 42.0),
    "overhang_span_y": (24.0, 200.0),
}

_generation_locks: dict[str, threading.Lock] = {}
_generation_locks_lock = threading.Lock()
_prefetch_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="prefetch")
_viewer_data_cache: dict[str, tuple[float, list[dict]]] = {}


def _get_lock(key: str) -> threading.Lock:
    with _generation_locks_lock:
        if key not in _generation_locks:
            _generation_locks[key] = threading.Lock()
        return _generation_locks[key]


def _load_allowlist() -> dict:
    if not ALLOWLIST_PATH.exists():
        return {"entries": []}
    return json.loads(ALLOWLIST_PATH.read_text())


_ALLOWLIST = _load_allowlist()
_ALLOWLIST_ENTRIES = _ALLOWLIST.get("entries", [])
_ALLOWED_RUN_IDS = {e["run_id"] for e in _ALLOWLIST_ENTRIES}
_ENTRY_TO_RUN = {e["entry_id"]: e["run_id"] for e in _ALLOWLIST_ENTRIES}


def _resolve_run_id(key: str) -> str | None:
    if key in _ENTRY_TO_RUN:
        return _ENTRY_TO_RUN[key]
    if key in _ALLOWED_RUN_IDS:
        return key
    return None


def _run_dir(key: str) -> Path | None:
    run_id = _resolve_run_id(key)
    if run_id is None:
        return None
    candidate = DATA_DIR / run_id
    if not candidate.is_dir():
        return None
    return candidate


def _statistics_from_manifest(manifest: dict) -> dict:
    return dict(manifest.get("results_summary") or {})


def _statistics_md_from_manifest(manifest: dict) -> str:
    rs = manifest.get("results_summary") or {}
    title = manifest.get("title") or manifest.get("run_id", "Run")
    task = manifest.get("task", "")
    n_starts = manifest.get("n_starts", "—")
    lines = [f"# {title}", "", "## Summary statistics", ""]
    if task == "ch65_zigzag":
        succ = rs.get("pipeline_success", rs.get("oracle_confirmed", "—"))
        lines.extend([
            f"- Pipeline success: **{succ}/{n_starts}**",
            f"- False success: **{rs.get('false_success', 0)}**",
        ])
    else:
        succ = rs.get("strict_success", rs.get("oracle_confirmed", "—"))
        lines.append(f"- Strict success: **{succ}/{n_starts}**")
        mean_red = rs.get("mean_pct_overlap_reduction")
        if isinstance(mean_red, (int, float)):
            lines.append(f"- Mean overlap-volume reduction: **{mean_red:.1f}%**")
        mean_dist = rs.get("mean_distance")
        if isinstance(mean_dist, (int, float)):
            lines.append(f"- Mean normalized distance: **{mean_dist:.4f}**")
        lines.append(f"- False success: **{rs.get('false_success', 0)}**")
    lines.extend([
        "",
        "Per-start overlap metrics, pair residuals, and trajectories are available",
        "in the assembly list and frame info panel. Ch6 viewer exports do not ship",
        "separate `statistics.json` / `statistics.md` files; this summary is derived",
        "from `manifest.json`.",
    ])
    return "\n".join(lines)


def _discover_runs() -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for entry in _ALLOWLIST_ENTRIES:
        run_id = entry["run_id"]
        run_dir = DATA_DIR / run_id
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, KeyError):
            continue
        stats_path = run_dir / "statistics.json"
        if stats_path.exists():
            try:
                statistics = json.loads(stats_path.read_text())
            except json.JSONDecodeError:
                statistics = _statistics_from_manifest(manifest)
        else:
            statistics = _statistics_from_manifest(manifest)
        runs[entry["entry_id"]] = {
            "run_id": run_id,
            "manifest": manifest,
            "statistics": statistics,
            "thesis_anchor": entry.get("thesis_anchor"),
            "display_label": entry.get("display_label"),
            "optional": bool(entry.get("optional")),
            "has_viewer_data": (run_dir / "viewer_data.json").exists(),
            "has_algorithm_md": (run_dir / "algorithm.md").exists(),
            "has_statistics": (run_dir / "statistics.md").exists(),
        }
    return runs


def _load_viewer_data(run_dir: Path) -> list[dict] | None:
    key = str(run_dir)
    vd_path = run_dir / "viewer_data.json"
    if vd_path.exists():
        mtime = vd_path.stat().st_mtime
        cached = _viewer_data_cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1]
        data = json.loads(vd_path.read_text())
        _viewer_data_cache[key] = (mtime, data)
        return data
    return None


def _load_benchmark_metadata() -> dict:
    summary_path = CH64_BENCHMARK_DIR / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        return {
            "benchmark_id": summary.get("benchmark_id", CH64_BENCHMARK_DIR.name),
            "n_starts": summary.get("n_starts", 50),
            "ch64_benchmark": CH64_BENCHMARK_DIR.name,
            "ch65_benchmark": CH65_BENCHMARK_DIR.name,
            "parameter_order": PARAM_NAMES,
            "stds": {},
        }
    return {
        "benchmark_id": CH64_BENCHMARK_DIR.name,
        "n_starts": 50,
        "ch64_benchmark": CH64_BENCHMARK_DIR.name,
        "ch65_benchmark": CH65_BENCHMARK_DIR.name,
        "parameter_order": PARAM_NAMES,
        "stds": {},
    }


def _generate_obj(params: dict) -> dict[str, str]:
    param_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
    cache_path = CACHE_DIR / param_hash

    if cache_path.exists():
        result = {}
        for part in ["bracket", "main_pin", "splint"]:
            obj_file = cache_path / f"{part}.obj"
            if obj_file.exists():
                result[part] = obj_file.read_text()
        if len(result) == 3:
            return result

    try:
        import cadquery as cq
        from generate_modified_clevis_dummy import DummyParams, build_parts

        kwargs = {k: float(params[k]) for k in PARAM_NAMES}
        kwargs["exploded_gap"] = 30.0
        p = DummyParams(**kwargs)
        parts = build_parts(p)

        cache_path.mkdir(parents=True, exist_ok=True)
        result = {}
        for part_name, workplane in parts.items():
            obj_path = cache_path / f"{part_name}.obj"
            try:
                cq.exporters.export(workplane, str(obj_path))
            except Exception:
                import tempfile

                import trimesh
                with tempfile.NamedTemporaryFile(suffix=".stl") as tmp:
                    cq.exporters.export(workplane, tmp.name)
                    mesh = trimesh.load_mesh(tmp.name, force="mesh")
                    mesh.export(str(obj_path))
            result[part_name] = obj_path.read_text()
        return result
    except Exception as e:  # noqa: BLE001 - surfaced to the UI as a soft warning
        return {"error": str(e)}


class ViewerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)  # noqa: F841 - reserved for future filters

        if path == "/" or path == "/index.html":
            self._serve_file(SCRIPT_DIR / "index.html", "text/html")
        elif path == "/api/runs":
            self._json_response(_discover_runs())
        elif path == "/api/allowlist":
            self._json_response(_ALLOWLIST)
        elif path == "/api/benchmark_metadata":
            self._json_response(_load_benchmark_metadata())
        elif path.startswith("/api/viewer_data/"):
            run_id = path.split("/api/viewer_data/", 1)[1]
            run_dir = _run_dir(run_id)
            data = _load_viewer_data(run_dir) if run_dir else None
            if data is None:
                self._json_response({"error": f"No viewer_data.json for allowlisted run '{run_id}'"}, 404)
            else:
                self._json_response(data)
        elif path.startswith("/api/algorithm_md/"):
            run_id = path.split("/api/algorithm_md/", 1)[1]
            run_dir = _run_dir(run_id)
            md_path = (run_dir / "algorithm.md") if run_dir else None
            if md_path and md_path.exists():
                self._text_response(md_path.read_text(), "text/markdown")
            else:
                self._json_response({"error": "No algorithm.md found"}, 404)
        elif path.startswith("/api/statistics_md/"):
            run_id = path.split("/api/statistics_md/", 1)[1]
            run_dir = _run_dir(run_id)
            if not run_dir:
                self._json_response({"error": "Run not found"}, 404)
                return
            md_path = run_dir / "statistics.md"
            if md_path.exists():
                self._text_response(md_path.read_text(), "text/markdown")
            else:
                manifest_path = run_dir / "manifest.json"
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text())
                    self._text_response(_statistics_md_from_manifest(manifest), "text/markdown")
                else:
                    self._json_response({"error": "No statistics available"}, 404)
        elif path.startswith("/api/statistics/"):
            run_id = path.split("/api/statistics/", 1)[1]
            run_dir = _run_dir(run_id)
            if not run_dir:
                self._json_response({"error": "Run not found"}, 404)
                return
            stats_path = run_dir / "statistics.json"
            if stats_path.exists():
                self._json_response(json.loads(stats_path.read_text()))
            else:
                manifest_path = run_dir / "manifest.json"
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text())
                    self._json_response(_statistics_from_manifest(manifest))
                else:
                    self._json_response({"error": "No statistics available"}, 404)
        elif path.startswith("/api/frame/"):
            parts = path.split("/api/frame/", 1)[1].split("/")
            if len(parts) >= 3:
                run_id = "/".join(parts[:-2])
                try:
                    assembly_idx = int(parts[-2])
                    frame_idx = int(parts[-1])
                except ValueError:
                    self._json_response({"error": "Assembly and frame indices must be integers"}, 400)
                    return
                if assembly_idx < 0 or frame_idx < 0:
                    self._json_response({"error": "Assembly and frame indices must be non-negative"}, 400)
                    return
                run_dir = _run_dir(run_id)
                data = _load_viewer_data(run_dir) if run_dir else None
                if data is None:
                    self._json_response({"error": f"No viewer_data.json for allowlisted run '{run_id}'"}, 404)
                    return
                if assembly_idx < len(data):
                    assembly = data[assembly_idx]
                    if frame_idx < len(assembly.get("frames", [])):
                        frame = assembly["frames"][frame_idx]
                        frame_params = frame["params"]
                        obj_data = _generate_obj(frame_params)
                        self._json_response({"params": frame_params, "obj": obj_data, "frame": frame})
                    else:
                        self._json_response({"error": "Frame index out of range"}, 404)
                else:
                    self._json_response({"error": "Assembly index out of range"}, 404)
            else:
                self._json_response({"error": "Invalid frame path"}, 400)
        elif path == "/api/manifest":
            self._json_response({
                "param_names": PARAM_NAMES,
                "param_ranges": PARAM_RANGES,
                "n_frames": 50,
            })
        else:
            self._json_response({"error": "Not found"}, 404)

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _text_response(self, text: str, content_type: str, status=200):
        body = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, filepath: Path, content_type: str):
        if filepath.exists():
            body = filepath.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", len(body))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json_response({"error": "File not found"}, 404)


def main():
    parser = argparse.ArgumentParser(description="Chapter 6 Overlap Repair Viewer (public release)")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), ViewerHandler)
    print(f"Chapter 6 Viewer: http://127.0.0.1:{args.port}")
    print(f"Run data directory: {DATA_DIR}")
    runs = _discover_runs()
    print(f"Discovered {len(runs)} allowlisted optimizer entry(ies)")
    server.serve_forever()


if __name__ == "__main__":
    main()
