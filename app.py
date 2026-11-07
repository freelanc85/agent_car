#!/usr/bin/env python3
"""
Self-Driving Cars Lab — web app. Everything happens in the browser:

    python app.py            # starts the server and opens http://localhost:8000

From the page you can start/stop training runs and batches, pick and preview
tracks, watch live metrics and replays, test a trained car on other tracks,
and draw new tracks. (The command-line scripts still work too.)

Only the standard library + numpy are needed.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
TRACKS_DIR = HERE / "tracks"

try:
    import numpy  # noqa: F401
except ImportError:
    sys.exit("numpy is not installed for this Python.\n"
             f"Install it with:  {Path(sys.executable).name} -m pip install numpy")

sys.path.insert(0, str(HERE))
import train as train_mod  # noqa: E402
from selfdrive.render import png_bytes, render_track  # noqa: E402
from selfdrive.track import PRESETS, load_track, parse_track_list  # noqa: E402

# processes started from the web UI: id -> dict(proc, log, kind)
PROCS: dict[str, dict] = {}
LOCK = threading.Lock()
_png_cache: dict[tuple, bytes] = {}

# train.py options handled by the app itself (not shown as plain form fields)
HIDDEN_OPTS = {"name", "batch", "runs_dir", "quiet", "train", "test", "help"}
# options a batch sets per run, so they must not be passed as fixed options
BATCH_OWNED = {"seed", "generations", "name", "batch", "runs_dir", "quiet"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class ApiError(Exception):
    pass


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", s.strip()).strip("_")[:60]


def safe_run_dir(run_id: str) -> Path:
    d = (RUNS / run_id).resolve()
    if RUNS.resolve() not in d.parents:
        raise ApiError("bad run id")
    return d


def unique_name(parent: Path, name: str) -> str:
    name = slug(name) or time.strftime("run_%Y%m%d_%H%M%S")
    out, k = name, 2
    while (parent / out).exists() or out in PROCS:
        out, k = f"{name}_{k}", k + 1
    return out


def parser_options():
    """Describe train.py's options (from its argparse parser) for the form."""
    p = train_mod.build_parser()
    groups = []
    for g in p._action_groups:
        fields = []
        for a in g._group_actions:
            if a.dest in HIDDEN_OPTS or not a.option_strings:
                continue
            typ = ("bool" if a.nargs == 0 else "int" if a.type is int
                   else "float" if a.type is float else "str")
            fields.append(dict(dest=a.dest, flag=a.option_strings[-1], default=a.default,
                               type=typ, help=a.help or ""))
        if fields:
            groups.append(dict(title=g.title, fields=fields))
    return groups


def flag_for(dest: str) -> str:
    for a in train_mod.build_parser()._actions:
        if a.dest == dest:
            return a.option_strings[-1]
    raise ApiError(f"unknown option {dest}")


def args_from(settings: dict, skip=()) -> list[str]:
    out = []
    for dest, v in settings.items():
        if dest in HIDDEN_OPTS or dest in skip or v is None or v == "":
            continue
        flag = flag_for(dest)
        if isinstance(v, bool):
            if v:
                out.append(flag)
        else:
            out += [flag, str(v)]
    return out


def validate_train_args(argv: list[str]):
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            train_mod.build_parser().parse_args(argv)
    except SystemExit:
        raise ApiError(err.getvalue().strip().splitlines()[-1] if err.getvalue() else "invalid settings")


def validate_tracks(specs: list[str], what: str):
    if not specs:
        raise ApiError(f"pick at least one {what} track")
    for s in specs:
        for one in parse_track_list(s)[:3]:
            try:
                load_track(one)
            except Exception as e:  # noqa: BLE001
                raise ApiError(f"{what} track '{one}': {e}")


def launch(key: str, cmd: list[str], log_path: Path, kind: str):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w")
    f.write("$ " + " ".join(cmd[1:]) + "\n\n")
    f.flush()
    proc = subprocess.Popen(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONUNBUFFERED": "1",
                                 "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    with LOCK:
        PROCS[key] = dict(proc=proc, log=log_path, kind=kind, started=time.time(), file=f)


def tail(path: Path, n: int = 4000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n))
            return f.read().decode("utf8", "replace")
    except OSError:
        return ""


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def api_runs(_q):
    runs, batches, seen = [], [], set()
    for st in sorted(RUNS.rglob("status.json")):
        try:
            s = json.loads(st.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rid = st.parent.relative_to(RUNS).as_posix()
        s["id"] = rid
        seen.add(rid)
        with LOCK:
            pr = PROCS.get(rid)
        if pr and pr["proc"].poll() is not None and s.get("state") == "running":
            # process ended without marking itself done -> it crashed
            s["state"] = "failed"
            s["error"] = tail(pr["log"], 1500)
            (st.parent / "status.json").write_text(json.dumps({k: v for k, v in s.items() if k != "id"}))
        s["log"] = (st.parent / "log.txt").exists()
        s["controllable"] = bool(pr)
        runs.append(s)
    # runs started from the UI that have not written status.json yet (or crashed early)
    with LOCK:
        items = list(PROCS.items())
    for key, pr in items:
        if pr["kind"] != "run" or key in seen:
            continue
        code = pr["proc"].poll()
        runs.append(dict(id=key, name=key.split("/")[-1], batch=None, generation=0, total_generations=0,
                         state="starting" if code is None else "failed", replays=[], started=pr["started"],
                         updated=time.time(), log=True, error=None if code is None else tail(pr["log"], 1500)))
    for b in sorted(RUNS.rglob("batch.json")):
        try:
            bj = json.loads(b.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        sp = b.parent / "summary.json"
        bj["summary"] = json.loads(sp.read_text()) if sp.exists() else None
        with LOCK:
            pr = PROCS.get("batch:" + bj["name"])
        bj["running"] = bool(pr and pr["proc"].poll() is None)
        bj["stopping"] = (b.parent / "STOP").exists()
        batches.append(bj)
    return dict(runs=runs, batches=batches)


def api_options(_q):
    files = sorted(f"tracks/{p.name}" for p in TRACKS_DIR.glob("*.json"))
    return dict(groups=parser_options(), presets=list(PRESETS), files=files,
                default_train=["preset:circuit"],
                default_test=[s.strip() for s in train_mod.DEFAULT_TEST.split(",")])


def api_start(body):
    settings = body.get("settings", {})
    train = [s for s in body.get("train", []) if s]
    test = [s for s in body.get("test", []) if s]
    validate_tracks(train, "training")
    if test:
        validate_tracks(test, "test")
    base = args_from(settings) + ["--train", ",".join(train), "--test", ",".join(test)]
    validate_train_args(base)
    batch = body.get("batch")
    if not batch:
        name = unique_name(RUNS, body.get("name") or time.strftime("run_%H%M%S"))
        cmd = [sys.executable, str(HERE / "train.py"), "--name", name] + base
        launch(name, cmd, RUNS / name / "log.txt", "run")
        return dict(ok=True, id=name)

    grid = [(slug(k).replace("_", "-"), v) for k, v in batch.get("grid", []) if k and v.strip()]
    if not grid:
        raise ApiError("a batch needs at least one setting to vary (with values separated by | )")
    for k, v in grid:
        for val in v.split("|"):
            validate_train_args(base + [f"--{k}", val.strip()])
    owned = BATCH_OWNED | {k.replace("-", "_") for k, _ in grid}
    fixed = args_from(settings, skip=owned)
    fixed += ["--train", ",".join(train)] if "train" not in owned else []
    fixed += ["--test", ",".join(test)] if test else []
    name = unique_name(RUNS, body.get("name") or time.strftime("batch_%H%M%S"))
    cmd = [sys.executable, str(HERE / "batch.py"), "--name", name,
           "--seeds", str(int(batch.get("seeds", 3))), "--parallel", str(int(batch.get("parallel", 2))),
           "--generations", str(settings.get("generations", 60))]
    for k, v in grid:
        cmd += ["--grid", f"{k}={v}"]
    cmd += ["--"] + fixed
    launch("batch:" + name, cmd, RUNS / name / "batch_log.txt", "batch")
    return dict(ok=True, batch=name)


def api_stop(body):
    if body.get("batch"):
        d = safe_run_dir(body["batch"])
        d.mkdir(parents=True, exist_ok=True)
        (d / "STOP").write_text("stop")
        for st in d.rglob("status.json"):
            (st.parent / "STOP").write_text("stop")
        return dict(ok=True)
    d = safe_run_dir(body["id"])
    (d / "STOP").write_text("stop")
    return dict(ok=True)


def api_delete(body):
    rid = body["id"]
    d = safe_run_dir(rid)
    with LOCK:
        pr = PROCS.get(rid) or PROCS.get("batch:" + rid)
    if pr and pr["proc"].poll() is None:
        raise ApiError("still running — stop it first")
    if d.exists():
        shutil.rmtree(d)
    with LOCK:
        PROCS.pop(rid, None)
        PROCS.pop("batch:" + rid, None)
    return dict(ok=True)


def api_evaluate(body):
    d = safe_run_dir(body["id"])
    if not (d / "best_genome.json").exists():
        raise ApiError("this run has no trained car yet (wait for generation 1)")
    tracks = [s for s in body.get("tracks", []) if s]
    validate_tracks(tracks, "test")
    out = d / "eval_result.json"
    out.unlink(missing_ok=True)
    cmd = [sys.executable, str(HERE / "evaluate.py"), str(d), "--tracks", ",".join(tracks),
           "--save", "--json", str(out)]
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=900)
    if r.returncode or not out.exists():
        raise ApiError("evaluation failed:\n" + (r.stderr or r.stdout)[-1500:])
    return json.loads(out.read_text())


def api_save_track(body):
    name = slug(body.get("name", "")) or "my_track"
    pts = body.get("points") or []
    if len(pts) < 3:
        raise ApiError("a track needs at least 3 points")
    TRACKS_DIR.mkdir(exist_ok=True)
    path = TRACKS_DIR / f"{name}.json"
    path.write_text(json.dumps(dict(name=name, width=float(body.get("width", 50)), smooth=True,
                                    points=[[round(x), round(y)] for x, y in pts]), indent=1))
    from selfdrive.track import file_track
    problems = file_track(str(path)).problems()
    return dict(ok=True, spec=f"tracks/{name}.json", problems=problems)


def track_png(q):
    spec = q.get("spec", ["preset:circuit"])[0]
    size = max(80, min(int(q.get("size", ["200"])[0]), 1200))
    key = (spec, size, (TRACKS_DIR / spec[7:]).stat().st_mtime if spec.startswith("tracks/")
           and (TRACKS_DIR / spec[7:]).exists() else 0)
    if key not in _png_cache:
        first = parse_track_list(spec)[0]
        tr = load_track(first)
        _png_cache[key] = png_bytes(render_track(tr, size=size, pad=6))
    return _png_cache[key]


GET_ROUTES = {"/api/runs": api_runs, "/api/options": api_options}
POST_ROUTES = {"/api/start": api_start, "/api/stop": api_stop, "/api/delete": api_delete,
               "/api/evaluate": api_evaluate, "/api/tracks": api_save_track}


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode())

    def _call(self, fn, arg):
        try:
            self._json(200, fn(arg))
        except ApiError as e:
            self._json(400, dict(error=str(e)))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._json(500, dict(error=f"{type(e).__name__}: {e}"))

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/dashboard/")
            self.end_headers()
            return
        if u.path in GET_ROUTES:
            return self._call(GET_ROUTES[u.path], parse_qs(u.query))
        if u.path == "/api/track.png":
            try:
                return self._send(200, track_png(parse_qs(u.query)), "image/png")
            except Exception as e:  # noqa: BLE001
                return self._json(400, dict(error=str(e)))
        return super().do_GET()

    def do_POST(self):
        u = urlparse(self.path)
        if u.path not in POST_ROUTES:
            return self._json(404, dict(error="not found"))
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json(400, dict(error="bad JSON"))
        return self._call(POST_ROUTES[u.path], body)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow other machines")
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args()
    os.chdir(HERE)  # track files like tracks/x.json are relative to the project
    RUNS.mkdir(exist_ok=True)
    TRACKS_DIR.mkdir(exist_ok=True)

    srv = None
    for port in range(a.port, a.port + 20):  # first free port
        try:
            srv = ThreadingHTTPServer((a.host, port), lambda *x: Handler(*x, directory=str(HERE)))
            break
        except OSError:
            continue
    if srv is None:
        sys.exit(f"no free port in {a.port}..{a.port + 19}")
    url = f"http://localhost:{srv.server_address[1]}/dashboard/"
    print(f"Self-Driving Cars Lab running at  {url}\n(keep this window open; Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down — stopping runs started from the page")
        with LOCK:
            for pr in PROCS.values():
                if pr["proc"].poll() is None:
                    pr["proc"].terminate()


if __name__ == "__main__":
    main()
