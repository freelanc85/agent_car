#!/usr/bin/env python3
"""
Tiny local web server for the dashboard (standard library only).

    python serve.py            ->  http://localhost:8000
    python serve.py --port 9000

It serves the dashboard/ page and the runs/ folder, plus /api/runs which lists
every run (any folder under runs/ that contains a status.json). Training can
run at the same time in another terminal; the page polls for new data.
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Handler(SimpleHTTPRequestHandler):
    runs_dir: Path = HERE / "runs"

    def log_message(self, *a):  # keep the terminal quiet
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/dashboard/")
            self.end_headers()
            return
        if self.path.startswith("/api/runs"):
            runs, batches = [], []
            for st in sorted(self.runs_dir.rglob("status.json")):
                try:
                    s = json.loads(st.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                s["id"] = st.parent.relative_to(self.runs_dir).as_posix()
                runs.append(s)
            for b in sorted(self.runs_dir.rglob("batch.json")):
                try:
                    bj = json.loads(b.read_text())
                    sp = b.parent / "summary.json"
                    bj["summary"] = json.loads(sp.read_text()) if sp.exists() else None
                    batches.append(bj)
                except (OSError, json.JSONDecodeError):
                    pass
            body = json.dumps(dict(runs=runs, batches=batches)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--runs-dir", default=str(HERE / "runs"))
    a = p.parse_args()
    Handler.runs_dir = Path(a.runs_dir).resolve()
    Handler.runs_dir.mkdir(exist_ok=True)
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), partial(Handler, directory=str(HERE)))
    print(f"dashboard: http://localhost:{a.port}/dashboard/   (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
