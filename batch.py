#!/usr/bin/env python3
"""
Run a batch of training runs (a grid of settings x several seeds), in parallel,
headless. Each run shows up in the dashboard under the batch name, and a
summary table is printed + saved at the end.

    # Does training on more tracks generalise better?
    python batch.py --name tracks_vs_generalisation --seeds 3 --generations 40 \
        --grid "train=preset:circuit | preset:circuit,preset:oval,random:1..3"

    # Network size sweep
    python batch.py --name hidden_sweep --grid "hidden=4 | 12 | 16,8"

    # Several axes at once (full grid): 2 x 2 configs x seeds
    python batch.py --name grid --grid "hidden=6 | 12" --grid "random-per-gen=0 | 2"

Any train.py option can be a grid axis; pass fixed options after `--`:
    python batch.py --name x --grid "pop=50 | 150" -- --laps 1
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", s).strip("_")[:40]


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--grid", action="append", default=[], help='"option=value1 | value2 | ..."')
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--generations", type=int, default=40)
    p.add_argument("--parallel", type=int, default=os.cpu_count() or 2)
    p.add_argument("--runs-dir", default="runs")
    args = p.parse_args(argv)

    axes = []
    for g in args.grid:
        key, vals = g.split("=", 1)
        axes.append((key.strip().lstrip("-"), [v.strip() for v in vals.split("|")]))
    combos = list(itertools.product(*[[(k, v) for v in vs] for k, vs in axes])) or [()]

    jobs = []
    for combo in combos:
        label = "__".join(f"{k}-{slug(v)}" for k, v in combo) or "default"
        for seed in range(args.seeds):
            cmd = [sys.executable, str(HERE / "train.py"), "--batch", args.name,
                   "--name", f"{label}__s{seed}", "--seed", str(seed),
                   "--generations", str(args.generations), "--runs-dir", args.runs_dir, "--quiet"]
            for k, v in combo:
                cmd += [f"--{k}", v]
            jobs.append((label, seed, cmd + extra))

    batch_dir = Path(args.runs_dir) / args.name
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "batch.json").write_text(json.dumps(dict(
        name=args.name, axes=axes, seeds=args.seeds, generations=args.generations,
        extra=extra, started=time.time(), runs=[f"{l}__s{s}" for l, s, _ in jobs]), indent=1))
    print(f"batch '{args.name}': {len(combos)} configs x {args.seeds} seeds = {len(jobs)} runs, "
          f"{args.parallel} in parallel. Live view: python serve.py")

    done = 0
    t0 = time.time()

    def work(job):
        nonlocal done
        label, seed, cmd = job
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                           env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        done += 1
        print(f"  [{done}/{len(jobs)}] {label} seed {seed} "
              f"{'ok' if r.returncode == 0 else 'FAILED'}  ({time.time()-t0:.0f}s)")
        if r.returncode:
            print(r.stderr[-2000:])

    with ThreadPoolExecutor(args.parallel) as ex:
        list(ex.map(work, jobs))

    # ---- summary: final-generation metrics averaged over seeds --------------
    def last_metrics(run):
        f = batch_dir / run / "metrics.jsonl"
        lines = f.read_text().strip().splitlines() if f.exists() else []
        return json.loads(lines[-1]) if lines else None

    summary = []
    for combo in combos:
        label = "__".join(f"{k}-{slug(v)}" for k, v in combo) or "default"
        ms = [m for s in range(args.seeds) if (m := last_metrics(f"{label}__s{s}"))]
        if not ms:
            continue
        avg = lambda k: sum((m.get(k) or 0) for m in ms) / len(ms)  # noqa: E731
        summary.append(dict(config=dict(combo), label=label, seeds=len(ms),
                            train_best=avg("best"), train_finish_rate=avg("finish_rate"),
                            test_progress=avg("test_progress"),
                            test_finish_rate=avg("test_finish_rate")))
    (batch_dir / "summary.json").write_text(json.dumps(summary, indent=1))

    w = max([len(s["label"]) for s in summary] + [6])
    print(f"\n{'config':<{w}}  {'train best':>10} {'train fin%':>10} {'TEST prog':>10} {'TEST fin%':>10}")
    for s in summary:
        print(f"{s['label']:<{w}}  {s['train_best']:10.3f} {100*s['train_finish_rate']:9.1f}% "
              f"{100*s['test_progress']:9.1f}% {100*s['test_finish_rate']:9.1f}%")
    print(f"\n(means over seeds, final generation) -> {batch_dir}/summary.json")


if __name__ == "__main__":
    main()
