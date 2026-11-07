#!/usr/bin/env python3
"""
Drop a trained car onto any track(s) and see how it does.

    python evaluate.py runs/smoke --tracks random:5000..5009
    python evaluate.py runs/smoke --tracks preset:hairpins,tracks/my_track.json --save

--save adds a replay to the run so you can watch it in the dashboard.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from selfdrive.brain import Brain
from selfdrive.render import save_track_images
from selfdrive.runlog import drive_paths, record_drive, write_json
from selfdrive.sim import SimConfig, Simulator
from selfdrive.track import load_track, parse_track_list


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="run directory (contains best_genome.json)")
    p.add_argument("--tracks", default="random:5000..5019")
    p.add_argument("--laps", type=int, default=None)
    p.add_argument("--save", action="store_true", help="save a replay into the run for the dashboard")
    p.add_argument("--json", default=None, help="also write the results table to this JSON file")
    args = p.parse_args()

    run = Path(args.run)
    g = json.loads((run / "best_genome.json").read_text())
    brain = Brain(g["layers"])
    genome = np.array(g["genome"])[None]
    sim_cfg = {**g["sim"], "sensor_noise": 0.0}
    if args.laps:
        sim_cfg["laps"] = args.laps
    cfg = SimConfig(**sim_cfg)

    print(f"{'track':<28}{'progress':>9}{'finished':>10}{'time (s)':>10}")
    rows, drives, tracks = [], [], []
    for spec in parse_track_list(args.tracks):
        sim = Simulator(load_track(spec), cfg)
        tracks.append(sim.track)
        r = sim.run(brain, genome)
        rows.append((spec, r["progress"][0], r["finished"][0], r["finish_time"][0]))
        t = "-" if np.isnan(r["finish_time"][0]) else f"{r['finish_time'][0]:.1f}"
        print(f"{spec:<28}{100*r['progress'][0]:8.1f}%{str(bool(r['finished'][0])):>10}{t:>10}")
        drives.append(record_drive(sim, brain, genome, "eval"))
    prog = np.mean([r[1] for r in rows])
    fin = np.mean([r[2] for r in rows])
    print(f"\n{len(rows)} tracks: mean progress {100*prog:.1f}%, finished {100*fin:.0f}%")

    key = time.strftime("eval_%H%M%S")
    files = save_track_images(run / "images" / key, [(t, "eval") for t in tracks], drive_paths(drives, tracks))
    print(f"track images (with the car's path): {files[-1]}")

    if args.json:
        Path(args.json).write_text(json.dumps(dict(
            key=key, mean_progress=float(prog), finish_rate=float(fin),
            sheet=files[-1].relative_to(run).as_posix(),
            rows=[dict(track=t, progress=float(p_), finished=bool(f), time=None if np.isnan(tm) else float(tm))
                  for t, p_, f, tm in rows])))

    if args.save:
        status_p = run / "status.json"
        status = json.loads(status_p.read_text())
        status.setdefault("images", {})[key] = dict(
            title=f"eval: {args.tracks}", files=[f.relative_to(run).as_posix() for f in files])
        write_json(run / "replays" / f"{key}.json", dict(label=f"eval: {args.tracks}", drives=drives))
        status["replays"].append(dict(key=key, label=f"eval: {args.tracks}"))
        write_json(status_p, status)
        print(f"saved replay '{key}' — open the dashboard to watch it")


if __name__ == "__main__":
    main()
