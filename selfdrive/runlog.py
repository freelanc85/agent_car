"""Writing run outputs that the dashboard reads (plain JSON files).

runs/<run_id>/
    config.json        everything needed to reproduce / reload the run
    status.json        state (running/done/stopped), generation, replay list
    metrics.jsonl      one JSON line per generation
    best_genome.json   champion network (load with evaluate.py)
    replays/*.json     recorded drives: track geometry + car paths
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from .brain import Brain
from .sim import Simulator


def write_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    os.replace(tmp, path)  # atomic: the dashboard never reads half a file


def _num(x):
    x = float(x)
    return None if np.isnan(x) else round(x, 4)


def record_drive(sim: Simulator, brain: Brain, genomes: np.ndarray, role: str,
                 fitness_order: bool = True, every: int = 2) -> dict:
    """Drive `genomes` on sim's track and return a replay entry for the viewer.
    Car 0 of the entry is the best one; its sensor readings are included."""
    res = sim.run(brain, genomes, record=True)
    order = np.argsort(-res["fitness"]) if fitness_order else np.arange(len(genomes))
    cars = []
    for rank, i in enumerate(order):
        T = int(res["steps"][i]) + 1
        path = np.round(res["traj"][i, :T:every], 1).tolist()
        cars.append(dict(rank=rank, fitness=_num(res["fitness"][i]),
                         progress=_num(res["progress"][i]),
                         finished=bool(res["finished"][i]), crashed=bool(res["crashed"][i]),
                         finish_time=_num(res["finish_time"][i]), path=path))
    b = order[0]
    T = int(res["steps"][b]) + 1
    return dict(role=role, track=sim.track.to_json(), cars=cars,
                rays=np.round(res["rays"][b, :T:every], 1).tolist(),
                ray_angles=sim.ray_angles.round(4).tolist(),
                ray_length=sim.cfg.ray_length, dt=sim.cfg.dt * every)


class RunLog:
    def __init__(self, run_dir: Path, config: dict):
        self.dir = Path(run_dir)
        (self.dir / "replays").mkdir(parents=True, exist_ok=True)
        (self.dir / "metrics.jsonl").write_text("")
        write_json(self.dir / "config.json", config)
        self.status = dict(state="running", generation=0, total_generations=config["generations"],
                           started=time.time(), updated=time.time(), replays=[],
                           name=config["name"], batch=config.get("batch"))
        self._save_status()

    def _save_status(self):
        self.status["updated"] = time.time()
        write_json(self.dir / "status.json", self.status)

    def log(self, metrics: dict):
        with open(self.dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps({k: (_num(v) if isinstance(v, (float, np.floating)) else v)
                                for k, v in metrics.items()}) + "\n")
        self.status["generation"] = metrics["gen"]
        self.status["last"] = {k: metrics[k] for k in
                               ("best", "mean", "finish_rate", "test_progress", "test_finish_rate")
                               if k in metrics}
        self._save_status()

    def save_genome(self, genome: np.ndarray, brain: Brain, info: dict):
        write_json(self.dir / "best_genome.json",
                   dict(layers=brain.layers, genome=genome.tolist(), **info))

    def save_replay(self, key: str, label: str, drives: list[dict]):
        write_json(self.dir / "replays" / f"{key}.json", dict(label=label, drives=drives))
        self.status["replays"] = [r for r in self.status["replays"] if r["key"] != key]
        self.status["replays"].append(dict(key=key, label=label))
        self._save_status()

    def finish(self, state: str = "done"):
        self.status["state"] = state
        self._save_status()
