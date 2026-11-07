#!/usr/bin/env python3
"""
Train self-driving cars with neuroevolution — headless.

    python train.py                                   # train on preset:circuit
    python train.py --train preset:circuit,preset:oval --name two_tracks
    python train.py --train preset:circuit --random-per-gen 2   # + 2 fresh random tracks / gen

Every generation the current champion is also driven on held-out *test*
tracks it never trained on; that's the generalisation score.
Watch progress live with:  python serve.py   ->  http://localhost:8000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from selfdrive.brain import Brain
from selfdrive.evolve import GeneticAlgorithm
from selfdrive.runlog import RunLog, drive_paths, record_drive
from selfdrive.sim import SimConfig, Simulator
from selfdrive.track import load_track, parse_track_list

DEFAULT_TEST = "random:1001..1006,preset:snake,preset:hairpins,preset:circuit:rev"


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("run")
    g.add_argument("--name", default=None, help="run name (default: timestamp)")
    g.add_argument("--batch", default=None, help="group runs under runs/<batch>/")
    g.add_argument("--runs-dir", default="runs")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--generations", type=int, default=60)
    g.add_argument("--quiet", action="store_true")
    g = p.add_argument_group("tracks")
    g.add_argument("--train", default="preset:circuit", help="comma list of training tracks")
    g.add_argument("--test", default=DEFAULT_TEST, help="comma list of held-out test tracks")
    g.add_argument("--random-per-gen", type=int, default=0,
                   help="also train on N brand-new random tracks every generation")
    g = p.add_argument_group("evolution")
    g.add_argument("--pop", type=int, default=100)
    g.add_argument("--elite", type=int, default=4)
    g.add_argument("--tournament", type=int, default=5)
    g.add_argument("--crossover-rate", type=float, default=0.7)
    g.add_argument("--mutation-rate", type=float, default=0.1)
    g.add_argument("--mutation-sigma", type=float, default=0.3)
    g = p.add_argument_group("car / network")
    g.add_argument("--hidden", default="12", help="hidden layer sizes, e.g. 16,8")
    g.add_argument("--rays", type=int, default=7)
    g.add_argument("--fov", type=float, default=180.0)
    g.add_argument("--laps", type=int, default=2)
    g.add_argument("--max-steps", type=int, default=1500)
    g.add_argument("--sensor-noise", type=float, default=0.0)
    g = p.add_argument_group("logging")
    g.add_argument("--replay-every", type=int, default=10, help="record a replay every N gens")
    g.add_argument("--replay-top", type=int, default=20, help="cars shown in training replays")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    name = args.name or time.strftime("run_%Y%m%d_%H%M%S")
    run_id = f"{args.batch}/{name}" if args.batch else name
    rng = np.random.default_rng(args.seed)

    cfg = SimConfig(n_rays=args.rays, ray_fov_deg=args.fov, laps=args.laps,
                    max_steps=args.max_steps, sensor_noise=args.sensor_noise)
    test_cfg = SimConfig(**{**cfg.to_dict(), "sensor_noise": 0.0})
    brain = Brain([cfg.n_inputs, *[int(h) for h in args.hidden.split(",") if h], 2])
    ga = GeneticAlgorithm(args.pop, args.elite, args.tournament, args.crossover_rate,
                          args.mutation_rate, args.mutation_sigma, rng)

    train_specs = parse_track_list(args.train)
    test_specs = [s for s in parse_track_list(args.test) if s not in train_specs]
    train_sims = [Simulator(load_track(s), cfg) for s in train_specs]
    test_sims = [Simulator(load_track(s), test_cfg) for s in test_specs]

    config = dict(vars(args), name=name, run_id=run_id, train_tracks=train_specs,
                  test_tracks=test_specs, sim=cfg.to_dict(), layers=brain.layers,
                  genome_size=brain.size, command=" ".join(sys.argv))
    log = RunLog(Path(args.runs_dir) / run_id, config)
    track_list = [(s.track, "train") for s in train_sims] + [(s.track, "test") for s in test_sims]
    log.save_images("tracks", track_list, title="training and test tracks")
    say = (lambda *a: None) if args.quiet else print
    say(f"[{run_id}] brain {brain.layers} ({brain.size} weights), pop {args.pop}")
    say(f"  train: {', '.join(train_specs)}" + (f" + {args.random_per_gen} random/gen" if args.random_per_gen else ""))
    say(f"  test : {', '.join(test_specs)}")
    say(f"{'gen':>4} {'best':>6} {'mean':>6} {'fin%':>5} {'test_prog':>9} {'test_fin%':>9} {'sec':>5}")

    pop = brain.init_population(args.pop, rng)
    champion, champion_fit = None, -np.inf
    state = "done"
    try:
        for gen in range(1, args.generations + 1):
            t0 = time.time()
            sims = list(train_sims)
            if args.random_per_gen:
                seeds = 100_000 + gen * args.random_per_gen + np.arange(args.random_per_gen)
                sims += [Simulator(load_track(f"random:{s}"), cfg) for s in seeds]

            # ---- evaluate population on every training track -------------
            results = [s.run(brain, pop, rng) for s in sims]
            fitness = np.mean([r["fitness"] for r in results], 0)
            finished = np.mean([r["finished"] for r in results], 0)
            progress = np.mean([r["progress"] for r in results], 0)
            best_i = int(np.argmax(fitness))
            best_g = pop[best_i].copy()
            ftimes = [r["finish_time"][best_i] for r in results]

            # ---- generalisation: champion of this gen on unseen tracks ----
            test = [s.run(brain, best_g[None]) for s in test_sims]
            test_prog = np.array([r["progress"][0] for r in test])
            test_fin = np.array([r["finished"][0] for r in test])

            m = dict(
                gen=gen,
                best=float(fitness[best_i]), mean=float(fitness.mean()),
                median=float(np.median(fitness)), p90=float(np.percentile(fitness, 90)),
                finish_rate=float(finished.mean()),          # share of cars finishing
                best_progress=float(progress[best_i]), mean_progress=float(progress.mean()),
                best_finish_time=float(np.nanmean(ftimes)) if np.isfinite(ftimes).any() else None,
                test_progress=float(test_prog.mean()) if len(test) else None,
                test_finish_rate=float(test_fin.mean()) if len(test) else None,
                test_by_track={s.track.name: round(float(p), 3) for s, p in zip(test_sims, test_prog)},
                seconds=time.time() - t0,
            )
            log.log(m)
            say(f"{gen:>4} {m['best']:6.3f} {m['mean']:6.3f} {100*m['finish_rate']:5.1f} "
                f"{100*(m['test_progress'] or 0):8.1f}% {100*(m['test_finish_rate'] or 0):8.1f}% "
                f"{m['seconds']:5.1f}")

            if fitness[best_i] >= champion_fit:
                champion, champion_fit = best_g, float(fitness[best_i])
                log.save_genome(champion, brain, dict(gen=gen, fitness=champion_fit,
                                                      sim=cfg.to_dict(), run_id=run_id))

            if gen == 1 or gen % args.replay_every == 0 or gen == args.generations:
                top = pop[np.argsort(-fitness)[: args.replay_top]]
                drives = [record_drive(train_sims[0] if train_sims else sims[0], brain, top, "train")]
                drives += [record_drive(s, brain, best_g[None], "test") for s in test_sims]
                log.save_replay(f"gen_{gen:04d}", f"generation {gen}", drives)
                shown = [train_sims[0] if train_sims else sims[0], *test_sims]
                log.save_images("champion", [(s.track, r) for s, r in zip(shown, ["train"] + ["test"] * len(test_sims))],
                                drive_paths(drives, [s.track for s in shown]),
                                title=f"best car's path, generation {gen}")

            pop = ga.next_generation(pop, fitness)
    except KeyboardInterrupt:
        state = "stopped"
        say("\ninterrupted — saving")
    log.finish(state)
    say(f"[{run_id}] {state}. champion fitness {champion_fit:.3f} -> {log.dir}/best_genome.json")
    return log.dir


if __name__ == "__main__":
    main()
