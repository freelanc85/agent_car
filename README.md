# Self-Driving Cars Lab

Cars learn to drive around a track with **neuroevolution** (a genetic algorithm
evolving small neural networks), like the
[deepCars](https://arztsamuel.github.io/en/projects/unity/deepCars/deepCars.html) Unity project,
but in ~600 lines of plain Python + numpy and a browser dashboard.

* Training is **headless** and fast (≈1 s per generation of 100 cars).
* Every generation is logged; a **live web dashboard** shows the curves, batches of
  runs, and replays of the cars driving (including what their sensors see).
* Every generation the best car is also tested on **tracks it has never seen**,
  so you can watch whether the learning transfers.

```
pip install numpy          # the only dependency
python train.py            # train on preset:circuit for 60 generations
python serve.py            # in a 2nd terminal -> open http://localhost:8000
```

## How it works

```
                 ┌────────── one car ──────────┐
 7 distance rays │ [d1..d7, speed]  (8 inputs)  │
  ─────────────▶ │        ↓                     │
                 │  8 → 12 (tanh) → 2 (tanh)    │  = 134 weights = the "genome"
                 │        ↓                     │
                 │  steer, throttle/brake       │
                 └──────────────────────────────┘
```

| Step | File | What happens |
|---|---|---|
| Track | `selfdrive/track.py` | A closed centre line + width, rasterised into a grid so collisions, ray-casting and progress are array lookups. Presets, random tracks (`random:<seed>`), or JSON files. |
| Sense → think → act | `selfdrive/sim.py` | All 100 cars drive at once (vectorised). Rays measure distance to the edge; each car's own network turns that into steering + throttle; leaving the road = crash. |
| Brain | `selfdrive/brain.py` | Tiny MLP. The whole population's weights are one `(100, 134)` matrix. |
| Learning | `selfdrive/evolve.py` | Keep the best 4, breed the rest from good parents (tournament selection, crossover), randomly mutate weights. No backprop. |
| Fitness | `sim.py` | laps driven (0–2) + up to +1 for finishing quickly. Cars that stop making progress are removed (no circling forever). |

### Why the learning carries over to other tracks

The car never sees its x/y position, the track id, or anything global — only
"how far is the wall in these 7 directions" and "how fast am I going". So the
network learns a *driving reflex*, not a memorised route, and the same genome
can be dropped onto any track. How well that reflex generalises depends on
what it was trained on — that's what the test tracks measure:

* `test_progress` – how much of each unseen track the current champion completes (avg).
* `test_finish_rate` – share of unseen tracks it finishes.

Default test set: 6 random tracks + `snake`, `hairpins`, and `circuit` driven in reverse.
Things that improve transfer (try them, compare in the dashboard):

* train on several tracks: `--train preset:circuit,preset:oval,random:1..3`
* fresh random tracks every generation: `--random-per-gen 2` (can never memorise)
* sensor noise: `--sensor-noise 0.03`
* include both directions: `--train preset:circuit,preset:circuit:rev`

## Commands

```bash
# single run (all options: python train.py -h)
python train.py --name my_run --train preset:circuit --generations 80

# batch: grid of settings × seeds, run in parallel, summary table at the end
python batch.py --name tracks_vs_transfer --seeds 3 --generations 50 \
    --grid "train=preset:circuit | preset:circuit,preset:oval,random:1..2" \
    --grid "random-per-gen=0 | 2"

# drop a trained car on any tracks (and save a replay for the dashboard)
python evaluate.py runs/my_run --tracks random:5000..5019,preset:hairpins --save

# dashboard (works while training is running)
python serve.py
```

Track specs (used by `--train`, `--test`, `--tracks`):
`preset:circuit|oval|snake|hairpins`, add `:rev` to drive backwards,
`random:42`, ranges `random:1..10`, or a JSON file like `tracks/example.json`.
Draw your own in the dashboard's **Track editor** tab and save it to `tracks/`.

## Dashboard

* **Run** – KPIs, fitness curves, training-vs-unseen progress, per-track transfer,
  and a replay player (top-20 cars on the training track, the champion on every
  test track; thumbnails show where it crashed).
* **Compare** – tick runs in the sidebar, overlay any metric.
* **Batches** – mean ± min/max over seeds per configuration, summary table.
* **Track images** (Run tab) – the PNGs above, with download links.
* **Track editor** – click to draw a track, download as JSON.

## Output files

```
runs/<batch?>/<run>/
  config.json       full settings (reproducible: same --seed → same result)
  metrics.jsonl     one JSON line per generation  (easy to load in pandas/Jupyter)
  status.json       running/done, generation, list of replays
  best_genome.json  champion network weights
  replays/*.json    recorded drives
  images/tracks/    PNG of every training + test track (+ all_tracks.png overview)
  images/champion/  same, with the best car's path drawn on (green = finished,
                    red X = crash point); refreshed at every replay snapshot
  images/eval_*/    written by evaluate.py
```

PNG images of any tracks, without training:

```bash
python render_tracks.py preset:circuit random:1..8 tracks/example.json --out track_images/
```

Using it from Jupyter:

```python
import pandas as pd
m = pd.read_json("runs/my_run/metrics.jsonl", lines=True)
m.plot(x="gen", y=["best_progress", "test_progress"])
```

## Knobs worth playing with

| Option | Default | Effect |
|---|---|---|
| `--pop` | 100 | cars per generation |
| `--hidden` | `12` | hidden layers, e.g. `16,8`; `--hidden ""` = no hidden layer |
| `--rays`, `--fov` | 7, 180° | sensor count / spread |
| `--mutation-rate`, `--mutation-sigma` | 0.1, 0.3 | how many weights change and by how much |
| `--laps`, `--max-steps` | 2, 1500 | episode length |
| physics | `SimConfig` in `sim.py` | max speed, acceleration, turn rate (turn radius = speed / turn rate, so fast cars must brake for tight corners) |
