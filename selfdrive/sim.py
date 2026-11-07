"""
Headless, vectorised driving simulator.

All P cars drive simultaneously on one track (they don't collide with each
other). Each step:

  1. sense   : cast N rays from each car, measure distance to the track edge
  2. think   : feed [ray distances, speed] through each car's own network
  3. act     : network outputs steer + throttle -> simple kinematic car model
  4. score   : look up the car's position on the track raster:
                 off track         -> crashed (car stops)
                 nearest centerline index -> progress along the track
               no new progress for `stall_steps` -> stopped (prevents circling)

Why this generalises to other tracks: the car only ever sees *relative*
information (what its sensors see, how fast it goes). It never knows its x/y
position or which track it is on, so the same network can be dropped onto any
track.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .brain import Brain
from .track import Track


@dataclass
class SimConfig:
    dt: float = 0.1             # seconds per step
    max_speed: float = 200.0    # world units / s
    accel: float = 100.0        # units / s^2 at full throttle
    brake: float = 160.0        # units / s^2 at full brake
    drag: float = 0.3           # fraction of speed lost per second when coasting
    turn_rate: float = 1.8      # rad / s at full steer (turn radius = v / turn_rate)
    n_rays: int = 7
    ray_fov_deg: float = 180.0  # rays spread evenly over this angle
    ray_length: float = 160.0
    ray_step: float = 4.0       # ray-march resolution
    sensor_noise: float = 0.0   # std-dev of noise added to normalised sensors
    laps: int = 2
    max_steps: int = 1500
    stall_steps: int = 60       # kill a car that makes no new progress for this long

    @property
    def n_inputs(self) -> int:
        return self.n_rays + 1

    def to_dict(self):
        return asdict(self)


class Simulator:
    def __init__(self, track: Track, cfg: SimConfig):
        self.track = track
        self.cfg = cfg
        half = np.deg2rad(cfg.ray_fov_deg) / 2
        self.ray_angles = np.linspace(-half, half, cfg.n_rays)
        self.ray_d = np.arange(cfg.ray_step, cfg.ray_length + 1e-9, cfg.ray_step)

    # ------------------------------------------------------------------ #
    def sense(self, pos: np.ndarray, heading: np.ndarray) -> np.ndarray:
        """Ray-march every ray of every car; return distance to first off-track
        sample. pos (P,2), heading (P,) -> (P, n_rays) distances."""
        ang = heading[:, None] + self.ray_angles[None]                      # (P,R)
        dirs = np.stack([np.cos(ang), np.sin(ang)], -1)                     # (P,R,2)
        pts = pos[:, None, None] + dirs[:, :, None] * self.ray_d[None, None, :, None]
        on, _ = self.track.lookup(pts)                                       # (P,R,S)
        hit = ~on
        first = np.argmax(hit, axis=-1)
        return np.where(hit.any(-1), self.ray_d[first], self.cfg.ray_length)

    # ------------------------------------------------------------------ #
    def run(self, brain: Brain, genomes: np.ndarray, rng: np.random.Generator | None = None,
            record: bool = False) -> dict:
        cfg, tr = self.cfg, self.track
        P = genomes.shape[0]
        params = brain.unpack(genomes)
        rng = rng or np.random.default_rng(0)

        pos = np.tile(tr.start_pos, (P, 1))
        heading = np.full(P, tr.start_heading)
        speed = np.zeros(P)
        alive = np.ones(P, bool)
        crashed = np.zeros(P, bool)
        idx = np.zeros(P, np.int64)          # nearest centerline point
        progress = np.zeros(P)               # signed, in centerline points
        best = np.zeros(P)                   # furthest progress reached
        last_improve = np.zeros(P, np.int64)
        finish_step = np.full(P, -1)
        steps_alive = np.zeros(P, np.int64)
        target = cfg.laps * tr.n

        traj, rays_log = [], []
        for t in range(cfg.max_steps):
            if not alive.any():
                break
            ai = np.flatnonzero(alive)  # only simulate cars that are still driving
            # 1. sense
            dist = np.full((P, cfg.n_rays), cfg.ray_length)
            dist[ai] = self.sense(pos[ai], heading[ai])
            obs = np.concatenate([dist / cfg.ray_length, (speed / cfg.max_speed)[:, None]], 1)
            if cfg.sensor_noise:
                obs = obs + rng.normal(0, cfg.sensor_noise, obs.shape)
            if record:
                traj.append(np.column_stack([pos, heading]).astype(np.float32))
                rays_log.append(dist.astype(np.float32))
            # 2. think
            out = brain.forward([(W[ai], b[ai]) for W, b in params], obs[ai])
            steer, throttle = out[:, 0], out[:, 1]
            # 3. act
            a = np.where(throttle >= 0, throttle * cfg.accel, throttle * cfg.brake)
            speed[ai] = np.clip(speed[ai] + (a - cfg.drag * speed[ai]) * cfg.dt, 0, cfg.max_speed)
            heading[ai] += steer * cfg.turn_rate * cfg.dt
            pos[ai] += np.stack([np.cos(heading[ai]), np.sin(heading[ai])], 1) \
                * speed[ai, None] * cfg.dt
            steps_alive += alive
            # 4. score
            on, new_idx = tr.lookup(pos)
            hit = alive & ~on
            crashed |= hit
            alive &= ~hit
            d = new_idx - idx
            d = np.where(d > tr.n // 2, d - tr.n, d)
            d = np.where(d < -tr.n // 2, d + tr.n, d)
            progress = np.where(alive, progress + d, progress)
            idx = np.where(alive, new_idx, idx)
            improved = alive & (progress > best)
            best = np.where(improved, progress, best)
            last_improve = np.where(improved, t, last_improve)
            done = alive & (best >= target)
            finish_step[done] = t + 1
            alive &= ~done
            alive &= (t - last_improve) < cfg.stall_steps

        laps_done = np.minimum(best, target) / tr.n
        finished = finish_step >= 0
        # Fitness: laps driven (0..laps) + up to +1 bonus for finishing fast.
        fitness = laps_done + np.where(finished, 1 - finish_step / cfg.max_steps, 0.0)
        res = dict(
            fitness=fitness,
            laps=laps_done,
            progress=laps_done / cfg.laps,          # 0..1 fraction of the course
            finished=finished,
            crashed=crashed,
            finish_time=np.where(finished, finish_step * cfg.dt, np.nan),
            steps=steps_alive,
        )
        if record:
            res["traj"] = np.stack(traj, 1) if traj else np.zeros((P, 0, 3))  # (P,T,3)
            res["rays"] = np.stack(rays_log, 1) if rays_log else np.zeros((P, 0, cfg.n_rays))
        return res
