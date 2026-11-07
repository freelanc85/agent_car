"""
Tracks: a closed centerline + a width.

Everything the simulator needs is precomputed into a raster grid ("occupancy
map") so that collision checks, ray-casting and progress tracking are simple
array lookups:

    grid_idx[y, x]  -> index of the nearest centerline point (or -1)
    grid_on[y, x]   -> True if that cell is drivable

Track specs (strings) used everywhere on the command line:

    preset:circuit        hand-designed track (see PRESETS)
    preset:circuit:rev    same track, driven in the opposite direction
    random:42             procedurally generated track from seed 42
    tracks/my_track.json  track file (e.g. drawn in the dashboard editor)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

CELL = 2.0  # raster resolution (world units per cell)
SPACING = 4.0  # distance between resampled centerline points

# Hand-designed tracks (control points, smoothed with a closed Catmull-Rom spline).
PRESETS = {
    # The default "specific" training track: mixed corners, a hairpin, a chicane.
    "circuit": dict(
        width=50,
        points=[
            (0, 0), (300, 0), (450, 40), (520, 150), (480, 260), (380, 290),
            (300, 250), (220, 290), (210, 380), (290, 450), (480, 440),
            (600, 380), (700, 420), (720, 540), (580, 630), (200, 630),
            (40, 570), (-60, 430), (-80, 200),
        ],
    ),
    "oval": dict(
        width=60,
        points=[(0, 0), (500, 0), (620, 80), (620, 220), (500, 300),
                (0, 300), (-120, 220), (-120, 80)],
    ),
    "snake": dict(
        width=48,
        points=[
            (0, 0), (150, -70), (300, 0), (450, -70), (600, 0), (720, 120),
            (680, 280), (540, 330), (400, 260), (250, 330), (100, 270),
            (-60, 300), (-140, 160),
        ],
    ),
    "hairpins": dict(
        width=46,
        points=[
            (0, 0), (275, 0), (550, 0), (650, 25), (700, 90), (660, 155),
            (560, 180), (390, 180), (220, 180), (130, 205), (100, 270),
            (140, 330), (230, 355), (395, 360), (560, 360), (660, 385),
            (700, 450), (650, 515), (550, 540), (275, 540), (0, 540),
            (-90, 510), (-130, 400), (-130, 270), (-120, 140), (-90, 45),
        ],
    ),
}


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def catmull_rom_closed(ctrl: np.ndarray, samples_per_seg: int = 24) -> np.ndarray:
    """Smooth closed curve through the control points (works on any #columns)."""
    P = np.asarray(ctrl, float)
    n = len(P)
    t = np.linspace(0, 1, samples_per_seg, endpoint=False)[:, None]
    t2, t3 = t * t, t * t * t
    out = []
    for i in range(n):
        p0, p1, p2, p3 = P[(i - 1) % n], P[i], P[(i + 1) % n], P[(i + 2) % n]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.vstack(out)


def resample_closed(pts: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a closed polyline to (roughly) equal spacing. Extra columns
    (e.g. width) are interpolated along with x, y."""
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(s[-1] / spacing), 16)
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, closed[:, k]) for k in range(pts.shape[1])], axis=1)


# --------------------------------------------------------------------------- #
# Track
# --------------------------------------------------------------------------- #
class Track:
    def __init__(self, name: str, center: np.ndarray, width, reverse: bool = False):
        center = np.asarray(center, float)
        w = np.broadcast_to(np.asarray(width, float), (len(center),))
        pts = np.column_stack([center, w])
        if reverse:
            pts = pts[::-1]
        pts = resample_closed(pts, SPACING)

        self.name = name
        self.center = pts[:, :2]
        self.half_w = pts[:, 2] / 2
        self.n = len(self.center)

        # Tangents / normals / walls
        d = np.roll(self.center, -1, 0) - np.roll(self.center, 1, 0)
        self.tangent = d / np.linalg.norm(d, axis=1, keepdims=True)
        self.normal = np.column_stack([-self.tangent[:, 1], self.tangent[:, 0]])
        self.left = self.center + self.normal * self.half_w[:, None]
        self.right = self.center - self.normal * self.half_w[:, None]
        self.length = float(np.sum(np.linalg.norm(
            np.roll(self.center, -1, 0) - self.center, axis=1)))

        self.start_pos = self.center[0].copy()
        self.start_heading = float(math.atan2(self.tangent[0, 1], self.tangent[0, 0]))
        self._build_grid()

    # -- raster ------------------------------------------------------------- #
    def _build_grid(self):
        pad = self.half_w.max() + 4 * SPACING + 2 * CELL
        lo = self.center.min(0) - pad
        hi = self.center.max(0) + pad
        self.origin = lo
        W = int(np.ceil((hi[0] - lo[0]) / CELL)) + 1
        H = int(np.ceil((hi[1] - lo[1]) / CELL)) + 1
        dist = np.full((H, W), np.inf, np.float32)
        idx = np.full((H, W), -1, np.int32)

        r = int(np.ceil((self.half_w.max() + 2 * SPACING) / CELL))
        oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
        oy, ox = oy.ravel(), ox.ravel()
        for i, p in enumerate(self.center):
            cx = int((p[0] - lo[0]) / CELL)
            cy = int((p[1] - lo[1]) / CELL)
            xs, ys = cx + ox, cy + oy
            wx = lo[0] + (xs + 0.5) * CELL
            wy = lo[1] + (ys + 0.5) * CELL
            dd = np.hypot(wx - p[0], wy - p[1]).astype(np.float32)
            better = dd < dist[ys, xs]
            dist[ys[better], xs[better]] = dd[better]
            idx[ys[better], xs[better]] = i

        self.grid_idx = idx
        self.grid_on = (idx >= 0) & (dist <= self.half_w[np.clip(idx, 0, None)])
        self.grid_idx[~self.grid_on] = -1

    def lookup(self, xy: np.ndarray):
        """xy: (..., 2) world coords -> (on_track bool[...], centerline idx int[...])."""
        cx = np.floor((xy[..., 0] - self.origin[0]) / CELL).astype(np.int64)
        cy = np.floor((xy[..., 1] - self.origin[1]) / CELL).astype(np.int64)
        H, W = self.grid_on.shape
        inside = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
        cx = np.clip(cx, 0, W - 1)
        cy = np.clip(cy, 0, H - 1)
        on = self.grid_on[cy, cx] & inside
        return on, self.grid_idx[cy, cx]

    # -- validation ---------------------------------------------------------- #
    def problems(self) -> list[str]:
        """Detect walls that fold over themselves or sections that overlap."""
        out = []
        # Radius of curvature must exceed the half width, or the inner wall folds.
        k = 3  # measure over ~12 units to ignore spline jitter
        a = np.roll(self.center, k, 0)
        b = self.center
        c = np.roll(self.center, -k, 0)
        ab, bc, ca = (np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                      np.linalg.norm(a - c, axis=1))
        cross = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
        radius = ab * bc * ca / np.maximum(2 * cross, 1e-9)
        if np.min(radius) < self.half_w.max() * 1.05:
            out.append(f"corner too tight (min radius {np.min(radius):.0f} "
                       f"< half width {self.half_w.max():.0f})")
        # Non-neighbouring sections closer than the sum of their half widths.
        D = np.linalg.norm(self.center[:, None] - self.center[None], axis=2)
        i = np.arange(self.n)
        circ = np.abs(i[:, None] - i[None])
        circ = np.minimum(circ, self.n - circ) * SPACING
        need = self.half_w[:, None] + self.half_w[None] + 12
        bad = (circ > need * 2.5) & (D < need)
        if bad.any():
            out.append("track sections overlap")
        return out

    # -- export -------------------------------------------------------------- #
    def to_json(self, step: int = 2) -> dict:
        r = lambda a: np.round(a[::step], 1).tolist()  # noqa: E731
        return dict(name=self.name, center=r(self.center), left=r(self.left),
                    right=r(self.right), length=round(self.length, 1),
                    start=[*np.round(self.start_pos, 1).tolist(), self.start_heading])


# --------------------------------------------------------------------------- #
# Constructors
# --------------------------------------------------------------------------- #
def preset_track(name: str, reverse: bool = False) -> Track:
    p = PRESETS[name]
    pts = catmull_rom_closed(np.array(p["points"], float))
    return Track(f"preset:{name}" + (":rev" if reverse else ""), pts, p["width"], reverse)


def random_track(seed: int) -> Track:
    """Star-shaped random loop (radius = sum of a few random harmonics), with
    varying width and random driving direction. Retries until valid."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    for _ in range(500):
        R = rng.uniform(260, 380)
        r = np.ones_like(theta)
        for k in range(2, 6):
            r += rng.uniform(0, 0.28 / k ** 0.6) * np.cos(k * theta + rng.uniform(0, 2 * np.pi))
        pts = np.column_stack([np.cos(theta), np.sin(theta)]) * (r * R)[:, None]
        pts[:, 0] *= rng.uniform(0.75, 1.3)
        base = rng.uniform(40, 60)
        width = base * (1 + rng.uniform(0, 0.25) *
                        np.sin(rng.integers(1, 4) * theta + rng.uniform(0, 2 * np.pi)))
        pts = np.roll(pts, rng.integers(len(pts)), axis=0)  # random start point
        width = np.roll(width, rng.integers(len(pts)))
        t = Track(f"random:{seed}", pts, width, reverse=bool(rng.integers(2)))
        if not t.problems():
            return t
    raise RuntimeError(f"could not generate a valid random track for seed {seed}")


def file_track(path: str) -> Track:
    data = json.loads(Path(path).read_text())
    pts = np.array(data["points"], float)
    if data.get("smooth", True):
        pts = catmull_rom_closed(pts)
        w = data.get("width", 50)
        if np.ndim(w):  # per-control-point widths -> spline them too
            pts = catmull_rom_closed(np.column_stack([data["points"], w]))
            return Track(f"file:{Path(path).stem}", pts[:, :2], pts[:, 2])
    return Track(f"file:{Path(path).stem}", pts, data.get("width", 50))


def load_track(spec: str) -> Track:
    spec = spec.strip()
    if spec.startswith("preset:"):
        parts = spec.split(":")
        if parts[1] not in PRESETS:
            raise ValueError(f"unknown preset '{parts[1]}' (choose from {', '.join(PRESETS)})")
        return preset_track(parts[1], reverse=len(parts) > 2 and parts[2] == "rev")
    if spec.startswith("random:"):
        return random_track(int(spec.split(":")[1]))
    if spec.startswith("file:"):
        spec = spec[5:]
    t = file_track(spec)
    for msg in t.problems():
        print(f"warning: {spec}: {msg} — cars may not be able to drive it cleanly")
    return t


def parse_track_list(specs: str) -> list[str]:
    """'preset:circuit,random:1..5' -> ['preset:circuit','random:1',...,'random:5']"""
    out = []
    for s in specs.split(","):
        s = s.strip()
        if not s:
            continue
        if s.startswith("random:") and ".." in s:
            a, b = s[7:].split("..")
            out += [f"random:{i}" for i in range(int(a), int(b) + 1)]
        else:
            out.append(s)
    return out
