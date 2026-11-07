"""
Render tracks (and optionally a car's driving line) to PNG images.

Pure numpy + the standard library (zlib) — no Pillow/matplotlib needed.

    render_track(track, path=None)          -> RGB array (H, W, 3)
    save_png(rgb, "file.png")
    contact_sheet([(img, label), ...])      -> one image with all tracks

Picture legend:
    grey road, dark edges, black/white chequered start line,
    small chevrons showing the driving direction,
    car path (if given): green = finished, red = crashed/stopped (X at the end).
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from .track import Track

# colours
BG = (246, 247, 249)
ROAD = (214, 218, 225)
EDGE = (90, 98, 112)
ARROW = (150, 158, 172)
GOOD = (22, 163, 74)
BAD = (220, 38, 38)
INK = (27, 31, 36)


# --------------------------------------------------------------------------- #
# PNG writer
# --------------------------------------------------------------------------- #
def png_bytes(rgb: np.ndarray) -> bytes:
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    return png


def save_png(rgb: np.ndarray, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes(rgb))
    return path


# --------------------------------------------------------------------------- #
# tiny bitmap font (5x7) for labels
# --------------------------------------------------------------------------- #
_FONT = {
    "A": "01110100011000111111100011000110001", "B": "11110100011000111110100011000111110",
    "C": "01110100011000010000100001000101110", "D": "11110100011000110001100011000111110",
    "E": "11111100001000011110100001000011111", "F": "11111100001000011110100001000010000",
    "G": "01110100011000010111100011000101111", "H": "10001100011000111111100011000110001",
    "I": "01110001000010000100001000010001110", "J": "00111000100001000010000101001001100",
    "K": "10001100101010011000101001001010001", "L": "10000100001000010000100001000011111",
    "M": "10001110111010110101100011000110001", "N": "10001100011100110101100111000110001",
    "O": "01110100011000110001100011000101110", "P": "11110100011000111110100001000010000",
    "Q": "01110100011000110001101011001001101", "R": "11110100011000111110101001001010001",
    "S": "01111100001000001110000010000111110", "T": "11111001000010000100001000010000100",
    "U": "10001100011000110001100011000101110", "V": "10001100011000110001100010101000100",
    "W": "10001100011000110101101011010101010", "X": "10001100010101000100010101000110001",
    "Y": "10001100010101000100001000010000100", "Z": "11111000010001000100010001000011111",
    "0": "01110100011001110101110011000101110", "1": "00100011000010000100001000010001110",
    "2": "01110100010000100010001000100011111", "3": "11111000100010000010000011000101110",
    "4": "00010001100101010010111110001000010", "5": "11111100001111000001000011000101110",
    "6": "00110010001000011110100011000101110", "7": "11111000010001000100010000100001000",
    "8": "01110100011000101110100011000101110", "9": "01110100011000101111000010001001100",
    ":": "00000011000110000000011000110000000", ".": "00000000000000000000000000110001100",
    "-": "00000000000000011111000000000000000", "%": "11000110010001000100010001001100011",
    "/": "00001000010001000100010001000010000", " ": "0" * 35, "_": "00000000000000000000000000000011111",
    "(": "00010001000100001000010000010000010", ")": "01000001000001000010000100010001000",
    "+": "00000001000010011111001000010000000",
}


def draw_text(img, x, y, text, color=INK, scale=2):
    for ch in text.upper():
        bits = _FONT.get(ch, _FONT[" "])
        for i, b in enumerate(bits):
            if b == "1":
                r, c = divmod(i, 5)
                y0, x0 = y + r * scale, x + c * scale
                img[max(y0, 0):y0 + scale, max(x0, 0):x0 + scale] = color
        x += 6 * scale
    return img


# --------------------------------------------------------------------------- #
# drawing helpers
# --------------------------------------------------------------------------- #


def _polyline(img, pts, color, r):
    """Thick polyline through pixel-space points (N,2)."""
    if len(pts) < 2:
        return
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    n = np.maximum(np.ceil(seg / max(r * 0.7, 0.5)).astype(int), 1)
    dense = np.concatenate([np.linspace(a, b, k, endpoint=False) for a, b, k in zip(pts[:-1], pts[1:], n)]
                           + [pts[-1:]])
    h, w, _ = img.shape
    ri = int(np.ceil(r))
    oy, ox = np.mgrid[-ri:ri + 1, -ri:ri + 1]
    keep = ox ** 2 + oy ** 2 <= r * r
    oy, ox = oy[keep], ox[keep]
    ys = (np.round(dense[:, 1])[:, None] + oy).ravel().astype(int)
    xs = (np.round(dense[:, 0])[:, None] + ox).ravel().astype(int)
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    img[ys[ok], xs[ok]] = color


def _triangle(img, a, b, c, color):
    xs, ys = [a[0], b[0], c[0]], [a[1], b[1], c[1]]
    h, w, _ = img.shape
    x0, x1 = int(max(min(xs), 0)), int(min(max(xs) + 1, w))
    y0, y1 = int(max(min(ys), 0)), int(min(max(ys) + 1, h))
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1] + 0.5

    def side(p, q):
        return (q[0] - p[0]) * (yy - p[1]) - (q[1] - p[1]) * (xx - p[0])

    s1, s2, s3 = side(a, b), side(b, c), side(c, a)
    inside = ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))
    img[y0:y1, x0:x1][inside] = color


# --------------------------------------------------------------------------- #
# main renderer
# --------------------------------------------------------------------------- #
def render_track(track: Track, path_xy: np.ndarray | None = None, finished: bool | None = None,
                 size: int = 640, label: str | None = None, pad: int = 16) -> np.ndarray:
    """Return an RGB image of the track; optionally draw a car's path (N,2+) on it."""
    lo = np.minimum(track.left.min(0), track.right.min(0))
    hi = np.maximum(track.left.max(0), track.right.max(0))
    span = hi - lo
    scale = (size - 2 * pad) / span.max()
    W = int(np.ceil(span[0] * scale)) + 2 * pad
    H = int(np.ceil(span[1] * scale)) + 2 * pad
    top = 22 if label else 0

    def to_px(p):
        p = np.asarray(p, float)
        return np.stack([(p[..., 0] - lo[0]) * scale + pad, (p[..., 1] - lo[1]) * scale + pad + top], -1)

    img = np.empty((H + top, W, 3), np.uint8)
    img[:] = BG

    # road: look up every pixel centre on the track raster
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    world = np.stack([(xx - pad) / scale + lo[0], (yy - pad) / scale + lo[1]], -1)
    on, _ = track.lookup(world)
    body = img[top:]
    body[on] = ROAD
    # edges = road pixels next to a non-road pixel
    pad_on = np.pad(on, 1)
    inner = (pad_on[:-2, 1:-1] & pad_on[2:, 1:-1] & pad_on[1:-1, :-2] & pad_on[1:-1, 2:])
    edge = on & ~inner
    edge = edge | np.pad(edge, 1)[:-2, 1:-1] | np.pad(edge, 1)[1:-1, :-2]  # 2px thick
    body[edge & on] = EDGE

    # direction chevrons along the centre line
    step = max(int(220 / 4.0), 1)  # every ~220 world units
    for i in range(step // 2, track.n, step):
        c = to_px(track.center[i])
        t = track.tangent[i]
        nrm = np.array([-t[1], t[0]])
        s = max(track.half_w[i] * scale * 0.45, 4)
        tip = c + t * s
        _triangle(img, tip, c - t * s * 0.6 + nrm * s * 0.8, c - t * s * 0.6 - nrm * s * 0.8, ARROW)

    # chequered start line
    a, b = to_px(track.left[0]), to_px(track.right[0])
    n_sq = 6
    for k in range(n_sq):
        col = INK if k % 2 == 0 else (255, 255, 255)
        _polyline(img, np.stack([a + (b - a) * k / n_sq, a + (b - a) * (k + 1) / n_sq]), col, r=3.5)

    # car path
    if path_xy is not None and len(path_xy) > 1:
        color = GOOD if finished else BAD
        pts = to_px(np.asarray(path_xy)[:, :2])
        _polyline(img, pts, color, r=1.6)
        if not finished:  # mark the crash point with an X
            e = pts[-1]
            for d in (np.array([1, 1]), np.array([1, -1])):
                _polyline(img, np.stack([e - d * 6, e + d * 6]), BAD, r=1.6)

    if label:
        draw_text(img, 8, 4, label, INK, 2)
    return img


def contact_sheet(items: list[tuple[np.ndarray, str]], cols: int = 4, gap: int = 10) -> np.ndarray:
    """Tile several images (each with a caption already drawn) into one."""
    cols = min(cols, len(items))
    rows = -(-len(items) // cols)
    cw = max(i.shape[1] for i, _ in items)
    ch = max(i.shape[0] for i, _ in items)
    sheet = np.empty((rows * (ch + gap) + gap, cols * (cw + gap) + gap, 3), np.uint8)
    sheet[:] = (255, 255, 255)
    for k, (im, _) in enumerate(items):
        r, c = divmod(k, cols)
        y, x = gap + r * (ch + gap), gap + c * (cw + gap)
        sheet[y:y + im.shape[0], x:x + im.shape[1]] = im
    return sheet


def save_track_images(out_dir, tracks: list[tuple[Track, str]], paths: dict | None = None,
                      size: int = 640, sheet_name: str = "all_tracks.png") -> list[Path]:
    """Write one PNG per track + an overview sheet.
    tracks: [(track, role)], role e.g. 'train' / 'test'.
    paths : optional {track.name: (path_xy, finished, caption_suffix)} to draw a car's line."""
    out = Path(out_dir)
    written, thumbs = [], []
    for tr, role in tracks:
        p = (paths or {}).get(tr.name)
        cap = f"{role}: {tr.name}" + (f"  {p[2]}" if p and len(p) > 2 else "")
        img = render_track(tr, p[0] if p else None, p[1] if p else None, size=size, label=cap)
        fname = f"{role}_{tr.name.replace(':', '_').replace('/', '_')}.png"
        written.append(save_png(img, out / fname))
        short = tr.name.replace("preset:", "").replace("random:", "rnd ").replace("file:", "")
        tcap = f"{role} {short}" + (f" {p[2]}" if p and len(p) > 2 else "")
        thumbs.append((render_track(tr, p[0] if p else None, p[1] if p else None, size=320,
                                    label=tcap[:26]), tcap))
    if thumbs:
        written.append(save_png(contact_sheet(thumbs), out / sheet_name))
    return written
