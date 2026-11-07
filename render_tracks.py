#!/usr/bin/env python3
"""
Save PNG images of any tracks (no training needed).

    python render_tracks.py preset:circuit random:1..8 tracks/example.json
    python render_tracks.py preset:hairpins --out images/ --size 1000

Writes one PNG per track plus all_tracks.png (an overview of all of them).
"""
from __future__ import annotations

import argparse

from selfdrive.render import save_track_images
from selfdrive.track import load_track, parse_track_list


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tracks", nargs="+", help="track specs (comma lists and random:a..b ranges work)")
    p.add_argument("--out", default="track_images")
    p.add_argument("--size", type=int, default=640, help="longest side in pixels")
    a = p.parse_args()
    specs = [s for t in a.tracks for s in parse_track_list(t)]
    files = save_track_images(a.out, [(load_track(s), "track") for s in specs], size=a.size)
    for f in files:
        print(f)


if __name__ == "__main__":
    main()
