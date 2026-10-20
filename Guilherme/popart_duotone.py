#!/usr/bin/env python3
"""
Pop-art gradient-map (duotone/tritone) filter.

Pipeline (per tile):
  1. Convert the image to greyscale (luminance).
  2. Contrast-stretch + optional gamma so the full tonal range is used.
  3. Build a 256-entry colour lookup table (LUT) by linearly interpolating
     between 3 colours: shadows -> midtones -> highlights.
  4. Replace every grey level with its LUT colour (gradient map).
  The highlight colour is also the tile's background colour, so a photo
  shot on a white/light background gets a flat coloured background.
  5. Tile the results into a grid (default 3 columns x 2 rows).

Usage:
    python popart_duotone.py input.jpg                  # -> input_popart.png
    python popart_duotone.py input.jpg -o out.png --cols 3 --rows 2
    python popart_duotone.py input.jpg --tile-size 400 --gamma 1.2
    python popart_duotone.py input.jpg --single 2       # only palette #2

Requires: numpy, pillow   (pip install numpy pillow)
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# (shadow, midtone, highlight/background) in RGB.
PALETTES = [
    ((35, 60, 120),  (190, 60, 60),   (255, 60, 50)),     # red
    ((110, 40, 130), (200, 90, 100),  (245, 215, 110)),   # yellow
    ((20, 30, 230),  (40, 110, 255),  (0, 255, 255)),     # cyan
    ((130, 40, 150), (60, 150, 180),  (0, 225, 195)),     # teal
    ((150, 20, 20),  (190, 80, 80),   (140, 215, 225)),   # light blue
    ((20, 110, 90),  (130, 170, 90),  (255, 240, 150)),   # pale yellow
]


def build_lut(shadow, mid, light):
    """256x3 uint8 LUT interpolating shadow -> mid -> light."""
    xs = [0, 127, 255]
    stops = np.array([shadow, mid, light], dtype=np.float64)
    grey = np.arange(256)
    lut = np.stack([np.interp(grey, xs, stops[:, c]) for c in range(3)], axis=1)
    return lut.round().astype(np.uint8)


def to_grey(img, cutoff=1.0, gamma=1.0):
    """Greyscale, percentile contrast stretch, gamma. Returns uint8 HxW."""
    grey = ImageOps.autocontrast(img.convert("L"), cutoff=cutoff)
    a = np.asarray(grey, dtype=np.float64) / 255.0
    a = a ** gamma  # gamma > 1 darkens, < 1 brightens
    return (a * 255).round().astype(np.uint8)


def gradient_map(grey, lut):
    """Apply LUT to a uint8 greyscale array -> HxWx3 uint8."""
    return lut[grey]


def make_popart(img, cols=3, rows=2, tile_size=None, cutoff=1.0, gamma=1.0,
                palettes=PALETTES):
    if tile_size:
        img = ImageOps.contain(img, (tile_size, tile_size))
    grey = to_grey(img, cutoff, gamma)
    h, w = grey.shape
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i in range(rows * cols):
        lut = build_lut(*palettes[i % len(palettes)])
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = gradient_map(grey, lut)
    return Image.fromarray(canvas)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", type=Path)
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--tile-size", type=int, default=None,
                   help="max width/height of each tile in px")
    p.add_argument("--cutoff", type=float, default=1.0,
                   help="%% of extreme pixels ignored in contrast stretch")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--single", type=int, metavar="N",
                   help="output only palette N (1-%d), no grid" % len(PALETTES))
    a = p.parse_args()

    img = Image.open(a.input)
    if a.single:
        pal = [PALETTES[(a.single - 1) % len(PALETTES)]]
        out = make_popart(img, 1, 1, a.tile_size, a.cutoff, a.gamma, pal)
    else:
        out = make_popart(img, a.cols, a.rows, a.tile_size, a.cutoff, a.gamma)

    dest = a.output or a.input.with_name(a.input.stem + "_popart.png")
    out.save(dest)
    print("Saved", dest)


if __name__ == "__main__":
    main()
