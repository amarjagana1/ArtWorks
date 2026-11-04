"""
Pop-art gradient-map (duotone/tritone) filter.

Pipeline (per tile):
  1. Convert the image to greyscale (luminance).
  2. Contrast-stretch so the full tonal range is used.
  3. Build a 256-entry colour lookup table (LUT) by linearly interpolating
     between 3 colours: shadows -> midtones -> highlights.
  4. Replace every grey level with its LUT colour (gradient map).
  5. Tile the 6 colourways into a grid of 3 columns x 2 rows.

Usage:
    python popart_duotone.py imagename.jpg

The result is saved in the Output folder as imagename_popart.png

Requires: numpy, pillow   (pip install numpy pillow)
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

COLS, ROWS = 3, 2

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


def to_grey(img):
    """Greyscale + contrast stretch (ignoring the 1% most extreme pixels). Returns uint8 HxW."""
    grey = ImageOps.autocontrast(img.convert("L"), cutoff=1.0)
    return np.asarray(grey, dtype=np.uint8)


def make_popart(img):
    grey = to_grey(img)
    h, w = grey.shape
    canvas = np.zeros((ROWS * h, COLS * w, 3), dtype=np.uint8)
    for i, palette in enumerate(PALETTES):
        lut = build_lut(*palette)
        r, c = divmod(i, COLS)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = lut[grey]  # gradient map
    return Image.fromarray(canvas)


def main():
    if len(sys.argv) != 2:
        print("Usage: python popart_duotone.py imagename.jpg", file=sys.stderr)
        return 1

    here = Path(__file__).resolve().parent
    src = Path(sys.argv[1])
    if not src.exists() and (here / "Images" / src.name).exists():
        src = here / "Images" / src.name          # fall back to the Images folder
    try:
        img = Image.open(src)
    except Exception as e:
        print(f"Error: can't open {src}: {e}", file=sys.stderr)
        return 1

    out_dir = here / "Output"
    out_dir.mkdir(exist_ok=True)
    dst = out_dir / f"{src.stem}_popart.png"
    make_popart(img).save(dst)
    print(f"Saved {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
