"""Thermal Vision: turns a photo into a heat-camera style image.

Maps each pixel's brightness to a colour scale (black -> purple -> red ->
orange -> yellow -> white) using a hand-built colour lookup table, contrast
stretching and a light Gaussian blur. Developed in emma_filters.ipynb.

Usage:
    python thermal_vision.py imagename.jpg

The result is saved in images/output as imagename_thermal.png
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def make_gaussian_kernel(size, sigma):
    # kernel size has to be odd so there's a centre pixel
    if size % 2 == 0:
        size += 1
    half = size // 2
    x = np.arange(-half, half + 1)
    xx, yy = np.meshgrid(x, x)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return (kernel / kernel.sum()).astype(np.float32)


def make_thermal_lut():
    # key points on the brightness scale (0-255) and the colour at each one in RGB
    positions = [0, 50, 100, 150, 200, 230, 255]
    colours = np.array([
        (0, 0, 0),        # black - coldest
        (30, 0, 90),      # dark blue
        (120, 0, 140),    # purple
        (220, 30, 30),    # red
        (255, 130, 0),    # orange
        (255, 220, 0),    # yellow
        (255, 255, 255),  # white - hottest
    ], dtype=np.float32)

    # fill in the 256 colours in between by blending smoothly between the key points
    x = np.arange(256)
    r = np.interp(x, positions, colours[:, 0])
    g = np.interp(x, positions, colours[:, 1])
    b = np.interp(x, positions, colours[:, 2])

    # opencv uses BGR order so stack them that way
    return np.stack([b, g, r], axis=1).astype(np.uint8)


def thermal_vision(image, blur_sigma=2, low_pct=1, high_pct=99):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # contrast stretch
    lo, hi = np.percentile(gray, (low_pct, high_pct))
    stretched = np.clip((gray - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)

    # slight blur to mimic a low resolution thermal sensor
    if blur_sigma > 0:
        k = make_gaussian_kernel(6 * blur_sigma + 1, blur_sigma)
        stretched = cv2.filter2D(stretched, -1, k, borderType=cv2.BORDER_REFLECT)

    # map brightness to the thermal colours
    return make_thermal_lut()[stretched]


def main():
    if len(sys.argv) != 2:
        print("Usage: python thermal_vision.py imagename.jpg", file=sys.stderr)
        return 1

    here = Path(__file__).resolve().parent
    src = Path(sys.argv[1])
    if not src.exists() and (here / "images" / "input" / src.name).exists():
        src = here / "images" / "input" / src.name  # fall back to the input folder

    img = cv2.imread(str(src))
    if img is None:
        print(f"Error: can't open {src}", file=sys.stderr)
        return 1

    out_dir = here / "images" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{src.stem}_thermal.png"
    cv2.imwrite(str(dst), thermal_vision(img))
    print(f"Saved {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
