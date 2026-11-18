"""Pencil Sketch: turn a photo into a grey pencil drawing.

Uses a median blur and a colour dodge blend: flat areas become white paper,
edges and details become dark pencil strokes. Developed in emma_filters.ipynb.

Usage:
    python pencil_sketch.py imagename.jpg

The result is saved in images/output as imagename_sketch.png
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def pencil_sketch_median(image, blur=21):
    """
    Pencil sketch using a median blur and a colour dodge blend.
    blur: odd kernel size >= 3. Larger = bolder strokes and broader shading.
    """
    if blur < 3 or blur % 2 == 0:
        raise ValueError("blur must be an odd integer >= 3")

    # Convert to grayscale (skip if the image is already single-channel)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Median blur: non-linear, keeps strong edges sharp and removes small specks
    blurred = cv2.medianBlur(gray, blur)

    # Colour dodge: flat areas -> white (paper), edges/details -> dark (pencil strokes)
    return cv2.divide(gray, blurred, scale=256)


def main():
    if len(sys.argv) != 2:
        print("Usage: python pencil_sketch.py imagename.jpg", file=sys.stderr)
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
    dst = out_dir / f"{src.stem}_sketch.png"
    cv2.imwrite(str(dst), pencil_sketch_median(img))
    print(f"Saved {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())