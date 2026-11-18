"""Emboss: make a photo look like a raised metal relief.

Combines x and y Sobel gradients so the light can come from any direction
(45 degrees = diagonal, like the classic emboss kernel). Flat areas become
mid-grey. Developed in emma_filters.ipynb.

Usage:
    python emboss.py imagename.jpg

The result is saved in images/output as imagename_emboss.png
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def to_gray_float(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return gray.astype(np.float32)


def directional_relief(image, angle, strength):
    """Relief from any light direction: combine x and y Sobel gradients using the angle."""
    gray = to_gray_float(image)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)   # change left -> right
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)   # change top -> bottom
    theta = np.deg2rad(angle)
    return strength * (np.cos(theta) * gx + np.sin(theta) * gy)


def emboss_directional(image, angle=45.0, strength=1.0):
    """
    Emboss with an adjustable light direction (angle in degrees).
    0 = left-right relief, 90 = top-bottom, 45 = diagonal (like the classic kernel).
    Rotating by 180 degrees flips raised areas into sunken ones (deboss).
    """
    if strength < 0:
        raise ValueError("strength must be >= 0")
    relief = directional_relief(image, angle, strength)
    return np.clip(relief + 128, 0, 255).astype(np.uint8)


def main():
    if len(sys.argv) != 2:
        print("Usage: python emboss.py imagename.jpg", file=sys.stderr)
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
    dst = out_dir / f"{src.stem}_emboss.png"
    cv2.imwrite(str(dst), emboss_directional(img))
    print(f"Saved {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())