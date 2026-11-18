"""Vignette: darken the edges of a photo so the eye is drawn to the centre.

Builds a smooth radial mask from two 1D Gaussian kernels and multiplies
it into the image. Developed in emma_filters.ipynb.

Usage:
    python vignette.py imagename.jpg

The result is saved in images/output as imagename_vignette.png
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def vignette_filter(image, strength=0.5):
    """
    Apply vignette effect using 1D Gaussian kernels for a smooth, natural falloff.
    strength: 0.0 (no effect) to 1.0 (strong/dramatic vignette)
    """
    rows, cols = image.shape[:2]

    # Map strength to standard deviation (sigma) scaling
    sigma_scale = max(0.1, 1.2 - (strength * 0.9))

    # Generate 1D Gaussian kernels for X and Y axes
    x_axis = cv2.getGaussianKernel(cols, cols * sigma_scale)
    y_axis = cv2.getGaussianKernel(rows, rows * sigma_scale)

    # Create 2D radial mask via outer product and normalise to [0, 1]
    mask = y_axis * x_axis.T
    mask = mask / mask.max()

    # Make mask 3D to match image channels (BGR)
    if image.ndim == 3:
        mask = mask[..., None]

    # Apply mask and keep values within the valid pixel range [0, 255]
    result = image.astype(np.float32) * mask
    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    if len(sys.argv) != 2:
        print("Usage: python vignette.py imagename.jpg", file=sys.stderr)
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
    dst = out_dir / f"{src.stem}_vignette.png"
    cv2.imwrite(str(dst), vignette_filter(img))
    print(f"Saved {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())