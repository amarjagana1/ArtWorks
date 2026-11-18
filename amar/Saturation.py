"""Saturation: make the colours of a photo more vivid (or more muted).

Scales the S (saturation) channel in HSV colour space, leaving hue and
brightness untouched. Developed in emma_filters.ipynb.

Usage:
    python saturation.py imagename.jpg

The result is saved in images/output as imagename_saturation.png
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def saturation_hsv(image, factor=1.5):
    """
    Adjust saturation by scaling the S channel in HSV colour space.
    factor: 0.0 = grayscale, 1.0 = unchanged, >1.0 = more vivid colours.
    """
    if factor < 0:
        raise ValueError("factor must be >= 0")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("saturation needs a 3-channel colour (BGR) image")

    # Convert BGR -> HSV. In 8-bit OpenCV: H = 0-179, S = 0-255, V = 0-255
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Scale only the saturation channel and clip to the valid range.
    # Hue (colour) and Value (brightness) are left untouched.
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)

    # Convert back to BGR for saving
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def main():
    if len(sys.argv) != 2:
        print("Usage: python saturation.py imagename.jpg", file=sys.stderr)
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
    dst = out_dir / f"{src.stem}_saturation.png"
    cv2.imwrite(str(dst), saturation_hsv(img))
    print(f"Saved {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())