"""Dreamy Glow: makes the bright parts of a photo bloom outwards with a soft glow.

Uses a hand-built Gaussian kernel, highlight extraction, two blurs (small and
large) and a screen blend. Developed in emma_filters.ipynb.

Usage:
    python dreamy_glow.py imagename.jpg

The result is saved in images/output as imagename_glow.png
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def make_gaussian_kernel(size, sigma):
    # kernel size has to be odd so there's a centre pixel
    if size % 2 == 0:
        size += 1

    # distances from the centre e.g. -2 -1 0 1 2
    half = size // 2
    x = np.arange(-half, half + 1)
    xx, yy = np.meshgrid(x, x)

    # gaussian formula - values drop off the further from the centre you go
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    # divide by the total so it adds up to 1 and brightness stays the same
    return (kernel / kernel.sum()).astype(np.float32)


def dreamy_glow(image, sigma=20, strength=0.8, threshold=0.5, boost=1.5):
    # convert to 0-1 floats
    a = image.astype(np.float32) / 255

    # pick out just the bright parts
    # anything darker than the threshold gets ignored, brighter parts fade in
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    mask = np.clip((gray - threshold) / (1 - threshold), 0, 1)
    highlights = a * mask[:, :, np.newaxis]

    # two blurs - a tight glow around the light and a wide soft haze
    small_sigma = max(1, sigma // 3)
    small = cv2.filter2D(highlights, -1, make_gaussian_kernel(6 * small_sigma + 1, small_sigma), borderType=cv2.BORDER_REFLECT)
    big = cv2.filter2D(highlights, -1, make_gaussian_kernel(6 * sigma + 1, sigma), borderType=cv2.BORDER_REFLECT)

    # add the two together and boost so the glow is stronger
    glow_layer = np.clip((small + big) * boost, 0, 1)

    # screen blend the glow on top, strength controls how much
    result = 1 - (1 - a) * (1 - glow_layer * strength)

    return np.clip(result * 255, 0, 255).astype(np.uint8)


def main():
    if len(sys.argv) != 2:
        print("Usage: python dreamy_glow.py imagename.jpg", file=sys.stderr)
        return 1

    here = Path(__file__).resolve().parent
    src = Path(sys.argv[1])
    if not src.exists() and (here / "images" / "input" / src.name).exists():
        src = here / "images" / "input" / src.name  # fall back to the input folder

    img = cv2.imread(str(src))
    if img is None:  # imread fails silently so check it actually loaded
        print(f"Error: can't open {src}", file=sys.stderr)
        return 1

    out_dir = here / "images" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{src.stem}_glow.png"
    cv2.imwrite(str(dst), dreamy_glow(img))
    print(f"Saved {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
