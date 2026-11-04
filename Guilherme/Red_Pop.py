"""Red Pop: keep the RED parts of a photo in colour, make everything else black & white.

Usage:
    python Red_Pop.py imagename.jpg

The result is saved in the Output folder as imagename_redpop.png
"""
import sys
from pathlib import Path

import cv2 as cv      # OpenCV: reads and processes images
import numpy as np    # NumPy: maths on the pixel numbers


def red_mask(hsv):
    """Find the red pixels. Returns a mask: 1 = red, 0 = not red, in-between = edge."""

    # HSV describes colour as Hue (which colour), Saturation (how strong), Value (how bright).
    # Red sits at BOTH ends of the hue scale (0-179), so we need two ranges:
    low = cv.inRange(hsv, np.array([0, 100, 50]), np.array([10, 255, 255]))     # orange-ish red
    high = cv.inRange(hsv, np.array([160, 100, 50]), np.array([179, 255, 255])) # pink-ish red
    mask = cv.bitwise_or(low, high)  # a pixel counts as red if it is in EITHER range

    # Clean the mask:
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)   # remove tiny stray dots
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)  # fill tiny holes in red areas

    # Blur the mask a little so the colour fades smoothly into grey (no jagged edges)
    mask = cv.GaussianBlur(mask, (7, 7), 0)

    return mask.astype(np.float32) / 255.0  # change 0-255 into 0-1


def red_pop(img):
    """Make the red-pop picture."""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)        # same photo, but described in HSV
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)      # black & white version
    gray3 = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)    # give it 3 channels so it can mix with the colour photo

    alpha = red_mask(hsv)[..., None]  # reshape so it can multiply the 3 colour channels

    # Mix the two photos: where the mask is 1 use colour, where it is 0 use grey
    result = alpha * img.astype(np.float32) + (1 - alpha) * gray3.astype(np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    if len(sys.argv) != 2:
        print("Usage: python Red_Pop.py imagename.jpg", file=sys.stderr)
        return 1

    here = Path(__file__).resolve().parent
    src = Path(sys.argv[1])
    if not src.exists() and (here / "Images" / src.name).exists():
        src = here / "Images" / src.name          # fall back to the Images folder

    img = cv.imread(str(src))
    if img is None:  # file missing or not an image
        print(f"Error: can't open {src}", file=sys.stderr)
        return 1

    out_dir = here / "Output"
    out_dir.mkdir(exist_ok=True)  # create the Output folder if it is missing
    dst = out_dir / f"{src.stem}_redpop.png"
    cv.imwrite(str(dst), red_pop(img))
    print(f"Saved {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
