"""Red Pop: keep the RED parts of a photo in colour, make everything else black & white.

How to run:
    python Red_Pop.py                      -> does every image in the Images folder
    python Red_Pop.py Images/macaw.jpg     -> does just this image
    python Red_Pop.py --show Images/SM.jpg -> also shows the result on screen
Results are saved in the Output folder.
"""
import argparse
import sys
from pathlib import Path

import cv2 as cv      # OpenCV: reads and processes images
import numpy as np    # NumPy: maths on the pixel numbers

# Folder this script lives in, so it works no matter where you run it from
HERE = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def red_mask(hsv, s_min=100, v_min=50, feather=7):
    """Find the red pixels. Returns a mask: 1 = red, 0 = not red, in-between = edge."""

    # HSV describes colour as Hue (which colour), Saturation (how strong), Value (how bright).
    # Red sits at BOTH ends of the hue scale (0-179), so we need two ranges:
    low = cv.inRange(hsv, np.array([0, s_min, v_min]), np.array([10, 255, 255]))     # orange-ish red
    high = cv.inRange(hsv, np.array([160, s_min, v_min]), np.array([179, 255, 255])) # pink-ish red
    mask = cv.bitwise_or(low, high)  # a pixel counts as red if it is in EITHER range

    # Clean the mask:
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)   # remove tiny stray dots
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)  # fill tiny holes in red areas

    # Blur the mask a little so the colour fades smoothly into grey (no jagged edges)
    if feather > 1:
        k = feather | 1  # blur size must be an odd number
        mask = cv.GaussianBlur(mask, (k, k), 0)

    return mask.astype(np.float32) / 255.0  # change 0-255 into 0-1


def red_pop(img, **kwargs):
    """Make the red-pop picture. Returns (final picture, mask)."""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)        # same photo, but described in HSV
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)      # black & white version
    gray3 = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)    # give it 3 channels so it can mix with the colour photo

    mask = red_mask(hsv, **kwargs)
    alpha = mask[..., None]  # reshape so it can multiply the 3 colour channels

    # Mix the two photos: where the mask is 1 use colour, where it is 0 use grey
    result = alpha * img.astype(np.float32) + (1 - alpha) * gray3.astype(np.float32)
    return np.clip(result, 0, 255).astype(np.uint8), mask


def collect_images(paths):
    """Use the images given on the command line, or else every image in Images/."""
    if paths:
        return [Path(p) for p in paths]
    return sorted(p for p in (HERE / "Images").iterdir() if p.suffix.lower() in IMAGE_EXTS)


def main():
    # Command-line options (all optional)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", help="image files (default: all in Images/)")
    ap.add_argument("--show", action="store_true", help="display results in windows")
    ap.add_argument("--sat", type=int, default=100, help="minimum saturation: lower = accepts paler reds")
    ap.add_argument("--val", type=int, default=50, help="minimum brightness: lower = accepts darker reds")
    ap.add_argument("--feather", type=int, default=7, help="edge softness (0 = hard edge)")
    args = ap.parse_args()

    out_dir = HERE / "Output"
    out_dir.mkdir(exist_ok=True)  # create the Output folder if it is missing
    status = 0

    for path in collect_images(args.images):
        img = cv.imread(str(path))
        if img is None:  # file missing or not an image
            print(f"[skip] could not read {path}", file=sys.stderr)
            status = 1
            continue

        result, mask = red_pop(img, s_min=args.sat, v_min=args.val, feather=args.feather)
        mask_img = (mask * 255).astype(np.uint8)  # turn the mask back into a viewable picture

        # Save: final result, the mask, and original | mask | result side by side
        stem = path.stem
        cv.imwrite(str(out_dir / f"{stem}_redpop.png"), result)
        cv.imwrite(str(out_dir / f"{stem}_mask.png"), mask_img)
        comparison = np.hstack([img, cv.cvtColor(mask_img, cv.COLOR_GRAY2BGR), result])
        cv.imwrite(str(out_dir / f"{stem}_comparison.png"), comparison)
        print(f"[ok] {path.name}: {100 * float(mask.mean()):.1f}% of pixels kept in colour")

        if args.show:
            cv.namedWindow(stem, cv.WINDOW_NORMAL)  # resizable window
            cv.imshow(stem, comparison)

    if args.show:
        cv.waitKey(0)  # wait for a key press, then close the windows
        cv.destroyAllWindows()
    return status


if __name__ == "__main__":
    sys.exit(main())
