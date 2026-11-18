"""Comic Book: flat colours and thick ink outlines, like a comic panel.

Uses bilateral smoothing, a saturation boost, k-means colour reduction and
adaptive-threshold ink lines. Developed in emma_filters.ipynb.

Usage:
    python comic_book.py imagename.jpg

The result is saved in images/output as imagename_comic.png
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def smooth_image(image, passes=3):
    # bilateral filter smooths flat areas but keeps edges sharp
    smooth = image.copy()
    for _ in range(passes):
        smooth = cv2.bilateralFilter(smooth, 9, 75, 75)
    return smooth


def boost_colour(image, saturation=1.5, brightness=1.1):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= saturation
    hsv[..., 2] *= brightness
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def reduce_colours(image, k=8, sample_size=50000):
    pixels = image.reshape(-1, 3).astype(np.float32)

    # pick the palette from a random sample of pixels
    idx = np.random.choice(len(pixels), min(sample_size, len(pixels)), replace=False)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centres = cv2.kmeans(pixels[idx], k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    # give every pixel the closest palette colour
    dist = -2 * pixels @ centres.T + (centres ** 2).sum(axis=1)
    labels = dist.argmin(axis=1)
    return centres.astype(np.uint8)[labels].reshape(image.shape)


def comic_book(image, k=8, block_size=9, c=5, thickness=2, saturation=1.3):
    smooth = smooth_image(image)
    colours = reduce_colours(boost_colour(smooth, saturation), k)

    # ink lines - pixels darker than their neighbourhood become black
    gray = cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 7)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, block_size, c)
    if thickness > 1:
        edges = cv2.erode(edges, np.ones((thickness, thickness), np.uint8))

    # lines are 0 in the mask so they come out black
    return cv2.bitwise_and(colours, colours, mask=edges)


def main():
    if len(sys.argv) != 2:
        print("Usage: python comic_book.py imagename.jpg", file=sys.stderr)
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
    dst = out_dir / f"{src.stem}_comic.png"
    cv2.imwrite(str(dst), comic_book(img))
    print(f"Saved {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
