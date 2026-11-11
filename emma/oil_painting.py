"""Oil Painting: painterly brush strokes with an embossed paint texture.

Uses a hand-built Gaussian-weighted Kuwahara filter, a colour boost and a
Sobel-shaded paint texture. Developed in emma_filters.ipynb.

Usage:
    python oil_painting.py imagename.jpg

The result is saved in images/output as imagename_oil.png
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


def boost_colour(image, saturation=1.5, brightness=1.1):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= saturation
    hsv[..., 2] *= brightness
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def quadrant_kernels(radius, weighted):
    size = 2 * radius + 1
    if weighted:
        base = make_gaussian_kernel(size, radius / 2)
    else:
        base = np.ones((size, size), np.float32)

    # cut the kernel into four overlapping quarters
    kernels = []
    for rows, cols in [(slice(0, radius + 1), slice(0, radius + 1)),
                       (slice(0, radius + 1), slice(radius, size)),
                       (slice(radius, size), slice(0, radius + 1)),
                       (slice(radius, size), slice(radius, size))]:
        k = np.zeros((size, size), np.float32)
        k[rows, cols] = base[rows, cols]
        kernels.append(k / k.sum())
    return kernels


def kuwahara(image, radius=8, weighted=True):
    img = image.astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    best_var = np.full(gray.shape, np.inf, np.float32)
    out = np.zeros_like(img)

    for k in quadrant_kernels(radius, weighted):
        mean = cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)
        m = cv2.filter2D(gray, -1, k, borderType=cv2.BORDER_REFLECT)
        m2 = cv2.filter2D(gray * gray, -1, k, borderType=cv2.BORDER_REFLECT)
        var = m2 - m * m  # Var = E[x^2] - (E[x])^2

        # keep the colour from whichever region is most uniform
        better = var < best_var
        best_var[better] = var[better]
        out[better] = mean[better]

    return np.clip(out, 0, 255).astype(np.uint8)


def paint_texture(painted, strength=0.6, work_size=2000):
    h, w = painted.shape[:2]
    scale = work_size / max(h, w)
    small = cv2.resize(painted, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255

    # streaky noise for bristle marks
    noise = np.random.default_rng(0).random(gray.shape).astype(np.float32)
    streaks = cv2.blur(noise, (1, 9))
    height = cv2.GaussianBlur(gray + 0.15 * streaks, (0, 0), 1.0)

    # slope of the paint, lit from the top-left
    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
    shade = np.clip(-(gx + gy) / np.sqrt(2) * strength, -0.35, 0.35)

    shade = cv2.resize(shade, (w, h), interpolation=cv2.INTER_LINEAR)
    out = painted.astype(np.float32) / 255 + shade[..., None]
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def oil_painting(image, brush=0.005, weighted=True, saturation=1.2, texture=0.7):
    # brush size as a fraction of the photo so every image gets the same strength
    radius = max(3, int(max(image.shape[:2]) * brush))
    painted = kuwahara(image, radius, weighted)
    painted = boost_colour(painted, saturation, brightness=1.0)
    if texture > 0:
        painted = paint_texture(painted, texture)
    return painted


def main():
    if len(sys.argv) != 2:
        print("Usage: python oil_painting.py imagename.jpg", file=sys.stderr)
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
    dst = out_dir / f"{src.stem}_oil.png"
    cv2.imwrite(str(dst), oil_painting(img))
    print(f"Saved {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
