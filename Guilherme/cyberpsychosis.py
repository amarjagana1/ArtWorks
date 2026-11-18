"""Cyberpsychosis: a rainbow glitch effect, like a character's mind breaking down.

Pipeline:
  1. Make many copies ("ghosts") of the photo, each tinted a rainbow colour (red -> violet).
  2. Spread the ghosts out sideways so they pile up on top of each other.
  3. Tear random horizontal bands of each ghost, so no two ghosts line up.
  4. Add the ghosts together: where they line up you get the real photo back,
     where they don't the edges split into rainbow bands.
  5. Put a sharp copy of the photo back on top so the subject stays readable.
  6. Boost contrast and colour, then add CRT scanlines and grain.

Usage:
    python cyberpsychosis.py imagename.jpg

The result is saved in the Output folder as imagename_cyberpsychosis.png

Requires: numpy, pillow   (pip install numpy pillow)
"""
import colorsys
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------- helpers
def to_numbers(img):
    """Picture -> float array from 0.0 (black) to 1.0 (bright)."""
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def to_picture(arr):
    """Float array -> picture."""
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def rainbow(count):
    """`count` rainbow colours from red to violet, as RGB 0-1."""
    return [np.array(colorsys.hsv_to_rgb(0.85 * i / (count - 1), 1, 1), dtype=np.float32)
            for i in range(count)]


# ---------------------------------------------------------------- effects
def tear(arr, rng, strength=0.8):
    """Slide random horizontal bands left or right, like a broken video signal."""
    h, w, _ = arr.shape
    arr = arr.copy()
    for _ in range(int(4 + 14 * strength)):
        y0 = int(rng.integers(0, h))
        band = int(rng.integers(2, max(3, h // 18)))
        shift = int(rng.normal(0, w * 0.06 * strength))
        arr[y0:y0 + band] = np.roll(arr[y0:y0 + band], shift, axis=1)
    return arr


def rainbow_ghosts(arr, rng, layers=14, spread=0.09, jitter=0.012, tear_strength=0.8):
    """Stack rainbow-tinted, shifted, torn copies of the photo on top of each other.

    Each ghost only passes its own rainbow colour. Adding them all up and dividing by the
    total colour gives the real photo back where the ghosts line up, and splits the edges
    into rainbow bands where they don't.

    layers   how many ghosts
    spread   how far the stack reaches, as a fraction of the image width
    jitter   random up/down wobble per ghost, as a fraction of the height
    """
    h, w, _ = arr.shape
    total = np.zeros_like(arr)
    weight = np.zeros(3, dtype=np.float32)
    for i, colour in enumerate(rainbow(layers)):
        t = i / (layers - 1)
        dx = int(round((t - 0.5) * spread * w))               # centred: left ghosts red, right violet
        dy = int(round(rng.normal(0, jitter * h)))
        layer = tear(np.roll(arr, (dy, dx), axis=(0, 1)), rng, tear_strength)
        total += layer * colour
        weight += colour
    return total / weight


def punch(arr, contrast=1.25, saturation=1.5):
    """Boost contrast and colour so the rainbow bands look neon instead of faded."""
    grey = (arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
    arr = grey + (arr - grey) * saturation
    return (arr - 0.5) * contrast + 0.5


def sharpen_subject(arr, original, keep=0.25):
    """Mix the clean photo back in so the subject is still readable."""
    return arr * (1 - keep) + original * keep


def scanlines(arr, strength=0.22):
    """Darken every third row, like an old CRT screen."""
    rows = (np.arange(arr.shape[0]) % 3 == 0).astype(np.float32)
    return arr * (1 - strength * rows)[:, None, None]


def film_grain(arr, rng, strength=0.03):
    """Very fine random speckle."""
    noise = rng.normal(0, 1, arr.shape[:2]).astype(np.float32)
    return arr + (noise * strength)[..., None]


# ---------------------------------------------------------------- pipeline
SEED = 3  # same seed = same glitch every run (change it for a different tear pattern)


def cyberpsychosis(img, seed=SEED):
    """Runs every effect in order. Returns a new picture the same size as the input."""
    rng = np.random.default_rng(seed)
    original = to_numbers(img)
    arr = rainbow_ghosts(original, rng, layers=14, spread=0.09, jitter=0.012, tear_strength=0.8)
    arr = sharpen_subject(arr, original, 0.2)
    arr = punch(arr, contrast=1.25, saturation=1.5)
    arr = scanlines(arr, 0.22)
    arr = film_grain(arr, rng, 0.03)
    return to_picture(arr)


def main():
    if len(sys.argv) != 2:
        print("Usage: python cyberpsychosis.py imagename.jpg", file=sys.stderr)
        return 1
    here = Path(__file__).resolve().parent
    src = Path(sys.argv[1])
    if not src.exists() and (here / "Images" / src.name).exists():
        src = here / "Images" / src.name          # fall back to the Images folder
    try:
        img = Image.open(src).convert("RGB")
    except Exception as e:
        print(f"Error: can't open {src}: {e}", file=sys.stderr)
        return 1
    out_dir = here / "Output"
    out_dir.mkdir(exist_ok=True)
    dst = out_dir / f"{src.stem}_cyberpsychosis.png"
    cyberpsychosis(img).save(dst)
    print(f"Saved {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
