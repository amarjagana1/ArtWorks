"""

Usage:
    python vintage_filter.py imagename.jpg

"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# ---------------------------------------------------------------- helpers
def to_numbers(img):
    """Picture -> float array from 0.0 (black) to 1.0 (bright)."""
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def to_picture(arr):
    """Float array -> picture."""
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def screen(base, light):
    """'Screen' blend: only ever brightens. Used for glows and light leaks."""
    return 1.0 - (1.0 - base) * (1.0 - light)


def smoothstep(edge0, edge1, x):
    """Smooth 0 -> 1 ramp between edge0 and edge1."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0, 1)
    return t * t * (3 - 2 * t)


def blur(arr, radius):
    """Gaussian blur of a float array (used for glows)."""
    return to_numbers(to_picture(arr).filter(ImageFilter.GaussianBlur(radius)))


def soft_noise(h, w, scale, rng):
    """Smooth random blobs. Bigger `scale` = bigger blobs."""
    small = rng.random((max(2, h // scale + 2), max(2, w // scale + 2))).astype(np.float32)
    big = np.asarray(Image.fromarray(small).resize((w, h), Image.BICUBIC))
    return (big - big.mean()) / (big.std() + 1e-6)


def edge_distance(h, w):
    """Distance of each pixel from the nearest edge, as a fraction of the short side."""
    y, x = np.mgrid[:h, :w].astype(np.float32)
    return np.minimum(np.minimum(x, w - 1 - x), np.minimum(y, h - 1 - y)) / min(h, w)


# ---------------------------------------------------------------- effects
def sepia_tone(arr, color_keep=0.12):
    """Brightness -> dark-brown-to-cream gradient, keeping 12% of the original color."""
    brightness = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    brightness = np.clip((brightness - 0.5) * 1.12 + 0.5, 0, 1)
    positions = [0.0, 0.35, 0.70, 1.0]
    colors = np.array([[0.09, 0.05, 0.03],
                       [0.34, 0.23, 0.14],
                       [0.74, 0.60, 0.42],
                       [0.98, 0.91, 0.77]])
    toned = np.stack([np.interp(brightness, positions, colors[:, c]) for c in range(3)], -1)
    toned = toned.astype(np.float32)
    return toned * (1 - color_keep) + arr * color_keep, brightness


def glow(arr, brightness, strength=0.4):
    """Blur only the bright parts, tint warm, screen back on top (original stays sharp)."""
    S = min(arr.shape[:2])
    bright_only = arr * smoothstep(0.45, 0.95, brightness)[..., None]
    halo = blur(bright_only, S * 0.012) + 0.8 * blur(bright_only, S * 0.04)
    warm = np.array([1.0, 0.82, 0.62], dtype=np.float32)
    return screen(arr, np.clip(halo * warm * strength, 0, 1))


def light_leaks(arr, rng, intensity=0.7, count=2):
    """Orange / pink / purple light bleeding in from the edges (screen blend)."""
    h, w, _ = arr.shape
    y, x = np.mgrid[:h, :w].astype(np.float32)
    palette = [(1.0, 0.50, 0.12), (1.0, 0.28, 0.35), (0.75, 0.35, 0.85), (1.0, 0.72, 0.30)]
    layer = np.zeros_like(arr)

    def blob(cx, cy, radius, stretch, color):
        d = np.sqrt(((x - cx) / stretch) ** 2 + (y - cy) ** 2) / radius
        return np.exp(-(d ** 2) * 2.5)[..., None] * np.array(color, dtype=np.float32)

    for _ in range(count):
        side = rng.integers(0, 4)
        if side == 0:   cx, cy = rng.uniform(0, w), -0.03 * h
        elif side == 1: cx, cy = rng.uniform(0, w), 1.03 * h
        elif side == 2: cx, cy = -0.03 * w, rng.uniform(0, h)
        else:           cx, cy = 1.03 * w, rng.uniform(0, h)
        r = rng.uniform(0.25, 0.45) * max(h, w)
        stretch = rng.uniform(0.6, 1.4)
        layer += blob(cx, cy, r, stretch, palette[rng.integers(0, 2)])
        ang = rng.uniform(0, 2 * np.pi)
        layer += 0.8 * blob(cx + np.cos(ang) * r * 0.5, cy + np.sin(ang) * r * 0.5,
                            r * 0.6, stretch, palette[rng.integers(1, 4)])
    return screen(arr, np.clip(layer * intensity, 0, 1))


def edge_burn(arr, rng, strength=0.5):
    """Edges darken toward reddish brown."""
    h, w, _ = arr.shape
    S = min(h, w)
    d = np.clip(edge_distance(h, w) + soft_noise(h, w, max(8, S // 6), rng) * 0.02, 0, None)
    burn = (np.clip(1 - d / 0.14, 0, 1) ** 2 * strength)[..., None]
    tint = np.array([0.45, 0.18, 0.10], dtype=np.float32)
    return arr * (1 - burn * (1 - tint))


def stains(arr, rng, strength=0.5):
    """Brown blotches of age, mostly near the edges."""
    h, w, _ = arr.shape
    S = min(h, w)
    n = soft_noise(h, w, max(2, S // 3), rng) + 0.5 * soft_noise(h, w, max(2, S // 10), rng)
    blotch = smoothstep(0.9, 2.2, n)
    near_edge = 0.4 + 0.6 * smoothstep(0.25, 0.0, edge_distance(h, w))
    amount = (blotch * near_edge * strength)[..., None]
    return arr * (1 - amount * np.array([0.15, 0.28, 0.45], dtype=np.float32))


def damage(arr, rng, creases=3, scratches=8, dust=1.0):
    """Folds, scratches and dust specks, drawn as light/dark marks then blended in."""
    h, w, _ = arr.shape
    S = min(h, w)
    light = Image.new("L", (w, h), 0)
    dark = Image.new("L", (w, h), 0)
    dl, dd = ImageDraw.Draw(light), ImageDraw.Draw(dark)
    lw = max(1, int(S / 500))

    def wobbly(p0, p1, steps=24, jitter=2.5):
        t = np.linspace(0, 1, steps)
        return [(p0[0] + (p1[0] - p0[0]) * a + rng.normal(0, jitter),
                 p0[1] + (p1[1] - p0[1]) * a + rng.normal(0, jitter)) for a in t]

    for i in range(creases):
        if i == 0:
            if rng.random() < 0.5:
                p0, p1 = (rng.uniform(.2, .8) * w, 0), (rng.uniform(.2, .8) * w, h)
            else:
                p0, p1 = (0, rng.uniform(.2, .8) * h), (w, rng.uniform(.2, .8) * h)
        else:
            cx, cy = rng.choice([0, w]), rng.choice([0, h])
            sx, sy = (1 if cx == 0 else -1), (1 if cy == 0 else -1)
            p0 = (cx + sx * rng.uniform(.04, .18) * w, cy)
            p1 = (cx, cy + sy * rng.uniform(.04, .18) * h)
        pts = wobbly(p0, p1)
        dl.line(pts, fill=int(rng.uniform(140, 230)), width=lw)
        dd.line([(px + lw + 1, py + lw) for px, py in pts],
                fill=int(rng.uniform(80, 140)), width=lw + 1)

    for _ in range(scratches):
        x0, length = rng.uniform(0, w), rng.uniform(0.1, 0.45) * h
        y0 = rng.uniform(0, h - length)
        pts = [(x0 + np.cumsum(rng.normal(0, .4, 30))[k], y0 + length * k / 29) for k in range(30)]
        dl.line(pts, fill=int(rng.uniform(90, 170)), width=1)

    for _ in range(int(dust * w * h / 8000)):
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(.5, 2.5) * S / 700
        target = dl if rng.random() < 0.5 else dd
        target.ellipse([cx - r, cy - r, cx + r, cy + r], fill=int(rng.uniform(120, 255)))

    l = np.asarray(light.filter(ImageFilter.GaussianBlur(0.6)), dtype=np.float32) / 255
    d = np.asarray(dark.filter(ImageFilter.GaussianBlur(0.6)), dtype=np.float32) / 255
    arr = screen(arr, (l * 0.55)[..., None] * np.array([1.0, 0.95, 0.85], dtype=np.float32))
    return arr * (1 - (d * 0.45)[..., None])


def film_grain(arr, rng, strength=0.025):
    """Very fine random speckle."""
    noise = rng.normal(0, 1, arr.shape[:2]).astype(np.float32)
    return arr + (noise * strength)[..., None]


def paper_border(arr, rng, border=0.014, fade=0.05, corner=0.03, background=(0.97, 0.96, 0.94)):
    """Cream paper frame with rounded, slightly ragged corners and fibre texture."""
    h, w, _ = arr.shape
    S = min(h, w)
    y, x = np.mgrid[:h, :w].astype(np.float32)

    r, inset = S * corner, S * 0.003
    qx = np.abs(x - (w - 1) / 2) - ((w - 1) / 2 - inset - r)
    qy = np.abs(y - (h - 1) / 2) - ((h - 1) / 2 - inset - r)
    dist = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0) - r
    dist += soft_noise(h, w, max(4, S // 40), rng) * S * 0.0007
    dist += soft_noise(h, w, max(8, S // 8), rng) * S * 0.002
    depth = -dist
    inside = np.clip(depth + 0.5, 0, 1)[..., None]

    fibers = soft_noise(h, w, 2, rng) * 0.02 + soft_noise(h, w, max(4, S // 4), rng) * 0.03
    paper = np.array([0.94, 0.88, 0.75], dtype=np.float32) * (1 + fibers[..., None])
    arr = arr * (1 + fibers[..., None] * 0.6)

    solid_border = smoothstep(S * border, S * border * 0.4, depth)
    soft_fade = smoothstep(S * fade, 0, depth) * 0.35
    mix = np.maximum(solid_border * 0.95, soft_fade)[..., None]
    arr = arr * (1 - mix) + paper * mix

    return arr * inside + np.array(background, dtype=np.float32) * (1 - inside)


def add_caption(img, text):
    """Caption in the bottom-right corner."""
    w, h = img.size
    S = min(w, h)
    try:
        font = ImageFont.load_default(size=int(S * 0.035))
    except Exception:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(img)
    box = d.textbbox((0, 0), text, font=font)
    d.text((w - (box[2] - box[0]) - S * 0.05, h - (box[3] - box[1]) - S * 0.06),
           text, font=font, fill=(238, 224, 196))
    return img


# ---------------------------------------------------------------- pipeline
SEED = 3  # same seed = same damage every run


def vintage(img, seed=SEED):
    """Runs every effect in order. Returns a new picture the same size as the input."""
    rng = np.random.default_rng(seed)
    arr = to_numbers(img)
    arr, brightness = sepia_tone(arr)
    arr = glow(arr, brightness, 0.4)
    arr = light_leaks(arr, rng, 0.7)
    arr = edge_burn(arr, rng, 0.5)
    arr = stains(arr, rng, 0.5)
    arr = damage(arr, rng, creases=3, scratches=8, dust=1.0)
    arr = film_grain(arr, rng, 0.025)
    arr = paper_border(arr, rng)
    return to_picture(arr)


def main():
    if len(sys.argv) != 2:
        print("Usage: python vintage_filter.py imagename.jpg", file=sys.stderr)
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
    dst = out_dir / f"{src.stem}_vintage.png"
    vintage(img).save(dst)
    print(f"Saved {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
