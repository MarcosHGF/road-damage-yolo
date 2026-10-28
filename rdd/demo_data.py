"""Synthetic road-damage images in the exact RDD2022 folder/XML layout.

This is NOT the real dataset. It exists so the whole pipeline (convert ->
split -> train -> evaluate -> predict) can be exercised in a minute, offline,
and in CI, without downloading gigabytes. The shapes are simple on purpose
(dark lines, blobs) but are learnable, so a short training run shows a
non-trivial loss curve.

Layout produced (identical to the real archives):

    <raw>/<Country>/train/images/<Country>_000001.jpg
    <raw>/<Country>/train/annotations/xmls/<Country>_000001.xml
    <raw>/<Country>/test/images/<Country>_100001.jpg      (no annotations)
"""
from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# (size, base gray level, noise sigma) -- different per "country" to mimic
# the camera/lighting differences between the real subsets.
STYLES = {
    "Japan": (600, 125, 8),
    "Czech": (600, 95, 10),
    "India": (720, 138, 14),
    "Norway": (640, 110, 9),
}
DEFAULT_COUNTRIES = ("Japan", "Czech", "India")
SYNTHETIC_MARKER = "SYNTHETIC_DEMO_DATA.txt"


def _background(size: int, base: int, sigma: int, rng: np.random.Generator) -> Image.Image:
    coarse = rng.normal(0, 10, (8, 8)).astype(np.float32)
    coarse_img = Image.fromarray(np.clip(coarse + 128, 0, 255).astype(np.uint8)).resize(
        (size, size), Image.BICUBIC
    )
    low = (np.asarray(coarse_img, dtype=np.float32) - 128) * 0.8
    noise = rng.normal(0, sigma, (size, size)).astype(np.float32)
    gray = np.clip(base + low + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([gray] * 3, axis=-1), "RGB")


def _walk(draw: ImageDraw.ImageDraw, pts: list[tuple[float, float]], width: int, color: int) -> None:
    draw.line(pts, fill=(color, color, color), width=width, joint="curve")


def _line_crack(size: int, vertical: bool, rng: random.Random, draw: ImageDraw.ImageDraw):
    length = rng.uniform(0.3, 0.6) * size
    a = rng.uniform(0.15, 0.85) * size  # position across the line direction
    b = rng.uniform(0.05, 0.35) * size  # start along the line direction
    steps = 8
    pts = []
    for i in range(steps + 1):
        along = b + length * i / steps
        across = a + rng.uniform(-6, 6) * (1 if i else 0) + (i * rng.uniform(-3, 3))
        pts.append((across, along) if vertical else (along, across))
    width = rng.randint(2, 4)
    _walk(draw, pts, width, rng.randint(15, 50))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pad = width + 2
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def _alligator(size: int, rng: random.Random, draw: ImageDraw.ImageDraw):
    side = rng.randint(90, 150)
    x0 = rng.uniform(0.05, 0.95) * (size - side)
    y0 = rng.uniform(0.2, 0.95) * (size - side)
    for _ in range(rng.randint(9, 15)):
        sx = x0 + rng.uniform(0, side)
        sy = y0 + rng.uniform(0, side)
        ex = min(max(sx + rng.uniform(-35, 35), x0), x0 + side)
        ey = min(max(sy + rng.uniform(-35, 35), y0), y0 + side)
        _walk(draw, [(sx, sy), (ex, ey)], rng.randint(1, 3), rng.randint(20, 55))
    return x0, y0, x0 + side, y0 + side


def _pothole(size: int, rng: random.Random, draw: ImageDraw.ImageDraw):
    rx = rng.randint(18, 45)
    ry = rng.randint(14, 38)
    cx = rng.uniform(rx + 5, size - rx - 5)
    cy = rng.uniform(0.35 * size, size - ry - 5)
    dark = rng.randint(20, 55)
    draw.ellipse([cx - rx - 4, cy - ry - 4, cx + rx + 4, cy + ry + 4], fill=(dark + 70,) * 3)
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(dark,) * 3)
    return cx - rx - 4, cy - ry - 4, cx + rx + 4, cy + ry + 4


def _distractor(size: int, rng: random.Random, draw: ImageDraw.ImageDraw):
    # Painted-over repair patch: labelled D44, which the converter ignores.
    w = rng.randint(40, 90)
    h = rng.randint(40, 90)
    x0 = rng.uniform(0, size - w)
    y0 = rng.uniform(0.3 * size, size - h)
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=(55, 55, 58))
    return x0, y0, x0 + w, y0 + h


MAKERS = {
    "D00": lambda s, r, d: _line_crack(s, True, r, d),
    "D10": lambda s, r, d: _line_crack(s, False, r, d),
    "D20": _alligator,
    "D40": _pothole,
    "D44": _distractor,
}
CODE_WEIGHTS = {"D00": 3, "D10": 2, "D20": 2, "D40": 2, "D44": 1}


def _write_xml(path: Path, filename: str, size: int, objects: list[tuple[str, tuple[int, int, int, int]]]) -> None:
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "images"
    ET.SubElement(root, "filename").text = filename
    sz = ET.SubElement(root, "size")
    ET.SubElement(sz, "width").text = str(size)
    ET.SubElement(sz, "height").text = str(size)
    ET.SubElement(sz, "depth").text = "3"
    for code, (x1, y1, x2, y2) in objects:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = code
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bb = ET.SubElement(obj, "bndbox")
        ET.SubElement(bb, "xmin").text = str(x1)
        ET.SubElement(bb, "ymin").text = str(y1)
        ET.SubElement(bb, "xmax").text = str(x2)
        ET.SubElement(bb, "ymax").text = str(y2)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def generate_demo_data(
    raw_dir: Path,
    n_per_country: int = 40,
    countries: tuple[str, ...] = DEFAULT_COUNTRIES,
    seed: int = 7,
    n_test_images: int = 4,
) -> dict:
    """Write synthetic data under ``raw_dir``; return simple counts."""
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    codes = list(CODE_WEIGHTS)
    weights = [CODE_WEIGHTS[c] for c in codes]
    summary = {"images": 0, "objects": 0, "countries": list(countries)}
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Marker read by `convert` so synthetic data can never be mistaken for RDD2022.
    (raw_dir / SYNTHETIC_MARKER).write_text(
        "Synthetic demo data generated by `python -m rdd demo-data`. Not RDD2022.\n", encoding="utf-8"
    )

    for country in countries:
        size, base, sigma = STYLES.get(country, (600, 120, 10))
        img_dir = raw_dir / country / "train" / "images"
        xml_dir = raw_dir / country / "train" / "annotations" / "xmls"
        test_dir = raw_dir / country / "test" / "images"
        for d in (img_dir, xml_dir, test_dir):
            d.mkdir(parents=True, exist_ok=True)

        for i in range(1, n_per_country + 1):
            name = f"{country}_{i:06d}"
            img = _background(size, base, sigma, np_rng)
            draw = ImageDraw.Draw(img)
            n_obj = py_rng.choices([0, 1, 2, 3], weights=[2, 4, 3, 1.5])[0]
            objects = []
            for _ in range(n_obj):
                code = py_rng.choices(codes, weights=weights)[0]
                x1, y1, x2, y2 = MAKERS[code](size, py_rng, draw)
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(size, int(x2)), min(size, int(y2))
                if x2 - x1 >= 4 and y2 - y1 >= 4:
                    objects.append((code, (x1, y1, x2, y2)))
            img.save(img_dir / f"{name}.jpg", quality=92)
            # Like the real data: some damage-free images have an empty XML,
            # others have no XML at all.
            if objects or i % 2 == 0:
                _write_xml(xml_dir / f"{name}.xml", f"{name}.jpg", size, objects)
            summary["images"] += 1
            summary["objects"] += len(objects)

        for j in range(1, n_test_images + 1):
            name = f"{country}_{100000 + j:06d}"
            _background(size, base, sigma, np_rng).save(test_dir / f"{name}.jpg", quality=92)

    return summary
