"""Pascal VOC (XML) -> YOLO (txt) conversion for RDD2022."""
from __future__ import annotations

import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .constants import CLASS_MAP, CLASS_TO_ID, IMAGE_EXTS
from .util import country_of, place


@dataclass
class Pair:
    image: Path
    xml: Path | None  # None = image without annotation file


def _skip_name(name: str) -> bool:
    # macOS archives ship "__MACOSX/._file.jpg" resource-fork junk.
    return name.startswith("._") or name.startswith(".")


def index_images(raw_dir: Path) -> dict[str, list[Path]]:
    """Map file stem -> every image path with that stem (one directory walk)."""
    index: dict[str, list[Path]] = defaultdict(list)
    for root, dirs, files in os.walk(raw_dir):
        dirs[:] = [d for d in dirs if d != "__MACOSX" and not d.startswith(".")]
        for fname in files:
            if _skip_name(fname):
                continue
            p = Path(root) / fname
            if p.suffix.lower() in IMAGE_EXTS:
                index[p.stem].append(p)
    return index


def _closest(xml: Path, candidates: list[Path]) -> Path:
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda c: len(os.path.commonpath([str(xml), str(c)])))


def find_pairs(raw_dir: Path, include_unannotated_train: bool = True) -> tuple[list[Pair], int]:
    """Pair each XML with its image. Returns (pairs, number_of_unmatched_xml).

    The images are indexed once, so this stays linear in the number of files
    (a per-XML directory search is quadratic and unusable on 10k+ images).
    """
    images = index_images(raw_dir)
    pairs: list[Pair] = []
    used: set[Path] = set()
    unmatched_xml = 0
    for root, dirs, files in os.walk(raw_dir):
        dirs[:] = [d for d in dirs if d != "__MACOSX" and not d.startswith(".")]
        for fname in files:
            if _skip_name(fname) or not fname.lower().endswith(".xml"):
                continue
            xml = Path(root) / fname
            cands = images.get(xml.stem)
            if not cands:
                unmatched_xml += 1
                continue
            img = _closest(xml, cands)
            used.add(img)
            pairs.append(Pair(img, xml))

    if include_unannotated_train:
        for cands in images.values():
            for img in cands:
                if img in used:
                    continue
                if any(part.lower() == "train" for part in img.relative_to(raw_dir).parts[:-1]):
                    pairs.append(Pair(img, None))
    pairs.sort(key=lambda p: str(p.image))
    return pairs, unmatched_xml


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size  # (w, h)


def parse_voc_xml(xml_path: Path, image_path: Path | None = None):
    """Return ``(width, height, [(label, xmin, ymin, xmax, ymax), ...])``."""
    root = ET.parse(xml_path).getroot()
    w = h = 0
    size = root.find("size")
    if size is not None:
        try:
            w = int(float(size.findtext("width", "0")))
            h = int(float(size.findtext("height", "0")))
        except ValueError:
            w = h = 0
    if (w <= 0 or h <= 0) and image_path is not None:
        w, h = _image_size(image_path)
    objects = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            xmin = float(bnd.findtext("xmin"))
            ymin = float(bnd.findtext("ymin"))
            xmax = float(bnd.findtext("xmax"))
            ymax = float(bnd.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        objects.append((name, xmin, ymin, xmax, ymax))
    return w, h, objects


def voc_box_to_yolo(class_id: int, w: int, h: int, xmin: float, ymin: float, xmax: float, ymax: float) -> str | None:
    """One YOLO label line, or ``None`` when the box is empty after clamping."""
    x1, x2 = sorted((min(max(xmin, 0.0), w), min(max(xmax, 0.0), w)))
    y1, y2 = sorted((min(max(ymin, 0.0), h), min(max(ymax, 0.0), h)))
    if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
        return None
    xc = (x1 + x2) / 2 / w
    yc = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def convert_dataset(
    raw_dir: Path,
    out_dir: Path,
    keep_empty_ratio: float = 0.15,
    include_unannotated_train: bool = True,
    seed: int = 42,
) -> dict:
    """Convert everything under ``raw_dir`` into ``out_dir/{images,labels}``."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"{raw_dir} does not exist. Run `python -m rdd download` (or `demo-data`) first.")

    pairs, unmatched_xml = find_pairs(raw_dir, include_unannotated_train)
    if not pairs:
        raise FileNotFoundError(f"No images found under {raw_dir}. Check the extracted folder structure.")

    objects_per_class: Counter = Counter()
    objects_per_country_class: dict[str, Counter] = defaultdict(Counter)
    ignored: Counter = Counter()
    degenerate = 0
    bad_xml = 0

    positives: list[tuple[Pair, list[str]]] = []
    negatives: dict[str, list[Pair]] = defaultdict(list)

    for pair in pairs:
        country = country_of(pair.image.name)
        lines: list[str] = []
        if pair.xml is not None:
            try:
                w, h, objs = parse_voc_xml(pair.xml, pair.image)
            except (ET.ParseError, OSError):
                bad_xml += 1
                continue
            if w <= 0 or h <= 0:
                bad_xml += 1
                continue
            for name, xmin, ymin, xmax, ymax in objs:
                cls_name = CLASS_MAP.get(name)
                if cls_name is None:
                    ignored[name] += 1
                    continue
                line = voc_box_to_yolo(CLASS_TO_ID[cls_name], w, h, xmin, ymin, xmax, ymax)
                if line is None:
                    degenerate += 1
                    continue
                lines.append(line)
                objects_per_class[cls_name] += 1
                objects_per_country_class[country][cls_name] += 1
        if lines:
            positives.append((pair, lines))
        else:
            negatives[country].append(pair)

    # Negatives are sampled per country, proportional to that country's
    # positives, with a seeded RNG so every run picks the same images.
    rng = random.Random(seed)
    pos_per_country = Counter(country_of(p.image.name) for p, _ in positives)
    chosen_neg: list[Pair] = []
    for country, pool in sorted(negatives.items()):
        k = min(len(pool), round(pos_per_country.get(country, 0) * keep_empty_ratio))
        chosen_neg.extend(rng.sample(sorted(pool, key=lambda p: str(p.image)), k))

    images_out = out_dir / "images"
    labels_out = out_dir / "labels"
    for d in (images_out, labels_out):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    images_per_country: Counter = Counter()
    for pair, lines in positives:
        place(pair.image, images_out / pair.image.name)
        (labels_out / f"{pair.image.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        images_per_country[country_of(pair.image.name)] += 1
    for pair in chosen_neg:
        place(pair.image, images_out / pair.image.name)
        (labels_out / f"{pair.image.stem}.txt").write_text("", encoding="utf-8")
        images_per_country[country_of(pair.image.name)] += 1

    stats = {
        "synthetic": (raw_dir / "SYNTHETIC_DEMO_DATA.txt").exists(),
        "pairs_found": len(pairs),
        "unmatched_xml": unmatched_xml,
        "bad_xml": bad_xml,
        "images_with_damage": len(positives),
        "damage_free_available": sum(len(v) for v in negatives.values()),
        "damage_free_kept": len(chosen_neg),
        "images_written": len(positives) + len(chosen_neg),
        "images_per_country": dict(sorted(images_per_country.items())),
        "objects_per_class": dict(objects_per_class.most_common()),
        "objects_per_country_class": {c: dict(v) for c, v in sorted(objects_per_country_class.items())},
        "ignored_codes": dict(ignored.most_common()),
        "degenerate_boxes_dropped": degenerate,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def print_stats(stats: dict) -> None:
    print(f"Image/annotation pairs found : {stats['pairs_found']}")
    print(f"Images with damage           : {stats['images_with_damage']}")
    print(f"Damage-free available / kept : {stats['damage_free_available']} / {stats['damage_free_kept']}")
    print(f"Images written               : {stats['images_written']}")
    print("Objects per class:")
    for name, n in stats["objects_per_class"].items():
        print(f"  {name:20s} {n}")
    if stats["ignored_codes"]:
        print("Ignored damage codes (outside the 4 challenge classes):")
        for name, n in stats["ignored_codes"].items():
            print(f"  {name:20s} {n}")
    if stats["degenerate_boxes_dropped"]:
        print(f"Degenerate boxes dropped     : {stats['degenerate_boxes_dropped']}")
    if stats["unmatched_xml"]:
        print(f"XML files without an image   : {stats['unmatched_xml']}")
    if stats["bad_xml"]:
        print(f"Unreadable XML files skipped : {stats['bad_xml']}")
