"""Carve a small, class- and country-balanced sample that fits in a GitHub repo.

GitHub rejects single files over 100 MB and asks repositories to stay well
under 1 GB, so the full RDD2022 subset (about 2 GB) cannot be committed. This
command copies a few hundred real images (from ``data/converted``) into
``data/sample`` so a fresh clone can train and evaluate immediately.
"""
from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from .constants import CLASS_NAMES
from .util import country_of, human_bytes

NOTICE = """\
# Sample data

A small subset of the **RDD2022** road damage dataset (Arya et al., 2022),
converted to YOLO format by this repository.

- Source: https://github.com/sekilab/RoadDamageDetector (CRDDC'2022)
- License: **CC BY-SA 4.0** (https://creativecommons.org/licenses/by-sa/4.0/).
  These images and labels are redistributed under the same license. Any
  derivative of them must credit the authors and use the same license.
- Citation: Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Sekimoto, Y.
  "RDD2022: A multi-national image dataset for automatic Road Damage
  Detection." arXiv:2209.08538, 2022.

Contents: {n_images} images, {n_boxes} boxes, {size}.

{table}

Use it with: `python -m rdd split --use-sample`
"""


def _read_classes(label: Path) -> list[int]:
    if not label.exists():
        return []
    return [int(ln.split()[0]) for ln in label.read_text(encoding="utf-8").splitlines() if ln.strip()]


def make_sample(
    converted_dir: Path,
    out_dir: Path,
    max_images: int = 300,
    max_mb: float = 60.0,
    negative_fraction: float = 0.10,
    seed: int = 42,
    allow_synthetic: bool = False,
) -> dict:
    images_dir = converted_dir / "images"
    labels_dir = converted_dir / "labels"
    if not images_dir.exists():
        raise FileNotFoundError(f"{images_dir} does not exist. Run `python -m rdd convert` first.")
    stats_file = converted_dir / "stats.json"
    if stats_file.exists() and json.loads(stats_file.read_text(encoding="utf-8")).get("synthetic") and not allow_synthetic:
        raise ValueError(
            "data/converted was built from synthetic demo data. A sample for the repository must come "
            "from the real RDD2022 download: run `python -m rdd download`, then `convert`, then `sample`."
        )

    rng = random.Random(seed)
    positives: list[tuple[Path, str, set[int]]] = []
    negatives: list[tuple[Path, str]] = []
    for img in sorted(p for p in images_dir.iterdir() if p.is_file()):
        classes = set(_read_classes(labels_dir / f"{img.stem}.txt"))
        c = country_of(img.name)
        (positives.append((img, c, classes)) if classes else negatives.append((img, c)))

    if not positives:
        raise ValueError("No images with damage found in the converted dataset.")

    n_neg = min(len(negatives), round(max_images * negative_fraction))
    n_pos = min(len(positives), max_images - n_neg)
    countries = sorted({c for _, c, _ in positives})

    chosen: list[Path] = []
    seen: set[Path] = set()

    def take(img: Path) -> bool:
        if img in seen or len(chosen) >= n_pos:
            return False
        seen.add(img)
        chosen.append(img)
        return True

    # Phase 1: guarantee every (country, class) cell is represented.
    per_cell = max(1, round(0.5 * n_pos / (len(countries) * len(CLASS_NAMES))))
    for country in countries:
        for cls in range(len(CLASS_NAMES)):
            pool = [img for img, c, cl in positives if c == country and cls in cl and img not in seen]
            rng.shuffle(pool)
            for img in pool[:per_cell]:
                take(img)
    # Phase 2: fill with random positives, spread across countries.
    rest = [img for img, _, _ in positives if img not in seen]
    rng.shuffle(rest)
    for img in rest:
        if not take(img):
            if len(chosen) >= n_pos:
                break
    # Phase 3: a few damage-free images, spread across countries.
    by_country_neg: dict[str, list[Path]] = defaultdict(list)
    for img, c in negatives:
        by_country_neg[c].append(img)
    neg_chosen: list[Path] = []
    for country in sorted(by_country_neg):
        pool = by_country_neg[country]
        k = max(0, round(n_neg * len(pool) / max(len(negatives), 1)))
        neg_chosen.extend(rng.sample(pool, min(k, len(pool))))
    ordered = chosen + neg_chosen[:n_neg]

    # Enforce the size budget (images dominate the size).
    budget = int(max_mb * 1024 * 1024)
    kept: list[Path] = []
    total = 0
    for img in ordered:
        size = img.stat().st_size
        if total + size > budget:
            continue
        kept.append(img)
        total += size

    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "labels").mkdir(parents=True)
    per_class = {n: 0 for n in CLASS_NAMES}
    per_country: dict[str, int] = defaultdict(int)
    for img in kept:
        shutil.copy2(img, out_dir / "images" / img.name)
        label = labels_dir / f"{img.stem}.txt"
        dst = out_dir / "labels" / f"{img.stem}.txt"
        if label.exists():
            shutil.copy2(label, dst)
        else:
            dst.write_text("", encoding="utf-8")
        for cls in _read_classes(label):
            per_class[CLASS_NAMES[cls]] += 1
        per_country[country_of(img.name)] += 1

    n_boxes = sum(per_class.values())
    lines = ["| Class | Boxes |", "|---|---|"] + [f"| {n} | {v} |" for n, v in per_class.items()]
    lines += ["", "| Country | Images |", "|---|---|"] + [f"| {c} | {v} |" for c, v in sorted(per_country.items())]
    (out_dir / "README.md").write_text(
        NOTICE.format(n_images=len(kept), n_boxes=n_boxes, size=human_bytes(total), table="\n".join(lines)),
        encoding="utf-8",
    )
    summary = {
        "images": len(kept),
        "boxes": n_boxes,
        "bytes": total,
        "per_class": per_class,
        "per_country": dict(per_country),
    }
    (out_dir / "sample_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
