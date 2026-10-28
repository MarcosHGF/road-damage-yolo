"""Train/val/test split, stratified by country and by damage vs damage-free."""
from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

from .constants import CLASS_NAMES
from .util import country_of, image_number, place


def _allocate(groups: list[list[Path]], ratios: tuple[float, float, float], rng: random.Random):
    """Assign whole groups to train/val/test so image counts follow ``ratios``."""
    rng.shuffle(groups)
    total = sum(len(g) for g in groups)
    targets = [total * r for r in ratios]
    buckets: list[list[Path]] = [[], [], []]
    counts = [0, 0, 0]
    for g in groups:
        # Put the group where it leaves the biggest shortfall relative to target.
        deficits = [targets[i] - counts[i] for i in range(3)]
        i = max(range(3), key=lambda k: deficits[k])
        buckets[i].extend(g)
        counts[i] += len(g)
    return buckets


def _make_groups(images: list[Path], group_size: int) -> list[list[Path]]:
    if group_size <= 1:
        return [[p] for p in images]
    by_block: dict[int, list[Path]] = defaultdict(list)
    loose: list[list[Path]] = []
    for p in images:
        n = image_number(p.name)
        if n is None:
            loose.append([p])
        else:
            by_block[n // group_size].append(p)
    return list(by_block.values()) + loose


def split_dataset(
    converted_dir: Path,
    out_dir: Path,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    group_size: int = 1,
    copy_files: bool = False,
) -> dict:
    """Create ``out_dir/{train,val,test}/{images,labels}`` plus ``data.yaml``."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must add up to 1.0, got {ratios}")
    images_dir = converted_dir / "images"
    labels_dir = converted_dir / "labels"
    if not images_dir.exists():
        raise FileNotFoundError(f"{images_dir} does not exist. Run `python -m rdd convert` first.")

    all_images = sorted(p for p in images_dir.iterdir() if p.is_file())
    if not all_images:
        raise FileNotFoundError(f"No images in {images_dir}.")

    # Stratum = (country, has damage). Each stratum is split on its own so every
    # split sees every country and a similar share of damage-free images.
    strata: dict[tuple[str, bool], list[Path]] = defaultdict(list)
    for p in all_images:
        label = labels_dir / f"{p.stem}.txt"
        has_damage = label.exists() and label.stat().st_size > 0
        strata[(country_of(p.name), has_damage)].append(p)

    rng = random.Random(seed)
    names = ("train", "val", "test")
    assigned: dict[str, list[Path]] = {n: [] for n in names}
    for key in sorted(strata):
        groups = _make_groups(strata[key], group_size)
        for name, bucket in zip(names, _allocate(groups, ratios, rng)):
            assigned[name].extend(bucket)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for name in names:
        (out_dir / name / "images").mkdir(parents=True)
        (out_dir / name / "labels").mkdir(parents=True)
        for img in assigned[name]:
            place(img, out_dir / name / "images" / img.name, copy=copy_files)
            label = labels_dir / f"{img.stem}.txt"
            dst = out_dir / name / "labels" / f"{img.stem}.txt"
            if label.exists():
                place(label, dst, copy=True)
            else:
                dst.write_text("", encoding="utf-8")

    data_yaml = {
        "path": out_dir.resolve().as_posix(),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    per_split_country: dict[str, dict[str, int]] = {}
    for name in names:
        counts: dict[str, int] = defaultdict(int)
        for img in assigned[name]:
            counts[country_of(img.name)] += 1
        per_split_country[name] = dict(sorted(counts.items()))
    summary = {
        "seed": seed,
        "ratios": list(ratios),
        "group_size": group_size,
        "images": {n: len(assigned[n]) for n in names},
        "per_country": per_split_country,
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
