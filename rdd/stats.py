"""Dataset statistics and report figures."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .constants import CLASS_NAMES
from .util import country_of

SPLITS = ("train", "val", "test")


def dataset_stats(yolo_dir: Path) -> dict:
    """Count images and boxes per split, class and country from the YOLO label files."""
    out: dict = {"splits": {}, "per_country": defaultdict(lambda: defaultdict(int))}
    for split in SPLITS:
        img_dir = yolo_dir / split / "images"
        lbl_dir = yolo_dir / split / "labels"
        if not img_dir.exists():
            continue
        objects: Counter = Counter()
        with_damage = 0
        images = [p for p in img_dir.iterdir() if p.is_file()]
        for img in images:
            out["per_country"][country_of(img.name)][split] += 1
            label = lbl_dir / f"{img.stem}.txt"
            if not label.exists():
                continue
            rows = [ln.split() for ln in label.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if rows:
                with_damage += 1
            for row in rows:
                objects[CLASS_NAMES[int(row[0])]] += 1
        out["splits"][split] = {
            "images": len(images),
            "images_with_damage": with_damage,
            "objects": {name: objects.get(name, 0) for name in CLASS_NAMES},
        }
    out["per_country"] = {c: dict(v) for c, v in sorted(out["per_country"].items())}
    return out


def print_stats(stats: dict) -> None:
    for split, s in stats["splits"].items():
        total = sum(s["objects"].values())
        print(f"{split:5s}: {s['images']:6d} images ({s['images_with_damage']} with damage), {total} boxes")
        for name, n in s["objects"].items():
            print(f"        {name:20s} {n}")
    print("Images per country:")
    for country, per in stats["per_country"].items():
        print(f"  {country:16s} " + "  ".join(f"{k}={v}" for k, v in per.items()))


def plot_stats(stats: dict, reports_dir: Path) -> list[Path]:
    """Write two PNG figures (boxes per class, images per country) and a JSON dump."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports_dir.mkdir(parents=True, exist_ok=True)
    palette = {"train": "#2a6f97", "val": "#e09f3e", "test": "#9e2a2b"}
    splits = list(stats["splits"])
    written: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.8 / max(len(splits), 1)
    for i, split in enumerate(splits):
        vals = [stats["splits"][split]["objects"][n] for n in CLASS_NAMES]
        xs = [k + i * width for k in range(len(CLASS_NAMES))]
        ax.bar(xs, vals, width=width, label=split, color=palette.get(split))
    ax.set_xticks([k + width * (len(splits) - 1) / 2 for k in range(len(CLASS_NAMES))])
    ax.set_xticklabels([n.replace("_", "\n") for n in CLASS_NAMES])
    ax.set_ylabel("Annotated boxes")
    ax.set_title("Boxes per class and split")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    p1 = reports_dir / "class_distribution.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    written.append(p1)

    countries = list(stats["per_country"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.8 / max(len(splits), 1)
    for i, split in enumerate(splits):
        vals = [stats["per_country"][c].get(split, 0) for c in countries]
        xs = [k + i * width for k in range(len(countries))]
        ax.bar(xs, vals, width=width, label=split, color=palette.get(split))
    ax.set_xticks([k + width * (len(splits) - 1) / 2 for k in range(len(countries))])
    ax.set_xticklabels(countries)
    ax.set_ylabel("Images")
    ax.set_title("Images per country and split")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    p2 = reports_dir / "country_distribution.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    written.append(p2)

    p3 = reports_dir / "dataset_stats.json"
    p3.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    written.append(p3)
    return written
