"""Evaluation on the held-out split, overall and per country."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import yaml

from .constants import CLASS_NAMES
from .util import country_of, place


def _num(x) -> float | None:
    x = float(x)
    return None if math.isnan(x) else round(x, 4)


def summarize(metrics) -> dict:
    """Plain-dict summary of an Ultralytics ``DetMetrics`` object.

    Per-class arrays are ordered by ``ap_class_index`` (only the classes that
    actually occur in the evaluated images), NOT by class id, so they are
    mapped back to names explicitly.
    """
    box = metrics.box
    out = {
        "map50": _num(box.map50),
        "map50_95": _num(box.map),
        "precision": _num(box.mp),
        "recall": _num(box.mr),
        "per_class": {},
    }
    for k, cls in enumerate(box.ap_class_index):
        out["per_class"][CLASS_NAMES[int(cls)]] = {
            "ap50": _num(box.ap50[k]),
            "ap50_95": _num(box.ap[k]),
            "precision": _num(box.p[k]),
            "recall": _num(box.r[k]),
        }
    return out


def _country_subset(split_dir: Path, country: str, tmp: Path) -> tuple[Path, int] | None:
    images = [p for p in (split_dir / "images").iterdir() if p.is_file() and country_of(p.name) == country]
    if not images:
        return None
    root = tmp / country
    for img in images:
        place(img, root / "images" / img.name)
        label = split_dir / "labels" / f"{img.stem}.txt"
        dst = root / "labels" / f"{img.stem}.txt"
        if label.exists():
            place(label, dst, copy=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text("", encoding="utf-8")
    cfg = {
        "path": root.resolve().as_posix(),
        "train": "images",
        "val": "images",
        "test": "images",
        "nc": len(CLASS_NAMES),
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    yaml_path = tmp / f"{country}.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return yaml_path, len(images)


def evaluate_model(
    weights: Path,
    data_yaml: Path,
    runs_dir: Path,
    reports_dir: Path,
    *,
    split: str = "test",
    imgsz: int = 640,
    batch: int = 16,
    device=None,
    per_country: bool = True,
    workers: int = 4,
) -> dict:
    from ultralytics import YOLO

    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found. Train first: `python -m rdd train`.")
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    split_dir = Path(cfg["path"]) / split

    model = YOLO(str(weights))
    common = dict(imgsz=imgsz, batch=batch, workers=workers)
    if device not in (None, ""):
        common["device"] = device

    print(f"\n=== Overall evaluation on the '{split}' split ===")
    overall = model.val(
        data=str(data_yaml),
        split=split,
        project=str((runs_dir / "eval").resolve()),
        name=split,
        exist_ok=True,
        plots=True,
        **common,
    )
    result = {"weights": str(weights), "split": split, "overall": summarize(overall), "per_country": {}}

    if per_country:
        countries = sorted({country_of(p.name) for p in (split_dir / "images").iterdir() if p.is_file()})
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_name:
            tmp = Path(tmp_name)
            for country in countries:
                built = _country_subset(split_dir, country, tmp)
                if built is None:
                    continue
                yaml_path, n = built
                print(f"\n--- {country} ({n} images) ---")
                m = model.val(
                    data=str(yaml_path),
                    split="val",
                    project=str(tmp / "runs"),
                    name=country,
                    exist_ok=True,
                    plots=False,
                    verbose=False,
                    **common,
                )
                entry = summarize(m)
                entry["images"] = n
                result["per_country"][country] = entry

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (reports_dir / "metrics.md").write_text(to_markdown(result), encoding="utf-8")
    print(f"\nSaved {reports_dir / 'metrics.json'} and {reports_dir / 'metrics.md'}")
    return result


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def to_markdown(result: dict) -> str:
    o = result["overall"]
    lines = [
        f"# Evaluation ({result['split']} split)",
        "",
        f"Weights: `{result['weights']}`",
        "",
        f"**mAP50 {_fmt(o['map50'])}**, mAP50-95 {_fmt(o['map50_95'])}, "
        f"precision {_fmt(o['precision'])}, recall {_fmt(o['recall'])}",
        "",
        "## Per class",
        "",
        "| Class | AP50 | AP50-95 | Precision | Recall |",
        "|---|---|---|---|---|",
    ]
    for name, m in o["per_class"].items():
        lines.append(f"| {name} | {_fmt(m['ap50'])} | {_fmt(m['ap50_95'])} | {_fmt(m['precision'])} | {_fmt(m['recall'])} |")
    if result["per_country"]:
        lines += ["", "## Per country", "", "| Country | Images | mAP50 | mAP50-95 |", "|---|---|---|---|"]
        for country, m in result["per_country"].items():
            lines.append(f"| {country} | {m['images']} | {_fmt(m['map50'])} | {_fmt(m['map50_95'])} |")
    lines.append("")
    return "\n".join(lines)
