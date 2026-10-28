"""Run a trained model on an image, a folder of images, or a video."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .constants import CLASS_NAMES


def predict(
    weights: Path,
    source: Path,
    out_dir: Path,
    *,
    conf: float = 0.25,
    imgsz: int = 640,
    device=None,
    save_txt: bool = False,
) -> dict[str, dict[str, int]]:
    """Save annotated outputs to ``out_dir`` and return ``{file: {class: count}}``."""
    from ultralytics import YOLO

    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found. Train first: `python -m rdd train`.")
    if not Path(source).exists():
        raise FileNotFoundError(f"Source not found: {source}")

    kwargs = dict(
        source=str(source),
        conf=conf,
        imgsz=imgsz,
        save=True,
        save_txt=save_txt,
        project=str(out_dir.parent.resolve()),
        name=out_dir.name,
        exist_ok=True,
        verbose=False,
    )
    if device not in (None, ""):
        kwargs["device"] = device
    results = YOLO(str(weights)).predict(**kwargs)

    summary: dict[str, dict[str, int]] = {}
    for r in results:
        counts = Counter(CLASS_NAMES[int(c)] for c in r.boxes.cls.tolist())
        summary[Path(r.path).name] = dict(counts)
    print(f"Annotated output saved to {out_dir}")
    for fname, counts in summary.items():
        desc = ", ".join(f"{n} x {c}" for c, n in counts.items()) or "no damage detected"
        print(f"  {fname}: {desc}")
    return summary
