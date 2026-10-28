"""YOLO training wrapper (Ultralytics)."""
from __future__ import annotations

from pathlib import Path


def find_latest_weights(detect_dir: Path, name: str, which: str = "best.pt") -> Path | None:
    """Newest ``<detect_dir>/<name>*/weights/<which>`` (Ultralytics numbers repeated runs)."""
    if not detect_dir.exists():
        return None
    candidates = [p for p in detect_dir.glob(f"{name}*/weights/{which}") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _warn_if_cpu(device) -> None:
    try:
        import torch
    except ImportError:
        return
    if device in (None, "", "auto") and not torch.cuda.is_available():
        print(
            "\n*** WARNING: CUDA is not available, training will run on the CPU and be very slow. ***\n"
            "    Run `python -m rdd doctor` to see why (on Windows, pip often installs a CPU-only PyTorch).\n"
        )


def train_model(
    data_yaml: Path,
    runs_dir: Path,
    *,
    model: str = "yolo11s.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = -1,
    patience: int = 25,
    workers: int = 4,
    seed: int = 0,
    name: str = "rdd_yolo",
    device=None,
    resume: bool = False,
    fraction: float = 1.0,
    extra: dict | None = None,
    plots: bool = True,
) -> Path:
    """Train and return the path of ``best.pt``.

    ``model`` ending in ``.pt`` starts from pretrained weights (downloaded
    automatically on first use); ending in ``.yaml`` builds the network from
    scratch with random weights (used by the offline smoke test).
    """
    from ultralytics import YOLO

    detect_dir = runs_dir / "detect"
    if resume:
        last = find_latest_weights(detect_dir, name, "last.pt")
        if last is None:
            raise FileNotFoundError(f"No previous run named '{name}*' under {detect_dir} to resume.")
        print(f"Resuming from {last}")
        YOLO(str(last)).train(resume=True)
        best = find_latest_weights(detect_dir, name, "best.pt")
        if best is None:
            raise FileNotFoundError("Resumed run finished but best.pt was not found.")
        return best

    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found. Run `python -m rdd split` first.")

    _warn_if_cpu(device)
    net = YOLO(model)
    args = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        workers=workers,
        seed=seed,
        project=str(detect_dir.resolve()),
        name=name,
        fraction=fraction,
        plots=plots,
        pretrained=not model.endswith(".yaml"),
    )
    if device not in (None, ""):
        args["device"] = device
    args.update(extra or {})
    net.train(**args)

    save_dir = Path(net.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        best = save_dir / "weights" / "last.pt"
    print(f"\nTraining finished. Weights: {best}")
    return best
