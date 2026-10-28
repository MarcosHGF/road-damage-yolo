"""End-to-end smoke test on synthetic data (no download, runs on a CPU)."""
from __future__ import annotations

from pathlib import Path

from .constants import Paths
from .convert import convert_dataset, print_stats
from .demo_data import generate_demo_data
from .evaluate import evaluate_model
from .predict import predict
from .split import split_dataset
from .train import train_model


def run_smoke_test(workdir: Path, images_per_country: int = 24, epochs: int = 2, imgsz: int = 128) -> dict:
    """Run every stage on tiny synthetic data in ``workdir``.

    The network is built from scratch (``yolo11n.yaml``) so nothing is
    downloaded. Accuracy is meaningless after two epochs; the point is to
    prove that the installation and every stage of the pipeline work.
    """
    paths = Paths(data_dir=workdir / "data", runs_dir=workdir / "runs", reports_dir=workdir / "reports")

    print("[1/6] generating synthetic demo data")
    generate_demo_data(paths.raw, n_per_country=images_per_country)

    print("[2/6] converting VOC -> YOLO")
    stats = convert_dataset(paths.raw, paths.converted, keep_empty_ratio=0.5)
    print_stats(stats)

    print("[3/6] splitting")
    summary = split_dataset(paths.converted, paths.yolo)
    print(summary["images"])

    print("[4/6] training (from scratch, CPU-friendly)")
    best = train_model(
        paths.data_yaml,
        paths.runs_dir,
        model="yolo11n.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=8,
        patience=50,
        workers=0,
        name="smoke",
        device="cpu",
        plots=False,
        extra={"amp": False, "val": True},
    )

    print("[5/6] evaluating on the test split")
    result = evaluate_model(
        best, paths.data_yaml, paths.runs_dir, paths.reports_dir, imgsz=imgsz, batch=8, device="cpu", workers=0
    )

    print("[6/6] predicting on a test image")
    sample_img = next((paths.yolo / "test" / "images").iterdir())
    predictions = predict(best, sample_img, paths.runs_dir / "predict", imgsz=imgsz, device="cpu", conf=0.05)

    print("\nSmoke test passed.")
    return {"weights": str(best), "metrics": result, "predictions": predictions}
