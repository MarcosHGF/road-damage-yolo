"""Command line interface: ``python -m rdd <command>``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .constants import COUNTRY_ZIPS, ROOT, Paths


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m rdd",
        description="Road damage detection with YOLO on RDD2022.",
    )
    p.add_argument("--config", default=None, help="Path to a config YAML (default: ./config.yaml)")
    p.add_argument("--data-dir", default=None, help="Data root (default: ./data)")
    p.add_argument("--runs-dir", default=None, help="Training/eval output root (default: ./runs)")
    p.add_argument("--reports-dir", default=None, help="Metrics/figures output (default: ./reports)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Check Python, PyTorch/CUDA, Ultralytics, disk space")

    d = sub.add_parser("download", help="Download and extract RDD2022 country archives")
    d.add_argument("--countries", nargs="+", choices=sorted(COUNTRY_ZIPS), help="Default: from config.yaml")
    d.add_argument("--keep-zip", action="store_true", help="Keep the .zip files after extraction")

    g = sub.add_parser("demo-data", help="Generate SYNTHETIC data in the RDD2022 layout (not the real dataset)")
    g.add_argument("--images-per-country", type=int, default=40)
    g.add_argument("--seed", type=int, default=7)

    c = sub.add_parser("convert", help="Convert Pascal VOC XML to YOLO labels")
    c.add_argument("--keep-empty-ratio", type=float, default=None)
    c.add_argument("--no-unannotated", action="store_true", help="Ignore train images that have no XML file")
    c.add_argument("--seed", type=int, default=None)

    s = sub.add_parser("split", help="Create train/val/test folders and data.yaml")
    s.add_argument("--use-sample", action="store_true", help="Split data/sample (the bundled real-data sample)")
    s.add_argument("--ratios", nargs=3, type=float, metavar=("TRAIN", "VAL", "TEST"))
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--group-size", type=int, default=None, help="Keep N consecutive image numbers together")
    s.add_argument("--copy", action="store_true", help="Copy files instead of hard-linking")

    sub.add_parser("stats", help="Print dataset statistics and write report figures")

    sm = sub.add_parser("sample", help="Build a GitHub-sized real-data sample in data/sample")
    sm.add_argument("--max-images", type=int, default=300)
    sm.add_argument("--max-mb", type=float, default=60.0)
    sm.add_argument("--seed", type=int, default=42)

    t = sub.add_parser("train", help="Train YOLO")
    t.add_argument("--model", default=None, help="e.g. yolo11n.pt, yolo11s.pt, yolo11m.pt, yolov8s.pt")
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--imgsz", type=int, default=None)
    t.add_argument("--batch", type=int, default=None, help="-1 = automatic")
    t.add_argument("--patience", type=int, default=None)
    t.add_argument("--workers", type=int, default=None, help="Dataloader workers (use 0 if Windows runs out of memory)")
    t.add_argument("--device", default=None, help="0, 0,1, cpu, mps (default: automatic)")
    t.add_argument("--name", default=None)
    t.add_argument("--fraction", type=float, default=1.0, help="Fraction of the training set to use")
    t.add_argument("--resume", action="store_true", help="Resume the latest interrupted run")

    e = sub.add_parser("evaluate", help="Evaluate on val/test, overall and per country")
    e.add_argument("--weights", default=None, help="Default: newest best.pt of the configured run name")
    e.add_argument("--split", default="test", choices=["val", "test"])
    e.add_argument("--imgsz", type=int, default=None)
    e.add_argument("--batch", type=int, default=16)
    e.add_argument("--device", default=None)
    e.add_argument("--workers", type=int, default=None)
    e.add_argument("--no-per-country", action="store_true")

    pr = sub.add_parser("predict", help="Run the trained model on images/videos")
    pr.add_argument("--source", required=True, help="Image, folder or video")
    pr.add_argument("--weights", default=None)
    pr.add_argument("--conf", type=float, default=0.25)
    pr.add_argument("--imgsz", type=int, default=None)
    pr.add_argument("--device", default=None)
    pr.add_argument("--save-txt", action="store_true")

    k = sub.add_parser("smoke-test", help="Run the whole pipeline on tiny synthetic data (no download)")
    k.add_argument("--workdir", default=None, help="Default: ./.smoke")
    k.add_argument("--epochs", type=int, default=2)
    return p


def _paths(args) -> Paths:
    base = Paths()
    return Paths(
        data_dir=Path(args.data_dir) if args.data_dir else base.data_dir,
        runs_dir=Path(args.runs_dir) if args.runs_dir else base.runs_dir,
        reports_dir=Path(args.reports_dir) if args.reports_dir else base.reports_dir,
    )


def _weights(args, paths: Paths, cfg: dict) -> Path:
    from .train import find_latest_weights

    if args.weights:
        return Path(args.weights)
    found = find_latest_weights(paths.runs_dir / "detect", cfg["train"]["name"])
    if found is None:
        raise FileNotFoundError(
            f"No trained weights under {paths.runs_dir / 'detect'}. Train first (`python -m rdd train`) "
            "or pass --weights path/to/best.pt."
        )
    return found


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = load_config(args.config)
    paths = _paths(args)
    cmd = args.command

    try:
        if cmd == "doctor":
            from .doctor import run

            return run(paths)

        if cmd == "download":
            from .download import fetch_countries

            if (paths.raw / "SYNTHETIC_DEMO_DATA.txt").exists():
                print(
                    f"{paths.raw} holds synthetic demo data. Delete that folder, or use a different "
                    "--data-dir, before downloading the real dataset.",
                    file=sys.stderr,
                )
                return 2
            fetch_countries(args.countries or cfg["countries"], paths, keep_zip=args.keep_zip)
            print("\nNext: python -m rdd convert")

        elif cmd == "demo-data":
            from .demo_data import generate_demo_data

            if paths.raw.exists() and any((p / ".extracted_ok").exists() for p in paths.raw.iterdir() if p.is_dir()):
                print(f"{paths.raw} already holds the real dataset; use a different --data-dir.", file=sys.stderr)
                return 2
            summary = generate_demo_data(paths.raw, args.images_per_country, seed=args.seed)
            print(f"Wrote {summary['images']} synthetic images ({summary['objects']} boxes) to {paths.raw}")
            print("This is NOT the real dataset. Next: python -m rdd convert")

        elif cmd == "convert":
            from .convert import convert_dataset, print_stats

            conv = cfg["convert"]
            stats = convert_dataset(
                paths.raw,
                paths.converted,
                keep_empty_ratio=conv["keep_empty_ratio"] if args.keep_empty_ratio is None else args.keep_empty_ratio,
                include_unannotated_train=False if args.no_unannotated else conv["include_unannotated_train"],
                seed=cfg["split"]["seed"] if args.seed is None else args.seed,
            )
            print_stats(stats)
            print("\nNext: python -m rdd split")

        elif cmd == "split":
            from .split import split_dataset

            sp = cfg["split"]
            if args.use_sample and not (paths.sample / "images").exists():
                raise FileNotFoundError(
                    f"{paths.sample} has no images yet. Build it from the real dataset with "
                    "`python -m rdd download`, `convert`, then `python -m rdd sample`."
                )
            ratios = tuple(args.ratios) if args.ratios else (sp["train"], sp["val"], sp["test"])
            summary = split_dataset(
                paths.sample if args.use_sample else paths.converted,
                paths.yolo,
                ratios=ratios,
                seed=sp["seed"] if args.seed is None else args.seed,
                group_size=sp["group_size"] if args.group_size is None else args.group_size,
                copy_files=args.copy,
            )
            print(f"Split sizes: {summary['images']}")
            print(f"data.yaml : {paths.data_yaml}")
            print("\nNext: python -m rdd train")

        elif cmd == "stats":
            from .stats import dataset_stats, plot_stats, print_stats

            stats = dataset_stats(paths.yolo)
            if not stats["splits"]:
                raise FileNotFoundError(f"No split dataset in {paths.yolo}. Run `python -m rdd split` first.")
            print_stats(stats)
            for path in plot_stats(stats, paths.reports_dir):
                print(f"wrote {path}")

        elif cmd == "sample":
            from .sample import make_sample

            s = make_sample(paths.converted, paths.sample, args.max_images, args.max_mb, seed=args.seed)
            print(f"Sample: {s['images']} images, {s['boxes']} boxes, {s['bytes'] / 1024 / 1024:.1f} MB -> {paths.sample}")
            print("Commit data/sample/ to GitHub. Keep the CC BY-SA notice in data/sample/README.md.")

        elif cmd == "train":
            from .train import train_model

            tr = cfg["train"]
            pick = lambda a, k: tr[k] if a is None else a  # noqa: E731
            train_model(
                paths.data_yaml,
                paths.runs_dir,
                model=pick(args.model, "model"),
                epochs=pick(args.epochs, "epochs"),
                imgsz=pick(args.imgsz, "imgsz"),
                batch=pick(args.batch, "batch"),
                patience=pick(args.patience, "patience"),
                workers=pick(args.workers, "workers"),
                seed=tr["seed"],
                name=pick(args.name, "name"),
                device=args.device,
                resume=args.resume,
                fraction=args.fraction,
                extra=tr["extra"],
            )
            print("\nNext: python -m rdd evaluate")

        elif cmd == "evaluate":
            from .evaluate import evaluate_model

            evaluate_model(
                _weights(args, paths, cfg),
                paths.data_yaml,
                paths.runs_dir,
                paths.reports_dir,
                split=args.split,
                imgsz=args.imgsz or cfg["train"]["imgsz"],
                batch=args.batch,
                device=args.device,
                per_country=not args.no_per_country,
                workers=cfg["train"]["workers"] if args.workers is None else args.workers,
            )

        elif cmd == "predict":
            from .predict import predict

            predict(
                _weights(args, paths, cfg),
                Path(args.source),
                paths.runs_dir / "predict",
                conf=args.conf,
                imgsz=args.imgsz or cfg["train"]["imgsz"],
                device=args.device,
                save_txt=args.save_txt,
            )

        elif cmd == "smoke-test":
            from .smoke import run_smoke_test

            run_smoke_test(Path(args.workdir) if args.workdir else ROOT / ".smoke", epochs=args.epochs)

    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # download errors etc. get a readable message
        from .download import DownloadError

        if isinstance(exc, DownloadError):
            print(f"error: {exc}", file=sys.stderr)
            return 1
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
