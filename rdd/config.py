"""Loads config.yaml and merges it over built-in defaults."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .constants import ROOT

DEFAULTS: dict = {
    "countries": ["Japan", "Czech", "India"],
    "convert": {
        # Fraction of damage-free images kept as background examples,
        # relative to the number of images that contain damage.
        "keep_empty_ratio": 0.15,
        # Treat images under a "train" folder that have no XML as negatives.
        "include_unannotated_train": True,
    },
    "split": {
        "train": 0.70,
        "val": 0.15,
        "test": 0.15,
        "seed": 42,
        # 1 = split image by image. N > 1 keeps runs of N consecutive image
        # numbers together, which limits near-duplicate frames leaking
        # between train and test.
        "group_size": 1,
    },
    "train": {
        "model": "yolo11s.pt",
        "epochs": 100,
        "imgsz": 640,
        "batch": -1,  # -1 = Ultralytics picks the largest batch that fits the GPU
        "patience": 25,
        "workers": 4,
        "seed": 0,
        "name": "rdd_yolo",
        "extra": {
            # Crack orientation (longitudinal vs transverse) carries the class,
            # so rotation is kept small and vertical flips are off.
            "degrees": 5.0,
            "shear": 2.0,
            "fliplr": 0.5,
            "flipud": 0.0,
            "mosaic": 1.0,
            "cos_lr": True,
        },
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path | str | None = None) -> dict:
    """Return the defaults overlaid with ``config.yaml`` (if it exists)."""
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        return _merge(DEFAULTS, user)
    return copy.deepcopy(DEFAULTS)
