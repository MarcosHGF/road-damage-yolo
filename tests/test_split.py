import json

import pytest
import yaml

from rdd.constants import CLASS_NAMES
from rdd.split import split_dataset
from rdd.util import country_of


def names(dirpath):
    return {p.name for p in dirpath.iterdir()}


def test_no_leakage_and_all_images_assigned(converted):
    paths, stats = converted
    summary = split_dataset(paths.converted, paths.yolo)
    train = names(paths.yolo / "train" / "images")
    val = names(paths.yolo / "val" / "images")
    test = names(paths.yolo / "test" / "images")
    assert not (train & val) and not (train & test) and not (val & test)
    assert len(train | val | test) == stats["images_written"]
    assert sum(summary["images"].values()) == stats["images_written"]
    # every image has a label file next to it
    for split in ("train", "val", "test"):
        imgs = {p.stem for p in (paths.yolo / split / "images").iterdir()}
        lbls = {p.stem for p in (paths.yolo / split / "labels").iterdir()}
        assert imgs == lbls


def test_every_country_in_every_split_and_ratios_close(converted):
    paths, _ = converted
    split_dataset(paths.converted, paths.yolo, ratios=(0.7, 0.15, 0.15))
    for split in ("train", "val", "test"):
        countries = {country_of(p.name) for p in (paths.yolo / split / "images").iterdir()}
        assert countries == {"Japan", "Czech", "India"}, split
    total = sum(len(list((paths.yolo / s / "images").iterdir())) for s in ("train", "val", "test"))
    train_n = len(list((paths.yolo / "train" / "images").iterdir()))
    assert 0.6 <= train_n / total <= 0.8


def test_split_is_reproducible_and_seed_changes_it(converted):
    paths, _ = converted
    split_dataset(paths.converted, paths.yolo, seed=1)
    a = names(paths.yolo / "test" / "images")
    split_dataset(paths.converted, paths.yolo, seed=1)
    assert names(paths.yolo / "test" / "images") == a
    split_dataset(paths.converted, paths.yolo, seed=2)
    assert names(paths.yolo / "test" / "images") != a


def test_data_yaml_is_valid_for_ultralytics(converted):
    paths, _ = converted
    split_dataset(paths.converted, paths.yolo)
    cfg = yaml.safe_load(paths.data_yaml.read_text())
    assert cfg["nc"] == 4
    assert cfg["names"] == {i: n for i, n in enumerate(CLASS_NAMES)}
    for key in ("train", "val", "test"):
        assert (paths.yolo / cfg[key]).is_dir()
    assert json.loads((paths.yolo / "split_summary.json").read_text())["seed"] == 42


def test_group_size_keeps_consecutive_images_together(converted):
    paths, _ = converted
    split_dataset(paths.converted, paths.yolo, group_size=10)
    where = {}
    for split in ("train", "val", "test"):
        for p in (paths.yolo / split / "images").iterdir():
            where[p.name] = split
    # Images 1..9 of one country (same block of 10) can differ only through
    # their damage/no-damage stratum, so at most two splits appear per block.
    from rdd.util import image_number

    blocks = {}
    for name, split in where.items():
        key = (country_of(name), image_number(name) // 10)
        blocks.setdefault(key, set()).add(split)
    assert all(len(s) <= 2 for s in blocks.values())
    assert any(len(s) == 1 for s in blocks.values())


def test_bad_ratios_and_missing_dir(converted, tmp_path):
    paths, _ = converted
    with pytest.raises(ValueError):
        split_dataset(paths.converted, paths.yolo, ratios=(0.5, 0.5, 0.5))
    with pytest.raises(FileNotFoundError, match="convert"):
        split_dataset(tmp_path / "missing", paths.yolo)
