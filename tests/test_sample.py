import pytest

from rdd.constants import CLASS_NAMES
from rdd.sample import make_sample
from rdd.split import split_dataset
from rdd.util import country_of


def test_sample_refuses_synthetic_data(converted):
    paths, _ = converted
    with pytest.raises(ValueError, match="synthetic"):
        make_sample(paths.converted, paths.sample)


def test_sample_is_balanced_small_and_splittable(converted):
    paths, _ = converted
    s = make_sample(paths.converted, paths.sample, max_images=40, max_mb=5, allow_synthetic=True)
    assert s["images"] <= 40
    assert s["bytes"] <= 5 * 1024 * 1024
    assert all(s["per_class"][n] > 0 for n in CLASS_NAMES)  # every class represented
    assert set(s["per_country"]) == {"Japan", "Czech", "India"}
    imgs = {p.stem for p in (paths.sample / "images").iterdir()}
    lbls = {p.stem for p in (paths.sample / "labels").iterdir()}
    assert imgs == lbls
    readme = (paths.sample / "README.md").read_text()
    assert "CC BY-SA 4.0" in readme and "arXiv:2209.08538" in readme
    # The sample must be usable as the input of the split stage.
    split_dataset(paths.sample, paths.yolo)
    assert paths.data_yaml.exists()
    assert {country_of(p.name) for p in (paths.yolo / "train" / "images").iterdir()}


def test_sample_respects_size_budget(converted):
    paths, _ = converted
    s = make_sample(paths.converted, paths.sample, max_images=200, max_mb=0.2, allow_synthetic=True)
    assert s["bytes"] <= 0.2 * 1024 * 1024
    assert s["images"] >= 1
