import pytest

from rdd.constants import Paths
from rdd.convert import convert_dataset
from rdd.demo_data import generate_demo_data


@pytest.fixture()
def paths(tmp_path):
    return Paths(data_dir=tmp_path / "data", runs_dir=tmp_path / "runs", reports_dir=tmp_path / "reports")


@pytest.fixture()
def demo_raw(paths):
    generate_demo_data(paths.raw, n_per_country=30)
    return paths


@pytest.fixture()
def converted(demo_raw):
    stats = convert_dataset(demo_raw.raw, demo_raw.converted, keep_empty_ratio=0.5)
    return demo_raw, stats
