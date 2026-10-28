import json

import pytest

from rdd.cli import main
from rdd.config import DEFAULTS, load_config


def run(paths, *args):
    return main(
        ["--data-dir", str(paths.data_dir), "--runs-dir", str(paths.runs_dir), "--reports-dir", str(paths.reports_dir), *args]
    )


def test_config_defaults_and_override(tmp_path):
    assert load_config(tmp_path / "missing.yaml") == DEFAULTS
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("train:\n  epochs: 7\n  extra:\n    degrees: 0.0\ncountries: [Norway]\n")
    cfg = load_config(cfg_file)
    assert cfg["train"]["epochs"] == 7
    assert cfg["train"]["extra"]["degrees"] == 0.0
    assert cfg["train"]["extra"]["fliplr"] == 0.5  # untouched default survives the merge
    assert cfg["countries"] == ["Norway"]
    assert DEFAULTS["train"]["epochs"] == 100  # defaults were not mutated


def test_pipeline_through_the_cli(paths, capsys):
    assert run(paths, "demo-data", "--images-per-country", "20") == 0
    assert run(paths, "convert") == 0
    assert run(paths, "split") == 0
    assert run(paths, "stats") == 0
    assert paths.data_yaml.exists()
    assert (paths.reports_dir / "class_distribution.png").stat().st_size > 1000
    stats = json.loads((paths.reports_dir / "dataset_stats.json").read_text())
    assert set(stats["splits"]) == {"train", "val", "test"}
    out = capsys.readouterr().out
    assert "NOT the real dataset" in out


def test_download_refuses_to_mix_with_synthetic_data(paths, capsys):
    run(paths, "demo-data", "--images-per-country", "5")
    assert run(paths, "download", "--countries", "Czech") == 2
    assert "synthetic demo data" in capsys.readouterr().err


def test_demo_data_refuses_to_overwrite_real_data(paths, capsys):
    (paths.raw / "Japan").mkdir(parents=True)
    (paths.raw / "Japan" / ".extracted_ok").write_text("ok")
    assert run(paths, "demo-data") == 2
    assert "real dataset" in capsys.readouterr().err


def test_helpful_errors_when_stages_are_skipped(paths, capsys):
    assert run(paths, "convert") == 1
    assert "download" in capsys.readouterr().err
    assert run(paths, "split") == 1
    assert run(paths, "stats") == 1
    assert run(paths, "evaluate") == 1
    assert "Train first" in capsys.readouterr().err
    assert run(paths, "train") == 1  # no data.yaml yet


def test_unknown_command_exits_with_usage():
    with pytest.raises(SystemExit):
        main(["not-a-command"])


def test_doctor_runs(paths, capsys):
    code = run(paths, "doctor")
    out = capsys.readouterr().out
    assert code in (0, 1)
    assert "Python" in out and "Free disk" in out


def test_use_sample_without_a_sample_explains_how_to_build_one(paths, capsys):
    assert run(paths, "split", "--use-sample") == 1
    assert "python -m rdd sample" in capsys.readouterr().err
