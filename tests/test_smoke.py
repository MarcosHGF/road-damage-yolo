import json

import pytest

pytest.importorskip("ultralytics")


@pytest.mark.slow
def test_full_pipeline_trains_evaluates_and_predicts(tmp_path):
    from rdd.smoke import run_smoke_test

    out = run_smoke_test(tmp_path / "smoke", images_per_country=16, epochs=1, imgsz=96)
    from pathlib import Path

    assert Path(out["weights"]).exists()
    metrics = json.loads((tmp_path / "smoke" / "reports" / "metrics.json").read_text())
    assert set(metrics["per_country"]) == {"Japan", "Czech", "India"}
    assert metrics["overall"]["map50"] is not None
    assert out["predictions"]  # one entry per predicted image
