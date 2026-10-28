"""Constants and filesystem layout shared by every pipeline stage."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Classes
#
# RDD2022 annotations use the CRDDC'2022 damage codes. The dataset's own
# label map defines D00, D01, D10, D11, D20, D40, D43, D44. The challenge
# trains and scores on the four below; the rest are rare/ambiguous and are
# dropped (and counted, so the conversion report shows what was ignored).
# ---------------------------------------------------------------------------
CLASS_MAP = {
    "D00": "longitudinal_crack",
    "D10": "transverse_crack",
    "D20": "alligator_crack",
    "D40": "pothole",
}
CLASS_NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# ---------------------------------------------------------------------------
# Countries and official per-country archives (CRDDC'2022 S3 bucket).
# ---------------------------------------------------------------------------
BASE_URL = (
    "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/"
    "RDD2022/Country_Specific_Data_CRDDC2022"
)
COUNTRY_ZIPS = {
    "Japan": f"{BASE_URL}/RDD2022_Japan.zip",
    "India": f"{BASE_URL}/RDD2022_India.zip",
    "Czech": f"{BASE_URL}/RDD2022_Czech.zip",
    "Norway": f"{BASE_URL}/RDD2022_Norway.zip",
    "United_States": f"{BASE_URL}/RDD2022_United_States.zip",
    "China_MotorBike": f"{BASE_URL}/RDD2022_China_MotorBike.zip",
    "China_Drone": f"{BASE_URL}/RDD2022_China_Drone.zip",
}
# Approximate archive sizes in MB (from the dataset's README), used only to
# warn about low disk space before downloading.
APPROX_ZIP_MB = {
    "Japan": 1023,
    "India": 502,
    "Czech": 245,
    "Norway": 9900,
}
# Longest names first so "China_MotorBike" is never mistaken for "China".
COUNTRIES_BY_LENGTH = sorted(COUNTRY_ZIPS, key=len, reverse=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Paths:
    """Every directory the pipeline reads or writes, rooted in one place."""

    data_dir: Path = ROOT / "data"
    runs_dir: Path = ROOT / "runs"
    reports_dir: Path = ROOT / "reports"

    @property
    def downloads(self) -> Path:
        return self.data_dir / "downloads"

    @property
    def raw(self) -> Path:
        return self.data_dir / "raw"

    @property
    def converted(self) -> Path:
        return self.data_dir / "converted"

    @property
    def yolo(self) -> Path:
        return self.data_dir / "yolo_dataset"

    @property
    def sample(self) -> Path:
        return self.data_dir / "sample"

    @property
    def data_yaml(self) -> Path:
        return self.yolo / "data.yaml"
