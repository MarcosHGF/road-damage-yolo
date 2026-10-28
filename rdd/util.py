"""Small helpers used across modules."""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from .constants import COUNTRIES_BY_LENGTH


def country_of(filename: str) -> str:
    """Country encoded in an RDD2022 file name.

    RDD2022 names carry the country as a prefix, e.g. ``Japan_007011.jpg``,
    ``United_States_003376.jpg`` or ``China_MotorBike_000378.jpg``.
    """
    stem = Path(filename).stem
    for country in COUNTRIES_BY_LENGTH:
        if stem.startswith(country + "_") or stem == country:
            return country
    m = re.match(r"([A-Za-z]+(?:_[A-Za-z]+)*?)_\d+$", stem)
    return m.group(1) if m else "Unknown"


def image_number(filename: str) -> int | None:
    """Trailing integer of a file name (``Japan_007011.jpg`` -> 7011)."""
    m = re.search(r"(\d+)$", Path(filename).stem)
    return int(m.group(1)) if m else None


def place(src: Path, dst: Path, copy: bool = False) -> None:
    """Hard-link ``src`` to ``dst`` (no extra disk use), else copy.

    Hard links work on NTFS and ext4 without admin rights. If linking is
    impossible (different drive, FAT/exFAT, unsupported FS) it falls back
    to a normal copy.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if not copy:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
