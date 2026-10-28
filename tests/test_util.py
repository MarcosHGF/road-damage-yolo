import os

import pytest

from rdd.util import country_of, human_bytes, image_number, place


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Japan_007011.jpg", "Japan"),
        ("United_States_003376.jpg", "United_States"),
        ("China_MotorBike_000378.jpg", "China_MotorBike"),
        ("China_Drone_000074.xml", "China_Drone"),
        ("Norway_002422.jpg", "Norway"),
        ("Czech_000001.jpg", "Czech"),
        ("Atlantis_000001.jpg", "Atlantis"),
        ("weird.jpg", "Unknown"),
    ],
)
def test_country_of(name, expected):
    assert country_of(name) == expected


def test_image_number():
    assert image_number("Japan_007011.jpg") == 7011
    assert image_number("noDigits.jpg") is None


def test_human_bytes():
    assert human_bytes(512) == "512 B"
    assert human_bytes(2048) == "2.0 KB"


def test_place_hardlinks_when_possible_and_overwrites(tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("one")
    dst = tmp_path / "sub" / "b.txt"
    place(src, dst)
    assert dst.read_text() == "one"
    assert os.path.samefile(src, dst)  # hard link on a normal filesystem
    other = tmp_path / "c.txt"
    other.write_text("two")
    place(other, dst)  # replaces the existing destination
    assert dst.read_text() == "two"
    place(src, dst, copy=True)
    assert dst.read_text() == "one"
    assert not os.path.samefile(src, dst)
