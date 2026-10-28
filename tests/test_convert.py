import json
from pathlib import Path

import pytest
from PIL import Image

from rdd.constants import CLASS_NAMES
from rdd.convert import convert_dataset, find_pairs, parse_voc_xml, voc_box_to_yolo

XML = """<annotation>
  <size><width>{w}</width><height>{h}</height><depth>3</depth></size>
  {objects}
</annotation>"""
OBJ = "<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin><xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>"


def make_item(root: Path, country: str, idx: int, objects, size=(600, 600), with_xml=True, split="train"):
    img_dir = root / country / split / "images"
    xml_dir = root / country / split / "annotations" / "xmls"
    img_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)
    name = f"{country}_{idx:06d}"
    Image.new("RGB", size, (90, 90, 90)).save(img_dir / f"{name}.jpg")
    if with_xml:
        body = "\n".join(OBJ.format(n=n, a=a, b=b, c=c, d=d) for n, (a, b, c, d) in objects)
        (xml_dir / f"{name}.xml").write_text(XML.format(w=size[0], h=size[1], objects=body))
    return name


def test_box_math_matches_hand_computation():
    # 600x600 image, box (50,60)-(200,220)
    line = voc_box_to_yolo(2, 600, 600, 50, 60, 200, 220)
    cls, xc, yc, w, h = line.split()
    assert cls == "2"
    assert float(xc) == pytest.approx(125 / 600, abs=1e-6)
    assert float(yc) == pytest.approx(140 / 600, abs=1e-6)
    assert float(w) == pytest.approx(150 / 600, abs=1e-6)
    assert float(h) == pytest.approx(160 / 600, abs=1e-6)


def test_box_is_clamped_and_degenerate_boxes_dropped():
    line = voc_box_to_yolo(0, 100, 100, -20, -20, 50, 50)  # sticks out of the image
    xc, yc, w, h = (float(v) for v in line.split()[1:])
    assert (xc, yc, w, h) == pytest.approx((0.25, 0.25, 0.5, 0.5))
    assert voc_box_to_yolo(0, 100, 100, 10, 10, 10.4, 40) is None  # < 1 px wide
    assert voc_box_to_yolo(0, 100, 100, 150, 150, 200, 200) is None  # fully outside
    swapped = voc_box_to_yolo(0, 100, 100, 50, 50, 10, 10)  # corners swapped
    assert swapped is not None


def test_parse_voc_falls_back_to_image_size(tmp_path):
    img = tmp_path / "a.jpg"
    Image.new("RGB", (320, 200)).save(img)
    xml = tmp_path / "a.xml"
    xml.write_text("<annotation>" + OBJ.format(n="D00", a=1, b=2, c=30, d=40) + "</annotation>")
    w, h, objs = parse_voc_xml(xml, img)
    assert (w, h) == (320, 200)
    assert objs == [("D00", 1.0, 2.0, 30.0, 40.0)]


def test_convert_filters_classes_and_reports_ignored(tmp_path):
    raw = tmp_path / "raw"
    make_item(raw, "Japan", 1, [("D00", (10, 10, 100, 40)), ("D44", (200, 200, 300, 300)), ("D40", (400, 400, 480, 470))])
    make_item(raw, "Japan", 2, [("D20", (0, 0, 200, 200))])
    out = tmp_path / "out"
    stats = convert_dataset(raw, out, keep_empty_ratio=0)
    assert stats["images_with_damage"] == 2
    assert stats["ignored_codes"] == {"D44": 1}
    assert stats["objects_per_class"] == {"longitudinal_crack": 1, "pothole": 1, "alligator_crack": 1}
    label = (out / "labels" / "Japan_000001.txt").read_text().splitlines()
    assert [row.split()[0] for row in label] == ["0", "3"]  # D00 -> 0, D40 -> 3 (D44 gone)
    assert json.loads((out / "stats.json").read_text())["images_written"] == 2


def test_negatives_are_sampled_per_country_and_unannotated_train_counts(tmp_path):
    raw = tmp_path / "raw"
    for i in range(1, 11):
        make_item(raw, "Japan", i, [("D00", (10, 10, 100, 40))])
    for i in range(11, 31):
        make_item(raw, "Japan", i, [], with_xml=(i % 2 == 0))  # half empty XML, half no XML
    make_item(raw, "Japan", 900, [], with_xml=False, split="test")  # test image: never a negative
    out = tmp_path / "out"
    stats = convert_dataset(raw, out, keep_empty_ratio=0.5)
    assert stats["damage_free_available"] == 20  # 10 empty XML + 10 no XML, test image excluded
    assert stats["damage_free_kept"] == 5  # 10 positives * 0.5
    assert not (out / "images" / "Japan_000900.jpg").exists()
    # damage-free images get an empty label file
    empties = [p for p in (out / "labels").iterdir() if p.stat().st_size == 0]
    assert len(empties) == 5

    stats2 = convert_dataset(raw, out, keep_empty_ratio=0.5, include_unannotated_train=False)
    assert stats2["damage_free_available"] == 10


def test_negative_sampling_is_deterministic(tmp_path):
    raw = tmp_path / "raw"
    for i in range(1, 6):
        make_item(raw, "Czech", i, [("D10", (10, 10, 100, 40))])
    for i in range(6, 40):
        make_item(raw, "Czech", i, [])
    runs = []
    for k in range(2):
        out = tmp_path / f"out{k}"
        convert_dataset(raw, out, keep_empty_ratio=1.0, seed=3)
        runs.append(sorted(p.name for p in (out / "images").iterdir()))
    assert runs[0] == runs[1]


def test_junk_and_unmatched_files_are_ignored(tmp_path):
    raw = tmp_path / "raw"
    make_item(raw, "India", 1, [("D00", (10, 10, 100, 40))])
    junk = raw / "__MACOSX" / "India" / "train" / "images"
    junk.mkdir(parents=True)
    Image.new("RGB", (10, 10)).save(junk / "._India_000001.jpg")
    (raw / "India" / "train" / "annotations" / "xmls" / "India_000777.xml").write_text("<annotation/>")  # no image
    pairs, unmatched = find_pairs(raw)
    assert len(pairs) == 1
    assert unmatched == 1


def test_missing_raw_dir_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download"):
        convert_dataset(tmp_path / "nope", tmp_path / "out")


def test_demo_data_roundtrip_has_all_classes(converted):
    _, stats = converted
    assert set(stats["objects_per_class"]) == set(CLASS_NAMES)
    assert stats["synthetic"] is True
    assert "D44" in stats["ignored_codes"]
    assert stats["unmatched_xml"] == 0
