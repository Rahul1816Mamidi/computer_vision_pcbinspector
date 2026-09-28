from pathlib import Path

from src.config import CLASS_NAMES
from src.data.prepare_dataset import normalize_class, prepare, to_yolo_lines
from src.data.synthetic import generate


def test_normalize_class():
    assert normalize_class("Mouse_bite") == "mouse_bite"
    assert normalize_class("open-circuit") == "open_circuit"
    assert normalize_class("Missing hole") == "missing_hole"
    assert normalize_class("banana") is None


def test_yolo_line_math():
    line = to_yolo_lines(100, 200, [(3, 10, 20, 30, 60)])[0].split()
    assert line[0] == "3"
    assert [round(float(v), 3) for v in line[1:]] == [0.2, 0.2, 0.2, 0.2]


def test_end_to_end_conversion(tmp_path: Path):
    raw = generate(tmp_path / "raw", n_per_class=10, seed=1)
    out = tmp_path / "ds"
    s = prepare(raw, out, ("rotat",), seed=42)
    assert sum(s["images_per_split"].values()) == 60
    assert s["images_per_split"]["train"] > s["images_per_split"]["test"] > 0
    assert (out / "data.yaml").exists()
    # every image has a label, every class id is valid
    for split in ("train", "val", "test"):
        for img in (out / "images" / split).iterdir():
            lab = out / "labels" / split / f"{img.stem}.txt"
            assert lab.exists()
            for line in lab.read_text().split("\n"):
                if line:
                    assert 0 <= int(line.split()[0]) < len(CLASS_NAMES)


def test_rotated_folder_is_excluded(tmp_path: Path):
    raw = generate(tmp_path / "raw", n_per_class=2, seed=2)
    (tmp_path / "raw" / "rotation").mkdir()
    (tmp_path / "raw" / "rotation" / "x.xml").write_text("<annotation/>")
    s = prepare(raw, tmp_path / "ds", ("rotat",), seed=1)
    assert s["scan"]["skipped_by_exclude_filter"] >= 1
