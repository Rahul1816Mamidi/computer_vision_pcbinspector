"""Convert the Kaggle PKU-Market-PCB dataset (Pascal VOC XML) to YOLO format.

Usage (Kaggle):
    python -m src.data.prepare_dataset --src /kaggle/input --out /kaggle/working/dataset
Usage (local):
    python -m src.data.prepare_dataset --src ./raw/pcb-defects --out ./dataset

The script scans --src recursively, so it does not depend on the exact folder
layout of the Kaggle download. It prints counts so you can verify what it found.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

from src.config import CLASS_NAMES

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

_ALIASES = {
    "missing_hole": "missing_hole", "missinghole": "missing_hole", "missing": "missing_hole",
    "mouse_bite": "mouse_bite", "mousebite": "mouse_bite", "mouse": "mouse_bite",
    "open_circuit": "open_circuit", "opencircuit": "open_circuit", "open": "open_circuit",
    "short": "short",
    "spur": "spur",
    "spurious_copper": "spurious_copper", "spuriouscopper": "spurious_copper", "copper": "spurious_copper",
}


def normalize_class(name: str) -> Optional[str]:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    return _ALIASES.get(key)


def _excluded(path: Path, exclude: Tuple[str, ...]) -> bool:
    parts = [p.lower() for p in path.parts]
    return any(any(tok in part for tok in exclude) for part in parts)


def find_pairs(src: Path, exclude: Tuple[str, ...]) -> Tuple[List[Tuple[Path, Path]], Dict[str, int]]:
    """Return [(image_path, xml_path)] and a small stats dict."""
    images: Dict[str, List[Path]] = defaultdict(list)
    xmls: List[Path] = []
    skipped_excluded = 0
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        if _excluded(p.relative_to(src), exclude):
            if p.suffix.lower() in IMG_EXTS or p.suffix.lower() == ".xml":
                skipped_excluded += 1
            continue
        ext = p.suffix.lower()
        if ext in IMG_EXTS:
            images[p.stem.lower()].append(p)
        elif ext == ".xml":
            xmls.append(p)

    pairs: List[Tuple[Path, Path]] = []
    unmatched = 0
    for x in sorted(xmls):
        cands = images.get(x.stem.lower(), [])
        if not cands:
            # fall back to <filename> inside the XML
            try:
                fn = ET.parse(x).getroot().findtext("filename") or ""
                cands = images.get(Path(fn).stem.lower(), [])
            except ET.ParseError:
                cands = []
        if not cands:
            unmatched += 1
            continue
        same_dir = [c for c in cands if c.parent.name == x.parent.name]
        pairs.append(((same_dir or cands)[0], x))
    stats = {
        "xml_files": len(xmls),
        "images_indexed": sum(len(v) for v in images.values()),
        "pairs": len(pairs),
        "xml_without_image": unmatched,
        "skipped_by_exclude_filter": skipped_excluded,
    }
    return pairs, stats


def parse_voc(xml_path: Path, image_path: Path) -> Tuple[int, int, List[Tuple[int, float, float, float, float]]]:
    """Return (width, height, [(cls_id, x1, y1, x2, y2), ...]) in pixels."""
    root = ET.parse(xml_path).getroot()
    w = int(float(root.findtext("size/width") or 0))
    h = int(float(root.findtext("size/height") or 0))
    if w <= 0 or h <= 0:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"cannot read image {image_path}")
        h, w = img.shape[:2]
    boxes = []
    for obj in root.findall("object"):
        cname = normalize_class(obj.findtext("name") or "")
        if cname is None:
            print(f"  [warn] unknown class '{obj.findtext('name')}' in {xml_path.name}, skipped")
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x1, y1, x2, y2 = (float(bb.findtext(k) or 0) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        boxes.append((CLASS_NAMES.index(cname), x1, y1, x2, y2))
    return w, h, boxes


def to_yolo_lines(w: int, h: int, boxes) -> List[str]:
    lines = []
    for cls_id, x1, y1, x2, y2 in boxes:
        xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def stratified_split(items: List[dict], ratios=(0.7, 0.15, 0.15), seed=42) -> Dict[str, List[dict]]:
    """Split by each image's dominant class so every split sees every defect type."""
    rng = random.Random(seed)
    groups: Dict[int, List[dict]] = defaultdict(list)
    for it in items:
        groups[it["dominant"]].append(it)
    out = {"train": [], "val": [], "test": []}
    for key in sorted(groups):
        g = groups[key]
        rng.shuffle(g)
        n = len(g)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        out["train"] += g[:n_train]
        out["val"] += g[n_train:n_train + n_val]
        out["test"] += g[n_train + n_val:]
    return out


def prepare(src: Path, out: Path, exclude: Tuple[str, ...], seed: int, limit: Optional[int] = None) -> dict:
    pairs, stats = find_pairs(src, exclude)
    print("Scan:", json.dumps(stats))
    if not pairs:
        raise SystemExit(
            "No (image, xml) pairs found. Check --src points at the extracted dataset "
            "(Kaggle: /kaggle/input/...). Use --exclude '' to disable the rotation filter."
        )
    if limit:
        pairs = pairs[:limit]

    items = []
    for img_path, xml_path in pairs:
        try:
            w, h, boxes = parse_voc(xml_path, img_path)
        except (ET.ParseError, ValueError) as e:
            print(f"  [warn] skipping {xml_path.name}: {e}")
            continue
        if not boxes:
            continue
        dominant = Counter(b[0] for b in boxes).most_common(1)[0][0]
        items.append({"img": img_path, "w": w, "h": h, "boxes": boxes, "dominant": dominant})

    splits = stratified_split(items, seed=seed)
    if out.exists():
        shutil.rmtree(out)
    manifest: Dict[str, List[str]] = {}
    counts = {s: Counter() for s in splits}
    used_names = set()
    for split, its in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        manifest[split] = []
        for it in its:
            name = it["img"].stem
            if name in used_names:  # avoid collisions across class folders
                name = f"{it['img'].parent.name}_{name}"
            used_names.add(name)
            dst = out / "images" / split / f"{name}{it['img'].suffix.lower()}"
            shutil.copy2(it["img"], dst)
            (out / "labels" / split / f"{name}.txt").write_text(
                "\n".join(to_yolo_lines(it["w"], it["h"], it["boxes"])) + "\n"
            )
            manifest[split].append(dst.name)
            for b in it["boxes"]:
                counts[split][CLASS_NAMES[b[0]]] += 1

    data_yaml = (
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(CLASS_NAMES)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )
    (out / "data.yaml").write_text(data_yaml)
    summary = {
        "scan": stats,
        "seed": seed,
        "images_per_split": {s: len(v) for s, v in manifest.items()},
        "boxes_per_class_per_split": {s: dict(c) for s, c in counts.items()},
    }
    (out / "dataset_stats.json").write_text(json.dumps(summary, indent=2))
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote YOLO dataset to {out}  (data.yaml ready)")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="folder containing the extracted Kaggle dataset")
    ap.add_argument("--out", default=Path("dataset"), type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--exclude", default="rotat,augment",
        help="comma-separated substrings; any path containing one is ignored "
             "(default skips rotated/augmented copies to avoid train/test leakage)",
    )
    ap.add_argument("--limit", type=int, default=None, help="use only first N pairs (debugging)")
    a = ap.parse_args()
    exclude = tuple(t.strip().lower() for t in a.exclude.split(",") if t.strip())
    prepare(a.src, a.out, exclude, a.seed, a.limit)


if __name__ == "__main__":
    main()
