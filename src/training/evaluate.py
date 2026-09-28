"""Evaluate on the held-out TEST split and run error analysis.

Produces in reports/:
  metrics_test.json        mAP50, mAP50-95, precision, recall, per-class AP50
  error_analysis.json/.md  TP/FP/FN/misclassified counts, per-class P/R, size buckets, confusion matrix
  error_cases/*.jpg        worst images (green = ground truth, red = prediction)

    python -m src.training.evaluate --data dataset/data.yaml --weights models/best.pt
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.config import CLASS_NAMES, ROOT
from src.utils.experiment_log import log_experiment
from src.utils.geometry import match_image

SIZE_BUCKETS = [("small (<32px)", 0, 32), ("medium (32-64px)", 32, 64), ("large (>=64px)", 64, 1e9)]


def read_yolo_labels(label_file: Path, w: int, h: int) -> Tuple[np.ndarray, List[int]]:
    boxes, cls = [], []
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            c, xc, yc, bw, bh = int(p[0]), *(float(v) for v in p[1:])
            boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
            cls.append(c)
    return np.array(boxes).reshape(-1, 4), cls


def bucket_of(box) -> str:
    side = float(np.sqrt(max(box[2] - box[0], 0) * max(box[3] - box[1], 0)))
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= side < hi:
            return name
    return SIZE_BUCKETS[-1][0]


def draw_case(img, gt_boxes, gt_cls, preds) -> np.ndarray:
    out = img.copy()
    t = max(2, int(max(out.shape[:2]) / 500))
    for b, c in zip(gt_boxes, gt_cls):
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 0), t)
        cv2.putText(out, f"GT {CLASS_NAMES[c]}", (int(b[0]), max(12, int(b[1]) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t / 2 + 0.3, (0, 200, 0), 1)
    for d in preds:
        b = d.xyxy
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), t)
        cv2.putText(out, f"{d.label} {d.conf:.2f}", (int(b[0]), int(b[3]) + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t / 2 + 0.3, (0, 0, 255), 1)
    return out


def error_analysis(detector, images_dir: Path, labels_dir: Path, conf: float, iou_thr: float,
                   out_dir: Path, n_worst: int = 12) -> Dict:
    nC = len(CLASS_NAMES)
    cm = np.zeros((nC + 1, nC + 1), dtype=int)  # rows = truth, cols = prediction, last = background
    per_class = {c: Counter() for c in CLASS_NAMES}
    bucket = {b[0]: Counter() for b in SIZE_BUCKETS}
    image_errors = []

    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_boxes, gt_cls = read_yolo_labels(labels_dir / f"{f.stem}.txt", w, h)
        preds = detector.predict(img, conf=conf, iou=0.5)
        m = match_image(gt_boxes, gt_cls, [d.xyxy for d in preds], [d.cls_id for d in preds],
                        [d.conf for d in preds], iou_thr)
        for p, g in m["tp"]:
            cm[gt_cls[g], preds[p].cls_id] += 1
            per_class[CLASS_NAMES[gt_cls[g]]]["tp"] += 1
            bucket[bucket_of(gt_boxes[g])]["tp"] += 1
        for p, g in m["confused"]:
            cm[gt_cls[g], preds[p].cls_id] += 1
            per_class[CLASS_NAMES[gt_cls[g]]]["fn"] += 1
            per_class[preds[p].label]["fp"] += 1
            per_class[CLASS_NAMES[gt_cls[g]]]["confused_as_" + preds[p].label] += 1
            bucket[bucket_of(gt_boxes[g])]["fn"] += 1
        for p in m["fp"]:
            cm[nC, preds[p].cls_id] += 1
            per_class[preds[p].label]["fp"] += 1
        for g in m["fn"]:
            cm[gt_cls[g], nC] += 1
            per_class[CLASS_NAMES[gt_cls[g]]]["fn"] += 1
            bucket[bucket_of(gt_boxes[g])]["fn"] += 1
        n_err = len(m["fp"]) + len(m["fn"]) + len(m["confused"])
        if n_err:
            image_errors.append((n_err, f, gt_boxes, gt_cls, preds))

    def prf(c: Counter) -> Dict:
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        return {"tp": tp, "fp": fp, "fn": fn,
                "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                "recall": round(tp / (tp + fn), 4) if tp + fn else None}

    result = {
        "operating_point": {"conf": conf, "iou_match": iou_thr, "n_images": len(files)},
        "per_class": {k: {**prf(v), **{kk: vv for kk, vv in v.items() if kk.startswith("confused_as_")}}
                      for k, v in per_class.items()},
        "by_defect_size": {k: prf(v) for k, v in bucket.items()},
        "confusion_matrix": {"labels": CLASS_NAMES + ["background"], "matrix": cm.tolist()},
    }
    tot = Counter()
    for v in per_class.values():
        tot.update({k: v[k] for k in ("tp", "fp", "fn")})
    result["overall"] = prf(tot)

    image_errors.sort(key=lambda x: -x[0])
    case_dir = out_dir / "error_cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    for old in case_dir.glob("*.jpg"):
        old.unlink()
    worst = []
    for n_err, f, gtb, gtc, preds in image_errors[:n_worst]:
        cv2.imwrite(str(case_dir / f"{n_err:02d}err_{f.stem}.jpg"), draw_case(cv2.imread(str(f)), gtb, gtc, preds))
        worst.append({"image": f.name, "errors": n_err})
    result["worst_images"] = worst
    return result


def write_markdown(res: Dict, metrics: Dict, path: Path) -> None:
    L = ["# Evaluation report (test split)", ""]
    if metrics:
        L += ["## Detector metrics (Ultralytics val, conf=0.001)", "",
              f"- mAP@0.5: **{metrics.get('mAP50', float('nan')):.4f}**",
              f"- mAP@0.5:0.95: **{metrics.get('mAP50-95', float('nan')):.4f}**",
              f"- precision: {metrics.get('precision', float('nan')):.4f}  recall: {metrics.get('recall', float('nan')):.4f}", ""]
        if metrics.get("per_class_AP50"):
            L += ["| class | AP@0.5 |", "|---|---|"] + [f"| {k} | {v:.4f} |" for k, v in metrics["per_class_AP50"].items()] + [""]
    op = res["operating_point"]
    L += [f"## Error analysis at conf={op['conf']}, IoU={op['iou_match']} ({op['n_images']} images)", "",
          f"Overall: {res['overall']}", "", "| class | TP | FP | FN | precision | recall |", "|---|---|---|---|---|---|"]
    for k, v in res["per_class"].items():
        L.append(f"| {k} | {v['tp']} | {v['fp']} | {v['fn']} | {v['precision']} | {v['recall']} |")
    L += ["", "### By defect size", "", "| bucket | TP | FN | recall |", "|---|---|---|---|"]
    for k, v in res["by_defect_size"].items():
        L.append(f"| {k} | {v['tp']} | {v['fn']} | {v['recall']} |")
    cmx = res["confusion_matrix"]
    L += ["", "### Confusion matrix (rows = truth, columns = prediction)", "",
          "| truth \\ pred | " + " | ".join(cmx["labels"]) + " |", "|---" * (len(cmx["labels"]) + 1) + "|"]
    for lab, row in zip(cmx["labels"], cmx["matrix"]):
        L.append(f"| {lab} | " + " | ".join(str(x) for x in row) + " |")
    L += ["", "### Worst images", ""] + [f"- {w['image']}: {w['errors']} errors (see reports/error_cases/)" for w in res["worst_images"]]
    L += ["", "### How to read this", "",
          "- High FN in the small bucket means the model misses tiny defects: raise `--imgsz` or train on tiles.",
          "- Large off-diagonal cells show which defect types get confused (e.g. spur vs mouse_bite).",
          "- A `background` column count is missed defects; a `background` row count is hallucinated defects."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--conf", type=float, default=0.25, help="operating point for error analysis")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", type=Path, default=ROOT / "reports")
    a = ap.parse_args()

    from ultralytics import YOLO

    from src.inference.detector import Detector, load_meta

    a.out.mkdir(parents=True, exist_ok=True)
    imgsz = a.imgsz or load_meta(Path(a.weights).parent).get("imgsz") or 640

    # 1) standard detection metrics
    m = YOLO(a.weights).val(data=str(a.data), split="test", imgsz=imgsz, conf=0.001, iou=0.6,
                            device=a.device, plots=True, project=str(a.out), name="val_test", exist_ok=True)
    per_class = {}
    for i, ap50 in zip(getattr(m.box, "ap_class_index", []), getattr(m.box, "ap50", [])):
        per_class[CLASS_NAMES[int(i)]] = float(ap50)
    metrics = {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map), "precision": float(m.box.mp),
               "recall": float(m.box.mr), "per_class_AP50": per_class}
    (a.out / "metrics_test.json").write_text(json.dumps(metrics, indent=2))

    # 2) error analysis with our own IoU matcher, using the same Detector class that serves production
    import yaml

    cfg = yaml.safe_load(a.data.read_text())
    root = Path(cfg["path"])
    det = Detector(a.weights, imgsz=imgsz, device=a.device)
    res = error_analysis(det, root / cfg["test"], root / cfg["test"].replace("images", "labels"),
                         a.conf, a.iou, a.out)
    (a.out / "error_analysis.json").write_text(json.dumps(res, indent=2))
    write_markdown(res, metrics, a.out / "evaluation_report.md")
    log_experiment("evaluate", Path(a.weights).name, {"imgsz": imgsz, "conf": a.conf},
                   {"mAP50": metrics["mAP50"], "mAP50-95": metrics["mAP50-95"],
                    "precision@conf": res["overall"]["precision"], "recall@conf": res["overall"]["recall"]})
    print(json.dumps({k: metrics[k] for k in ("mAP50", "mAP50-95", "precision", "recall")}, indent=2))
    print("Wrote", a.out / "evaluation_report.md")


if __name__ == "__main__":
    main()
