"""Task-level ablation: does the VLM second opinion actually help?

Runs the full pipeline (VLM policy 'flag') on test images and, from ONE pass, computes
precision / recall / F1 for:
  A  detector only (conf >= detector threshold)
  B  detector only, high-confidence detections (conf >= gate)
  C  detector + VLM 'reject' verdicts removed
  D  detector + VLM rejects removed + VLM 'relabel' verdicts applied
It also reports how often the VLM agrees with the truth on uncertain detections.

    python -m src.training.evaluate_pipeline --data dataset/data.yaml --limit 40

A 3B model on CPU is slow: keep --limit small, or run with VLM_PROVIDER=gemini / mock.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import cv2
import yaml

from src.config import CLASS_NAMES, ROOT, get_settings
from src.training.evaluate import read_yolo_labels
from src.utils.experiment_log import log_experiment
from src.utils.geometry import match_image


def score(items: List[Dict], gt: Dict[str, tuple], iou_thr: float) -> Dict:
    tp = fp = fn = 0
    for img_name, dets in items:
        gtb, gtc = gt[img_name]
        m = match_image(gtb, gtc, [d["box"] for d in dets], [d["cls"] for d in dets],
                        [d["conf"] for d in dets], iou_thr)
        tp += len(m["tp"])
        fp += len(m["fp"]) + len(m["confused"])
        fn += len(m["fn"]) + len(m["confused"])
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=ROOT / "reports")
    a = ap.parse_args()

    from src.inference.pipeline import InspectionPipeline

    s = dataclasses.replace(get_settings(), vlm_policy="flag")
    pipe = InspectionPipeline(s)
    cfg = yaml.safe_load(a.data.read_text())
    root = Path(cfg["path"])
    img_dir, lab_dir = root / cfg["test"], root / cfg["test"].replace("images", "labels")
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})[: a.limit]

    gt, raw = {}, []
    for f in files:
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        gt[f.name] = read_yolo_labels(lab_dir / f"{f.stem}.txt", w, h)
        res = pipe.run(img, use_vlm=True, report_mode="template", return_image=False)
        dets = [{"box": d["box_xyxy"], "cls": CLASS_NAMES.index(d["label"]), "conf": d["confidence"], "vlm": d["vlm"]}
                for d in res["detections"]]
        raw.append((f.name, dets))
        print(f"[{len(raw)}/{len(files)}] {f.name}: {len(dets)} detections, verdict {res['verdict']}")

    A = raw
    B = [(n, [d for d in ds if d["conf"] >= s.vlm_gate_conf]) for n, ds in raw]
    C = [(n, [d for d in ds if not (d["vlm"] and d["vlm"]["verdict"] == "reject")]) for n, ds in raw]
    D = []
    for n, ds in C:
        nd = []
        for d in ds:
            d = dict(d)
            if d["vlm"] and d["vlm"]["verdict"] == "relabel" and d["vlm"].get("label") in CLASS_NAMES:
                d["cls"] = CLASS_NAMES.index(d["vlm"]["label"])
            nd.append(d)
        D.append((n, nd))
    table = {"A_detector": score(A, gt, a.iou), "B_detector_high_conf_only": score(B, gt, a.iou),
             "C_plus_vlm_reject": score(C, gt, a.iou), "D_plus_vlm_reject_and_relabel": score(D, gt, a.iou)}

    # how good is the VLM as a judge on uncertain detections?
    judge = Counter()
    for n, ds in raw:
        gtb, gtc = gt[n]
        m = match_image(gtb, gtc, [d["box"] for d in ds], [d["cls"] for d in ds], [d["conf"] for d in ds], a.iou)
        tp_idx = {p for p, _ in m["tp"]}
        for i, d in enumerate(ds):
            if d["vlm"] and d["vlm"]["verdict"] in {"confirm", "reject", "relabel"}:
                judge[("tp" if i in tp_idx else "fp", "agree" if d["vlm"]["verdict"] == "confirm" else "disagree")] += 1
    judged = dict((f"{k[0]}_{k[1]}", v) for k, v in judge.items())

    out = {"provider": s.vlm_provider, "model": s.ollama_model if s.vlm_provider == "ollama" else s.gemini_model,
           "images": len(files), "results": table, "vlm_as_judge_on_uncertain_detections": judged,
           "reading": "If C/D do not beat A on F1, keep VLM_POLICY=flag (human review) instead of 'apply'."}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "pipeline_ablation.json").write_text(json.dumps(out, indent=2))
    md = ["# Pipeline ablation: does the VLM help?", "", f"Provider: {out['provider']} / {out['model']}, images: {len(files)}", "",
          "| variant | TP | FP | FN | precision | recall | F1 |", "|---|---|---|---|---|---|---|"]
    md += [f"| {k} | {v['tp']} | {v['fp']} | {v['fn']} | {v['precision']} | {v['recall']} | {v['f1']} |" for k, v in table.items()]
    md += ["", f"VLM as judge on uncertain detections: `{judged}`", "", out["reading"]]
    (a.out / "pipeline_ablation.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    log_experiment("ablation", "vlm_cascade", {"provider": s.vlm_provider, "images": len(files)},
                   {"F1_detector": table["A_detector"]["f1"], "F1_plus_reject": table["C_plus_vlm_reject"]["f1"]})


if __name__ == "__main__":
    main()
