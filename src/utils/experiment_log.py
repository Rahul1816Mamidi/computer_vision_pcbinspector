"""Tiny experiment tracker: JSONL log + auto-generated markdown table.

Dependency-free (no MLflow server needed). Each training / evaluation /
benchmark run appends one line to reports/experiments.jsonl.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict

from src.config import ROOT

LOG_PATH = ROOT / "reports" / "experiments.jsonl"
MD_PATH = ROOT / "docs" / "EXPERIMENT_LOG.md"


def log_experiment(kind: str, name: str, params: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kind": kind,
        "name": name,
        "params": params,
        "metrics": metrics,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    render_markdown()


def render_markdown() -> None:
    if not LOG_PATH.exists():
        return
    rows = [json.loads(x) for x in LOG_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
    lines = [
        "# Experiment log",
        "",
        "Auto-generated from `reports/experiments.jsonl`. Do not edit by hand.",
        "",
        "| Time | Kind | Name | Key params | Key metrics |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        p = ", ".join(f"{k}={v}" for k, v in list(r["params"].items())[:6])
        m = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in list(r["metrics"].items())[:6]
        )
        lines.append(f"| {r['time']} | {r['kind']} | {r['name']} | {p} | {m} |")
    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
