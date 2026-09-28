"""Verdict logic and inspection-report generation (LLM with a template fallback)."""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from src.inference.detector import Detection
from src.inference.rag import KnowledgeBase
from src.inference.vlm import BaseVLM


def decide_verdict(dets: List[Detection], gate: float) -> str:
    """PASS / REVIEW / FAIL.

    FAIL   : at least one strong detection (conf >= gate, or VLM confirmed/relabelled it)
    REVIEW : detections exist but all are weak or disputed -> a human should look
    PASS   : nothing detected
    """
    if not dets:
        return "PASS"
    for d in dets:
        vlm_ok = bool(d.vlm) and d.vlm.get("verdict") in {"confirm", "relabel"}
        if d.conf >= gate or vlm_ok:
            return "FAIL"
    return "REVIEW"


def summarize(dets: List[Detection]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for d in dets:
        out[d.label] = out.get(d.label, 0) + 1
    return out


def template_report(verdict: str, dets: List[Detection], kb: Optional[KnowledgeBase]) -> str:
    """Deterministic report built directly from the knowledge base. Always available."""
    if not dets:
        return "Inspection result: PASS. No defects were detected on this board."
    counts = ", ".join(f"{n} x {k}" for k, n in summarize(dets).items())
    lines = [f"Inspection result: {verdict}. Detected: {counts}."]
    seen = set()
    for d in dets:
        if d.label in seen:
            continue
        seen.add(d.label)
        tag = " (flagged for manual review)" if d.needs_review else ""
        lines.append(f"- {d.label}{tag}, confidence {d.conf:.2f}.")
        if kb:
            for sec in ("Severity", "Likely causes", "Recommended action"):
                txt = kb.section(d.label, sec)
                if txt:
                    lines.append(f"  {sec}: {txt}")
    if verdict == "REVIEW":
        lines.append("Only low-confidence findings exist. A person should confirm them on the physical board.")
    return "\n".join(lines)


def build_llm_prompt(verdict: str, dets: List[Detection], contexts: List[Dict]) -> str:
    det_json = json.dumps([d.to_dict() for d in dets], indent=1)
    ctx = "\n".join(f"[{c['title']} / {c['section']}] {c['body']}" for c in contexts)
    return (
        "You write short PCB inspection reports for a production engineer.\n"
        "Rules: use ONLY the facts in DETECTIONS and CONTEXT. Do not invent measurements, "
        "board names or standards. Mention any detection flagged needs_review. "
        "Maximum 150 words, plain text, no markdown.\n\n"
        f"VERDICT: {verdict}\nDETECTIONS:\n{det_json}\n\nCONTEXT:\n{ctx}\n\nReport:"
    )


def generate_report(
    mode: str,
    verdict: str,
    dets: List[Detection],
    kb: Optional[KnowledgeBase],
    vlm: Optional[BaseVLM],
) -> Tuple[str, str, List[Dict]]:
    """Return (text, mode_used, retrieved_contexts). Falls back to the template on any LLM problem."""
    contexts: List[Dict] = []
    if kb and dets:
        seen = set()
        for d in dets:
            if d.label in seen:
                continue
            seen.add(d.label)
            contexts += kb.retrieve_for_label(d.label, k=3)
        uniq = {(c["doc"], c["section"]): c for c in contexts}
        contexts = list(uniq.values())
        contexts += kb.search("verdict rules escalation", k=1)

    if mode == "llm" and vlm is not None and dets and vlm.available():
        try:
            text = (vlm.ask_text(build_llm_prompt(verdict, dets, contexts)) or "").strip()
            if len(text) > 30:
                return text, f"llm:{vlm.name}", contexts
        except Exception:
            pass
    return template_report(verdict, dets, kb), "template", contexts
