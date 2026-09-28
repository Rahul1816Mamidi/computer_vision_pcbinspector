"""Vision-language / LLM clients behind one small interface.

No vendor SDKs: every provider is a plain HTTP call made with `requests`.
  * ollama : local model served by the Ollama container (default, free, offline)
  * gemini : Google AI Studio REST API (optional, needs GEMINI_API_KEY)
  * mock   : instant deterministic answers for tests / demos without a model
  * none   : VLM disabled
"""
from __future__ import annotations

import base64
import json
import re
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests

from src.config import CLASS_NAMES, Settings

VERDICTS = {"confirm", "reject", "relabel"}

DEFECT_GLOSSARY = (
    "missing_hole (a drilled hole/pad that is absent), "
    "mouse_bite (small bites or notches along a copper trace edge), "
    "open_circuit (a copper trace that is broken), "
    "short (unwanted copper bridging two traces or pads), "
    "spur (a thin spike of copper sticking out of a trace), "
    "spurious_copper (an isolated leftover blob of copper)"
)


def encode_jpeg_b64(img_bgr: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class BaseVLM:
    name = "base"

    def available(self) -> bool:  # pragma: no cover - interface
        return False

    def ask_image(self, image_bgr: np.ndarray, prompt: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def ask_text(self, prompt: str) -> str:  # pragma: no cover
        raise NotImplementedError


class OllamaVLM(BaseVLM):
    name = "ollama"

    def __init__(self, host: str, model: str, timeout: float):
        self.host, self.model, self.timeout = host, model, timeout
        self._avail_cache = (0.0, False)

    def available(self) -> bool:
        """True if the server answers AND the model has been pulled. Cached for 10 s."""
        ts, val = self._avail_cache
        if time.time() - ts < 10:
            return val
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            names = [m.get("name", "") for m in r.json().get("models", [])]
            base = self.model.split(":")[0]
            val = any(n == self.model or n.startswith(self.model + ":") or n.split(":")[0] == base and ":" not in self.model for n in names)
        except Exception:
            val = False
        self._avail_cache = (time.time(), val)
        return val

    def _chat(self, content: str, images: Optional[List[str]], want_json: bool, max_tokens: int) -> str:
        msg: Dict[str, Any] = {"role": "user", "content": content}
        if images:
            msg["images"] = images
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [msg],
            "stream": False,
            "options": {"temperature": 0, "num_predict": max_tokens},
        }
        if want_json:
            payload["format"] = "json"
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def ask_image(self, image_bgr: np.ndarray, prompt: str) -> str:
        return self._chat(prompt, [encode_jpeg_b64(image_bgr)], want_json=True, max_tokens=160)

    def ask_text(self, prompt: str) -> str:
        return self._chat(prompt, None, want_json=False, max_tokens=350)


class GeminiVLM(BaseVLM):
    name = "gemini"

    def __init__(self, api_key: str, model: str, timeout: float):
        self.api_key, self.model, self.timeout = api_key, model, timeout

    def available(self) -> bool:
        return bool(self.api_key)

    def _generate(self, parts: List[Dict[str, Any]], want_json: bool, max_tokens: int) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        cfg: Dict[str, Any] = {"temperature": 0, "maxOutputTokens": max_tokens}
        if want_json:
            cfg["responseMimeType"] = "application/json"
        r = requests.post(
            url,
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json={"contents": [{"parts": parts}], "generationConfig": cfg},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]

    def ask_image(self, image_bgr: np.ndarray, prompt: str) -> str:
        parts = [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": encode_jpeg_b64(image_bgr)}}]
        return self._generate(parts, True, 200)

    def ask_text(self, prompt: str) -> str:
        return self._generate([{"text": prompt}], False, 400)


class MockVLM(BaseVLM):
    """Agrees with the detector. Text generation returns '' so the pipeline uses the template report."""

    name = "mock"

    def available(self) -> bool:
        return True

    def ask_image(self, image_bgr: np.ndarray, prompt: str) -> str:
        m = re.search(r"defect of type '([a-z_]+)'", prompt)
        label = m.group(1) if m else "none"
        return json.dumps({"verdict": "confirm", "label": label, "reason": "mock provider"})

    def ask_text(self, prompt: str) -> str:
        return ""


def build_vlm(s: Settings) -> Optional[BaseVLM]:
    p = s.vlm_provider
    if p == "ollama":
        return OllamaVLM(s.ollama_host, s.ollama_model, s.vlm_timeout)
    if p == "gemini":
        return GeminiVLM(s.gemini_api_key, s.gemini_model, s.vlm_timeout)
    if p == "mock":
        return MockVLM()
    return None


# ----------------------------------------------------------------------------
# Detection verification
# ----------------------------------------------------------------------------
def make_crop(img_bgr: np.ndarray, xyxy, pad: float = 0.6, min_side: int = 224, max_side: int = 512) -> np.ndarray:
    """Crop around a detection with context padding and upscale tiny crops."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh) * (1 + 2 * pad)
    side = max(side, 48)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ax1, ay1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    ax2, ay2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    crop = img_bgr[ay1:ay2, ax1:ax2]
    if crop.size == 0:
        crop = img_bgr
    ch, cw = crop.shape[:2]
    scale = 1.0
    if min(ch, cw) < min_side:
        scale = min_side / min(ch, cw)
    if max(ch, cw) * scale > max_side:
        scale = max_side / max(ch, cw)
    if abs(scale - 1.0) > 1e-3:
        crop = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))), interpolation=cv2.INTER_CUBIC)
    return crop


def verification_prompt(label: str) -> str:
    return (
        "You are a quality inspector for bare printed circuit boards (PCBs). "
        "The image is a zoomed crop of a board. An automatic detector says it contains a "
        f"defect of type '{label}'.\n"
        f"Defect types: {DEFECT_GLOSSARY}.\n"
        "Decide whether the detector is right. Reply with JSON only, in this exact form: "
        '{"verdict": "confirm" | "reject" | "relabel", '
        '"label": "<one defect type, or none>", "reason": "<max 15 words>"}'
    )


def parse_verdict(raw: str) -> Dict[str, Any]:
    """Robustly parse the model's JSON answer. Never raises."""
    m = re.search(r"\{.*\}", raw or "", flags=re.S)
    if not m:
        return {"verdict": "unparsed", "label": None, "reason": (raw or "")[:80]}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"verdict": "unparsed", "label": None, "reason": (raw or "")[:80]}
    verdict = str(d.get("verdict", "")).strip().lower()
    label = str(d.get("label", "")).strip().lower().replace(" ", "_").replace("-", "_")
    if verdict not in VERDICTS:
        return {"verdict": "unparsed", "label": None, "reason": str(d)[:80]}
    if verdict == "relabel" and label not in CLASS_NAMES:
        verdict = "reject"  # model claims a defect but names no valid class -> treat as disagreement
    return {"verdict": verdict, "label": label if label in CLASS_NAMES else None,
            "reason": str(d.get("reason", ""))[:120]}


def verify_crop(vlm: BaseVLM, crop_bgr: np.ndarray, label: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        raw = vlm.ask_image(crop_bgr, verification_prompt(label))
        out = parse_verdict(raw)
    except Exception as e:  # network, timeout, model not ready ...
        out = {"verdict": "unavailable", "label": None, "reason": f"{type(e).__name__}"}
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    out["provider"] = vlm.name
    return out
