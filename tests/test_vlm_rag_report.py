import numpy as np

from src.config import get_settings
from src.inference.detector import Detection
from src.inference.rag import KnowledgeBase
from src.inference.report import decide_verdict, generate_report, template_report
from src.inference.vlm import BaseVLM, MockVLM, make_crop, parse_verdict, verify_crop, verification_prompt


def det(label="short", conf=0.9, cls_id=3):
    return Detection(label=label, cls_id=cls_id, conf=conf, xyxy=[10, 10, 40, 40])


def test_parse_verdict_variants():
    assert parse_verdict('{"verdict":"confirm","label":"short","reason":"ok"}')["verdict"] == "confirm"
    assert parse_verdict('Sure! {"verdict": "REJECT", "label": "none", "reason": "x"} thanks')["verdict"] == "reject"
    assert parse_verdict("no json here")["verdict"] == "unparsed"
    # relabel to an invalid class degrades to reject
    assert parse_verdict('{"verdict":"relabel","label":"banana","reason":""}')["verdict"] == "reject"
    r = parse_verdict('{"verdict":"relabel","label":"Mouse Bite","reason":""}')
    assert r["verdict"] == "relabel" and r["label"] == "mouse_bite"


def test_mock_vlm_confirms_and_prompt_contains_label():
    crop = np.zeros((50, 50, 3), np.uint8)
    out = verify_crop(MockVLM(), crop, "spur")
    assert out["verdict"] == "confirm" and out["label"] == "spur" and out["provider"] == "mock"
    assert "defect of type 'spur'" in verification_prompt("spur")


def test_verify_crop_handles_failures():
    class Broken(BaseVLM):
        name = "broken"

        def ask_image(self, image_bgr, prompt):
            raise TimeoutError("slow")

    out = verify_crop(Broken(), np.zeros((20, 20, 3), np.uint8), "short")
    assert out["verdict"] == "unavailable"


def test_make_crop_upscales_small_and_stays_in_bounds():
    img = np.zeros((300, 400, 3), np.uint8)
    c = make_crop(img, [390, 290, 399, 299])
    assert min(c.shape[:2]) >= 200 and max(c.shape[:2]) <= 512


def test_rag_tfidf_fallback_retrieves_right_doc():
    kb = KnowledgeBase(get_settings().kb_dir, use_embeddings=False)
    assert kb.backend == "tfidf" and len(kb.chunks) >= 20
    hits = kb.retrieve_for_label("open_circuit", k=3)
    assert hits and any(h["doc"] == "open_circuit" for h in hits)
    assert "Critical" in kb.section("open_circuit", "Severity")


def test_verdict_logic():
    assert decide_verdict([], 0.5) == "PASS"
    assert decide_verdict([det(conf=0.9)], 0.5) == "FAIL"
    weak = det(conf=0.3)
    assert decide_verdict([weak], 0.5) == "REVIEW"
    weak.vlm = {"verdict": "confirm"}
    assert decide_verdict([weak], 0.5) == "FAIL"
    weak.vlm = {"verdict": "reject"}
    assert decide_verdict([weak], 0.5) == "REVIEW"


def test_template_and_llm_report_paths():
    kb = KnowledgeBase(get_settings().kb_dir, use_embeddings=False)
    dets = [det("short", 0.9)]
    text, mode, ctx = generate_report("template", "FAIL", dets, kb, None)
    assert mode == "template" and "Recommended action" in text and ctx

    class FakeLLM(BaseVLM):
        name = "fake"

        def available(self):
            return True

        def ask_text(self, prompt):
            assert "DETECTIONS" in prompt and "CONTEXT" in prompt
            return "This board FAILS because a short circuit was detected; reject and check etching."

    text, mode, _ = generate_report("llm", "FAIL", dets, kb, FakeLLM())
    assert mode == "llm:fake" and "short circuit" in text

    class Down(FakeLLM):
        def ask_text(self, prompt):
            raise ConnectionError("ollama down")

    text, mode, _ = generate_report("llm", "FAIL", dets, kb, Down())
    assert mode == "template"  # graceful fallback
    assert "PASS" in template_report("PASS", [], kb)
