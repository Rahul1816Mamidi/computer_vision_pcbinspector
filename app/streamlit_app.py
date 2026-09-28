"""Streamlit front-end. Talks to the FastAPI service (API_URL)."""
import base64
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
COLORS = {"PASS": "green", "REVIEW": "orange", "FAIL": "red"}

st.set_page_config(page_title="PCB Defect Inspector", layout="wide")
st.title("PCB Defect Inspector")
st.caption("YOLO detector, VLM second opinion for uncertain findings, RAG-grounded inspection report.")

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Detector confidence threshold", 0.05, 0.9, 0.25, 0.05)
    use_vlm = st.checkbox("VLM second opinion on uncertain detections", value=True)
    mode = st.radio("Report mode", ["template", "llm"], index=0,
                    help="'llm' asks the local model to write the report (slower on CPU); falls back to template.")
    st.divider()
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        if resp.status_code != 200:
            st.error(resp.json().get("detail", "API not ready"))
            st.stop()
        h = resp.json()
        st.success("API online")
        st.write({k: h[k] for k in ("model", "vlm_provider", "vlm_ready", "vlm_model", "rag_backend", "vlm_policy")})
        if h["vlm_provider"] == "ollama" and not h["vlm_ready"]:
            st.warning("Ollama model not ready yet (first start downloads it). The app still works without it.")
    except Exception:
        st.error(f"API not reachable at {API_URL}")

up = st.file_uploader("Upload a PCB image", type=["jpg", "jpeg", "png", "bmp"])
if up and st.button("Inspect", type="primary"):
    with st.spinner("Inspecting..."):
        try:
            r = requests.post(
                f"{API_URL}/inspect",
                params={"conf": conf, "use_vlm": use_vlm, "report_mode": mode},
                files={"file": (up.name, up.getvalue(), up.type or "image/jpeg")},
                timeout=600,
            )
            r.raise_for_status()
            res = r.json()
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    color = COLORS[res["verdict"]]
    st.markdown(f"## Verdict: :{color}[{res['verdict']}]")
    left, right = st.columns([3, 2])
    with left:
        if res.get("annotated_image_b64"):
            st.image(base64.b64decode(res["annotated_image_b64"]), caption="Detections (? = flagged for review)")
    with right:
        st.subheader("Report")
        st.write(res["report"])
        st.caption(f"report mode used: {res['report_mode_used']}")
        st.subheader("Timings (ms)")
        st.json(res["timings_ms"])
    st.subheader("Detections")
    if res["detections"]:
        rows = [{"label": d["label"], "confidence": d["confidence"], "size_px": d["size_px"],
                 "needs_review": d["needs_review"],
                 "vlm_verdict": (d["vlm"] or {}).get("verdict"), "vlm_reason": (d["vlm"] or {}).get("reason")}
                for d in res["detections"]]
        st.dataframe(rows)
    else:
        st.info("No defects detected.")
    with st.expander("Retrieved knowledge-base chunks (RAG)"):
        st.json(res["retrieved_context"])
