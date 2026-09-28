# Architecture

## 1. Goal and constraints

Build a production-shaped computer-vision product in about 10 hours, with **only a free Kaggle GPU** and a CPU-only laptop. That drove the main decisions: train once on Kaggle, export to ONNX, serve on CPU in Docker, keep the VLM optional and local.

## 2. System overview

```
                     TRAINING (Kaggle GPU, offline)
 PKU-Market-PCB (VOC XML)
        │  prepare_dataset.py   (parse, clean, stratified 70/15/15, seed 42)
        ▼
 YOLO dataset ──► train.py (YOLOv8s, 1024px) ──► best.pt
        │                                           │
        │                                  export_onnx.py ──► best.onnx (+ model_meta.json)
        ▼                                           │
 evaluate.py  (mAP, IoU matching, per-class P/R,    │
 confusion matrix, size buckets, worst cases)       │
 evaluate_pipeline.py (does the VLM help?)          │
 benchmark.py (pt vs onnx latency)                  │
                                                    ▼
                     SERVING (Docker, CPU)
 image ─► FastAPI /inspect
            │
            ▼
       Detector (ONNX Runtime via Ultralytics)  ── detections + confidence
            │
            ├─ conf ≥ gate ───────────────────────────────┐
            │                                              │
            └─ conf < gate (max N per image)               │
                  crop + padding + upscale                 │
                  ▼                                        │
              VLM second opinion (Ollama qwen2.5vl:3b)     │
              confirm / reject / relabel  (JSON)           │
                  ▼                                        ▼
              policy: flag (default) or apply ──► final detections
                                                       │
                                        verdict: PASS / REVIEW / FAIL
                                                       │
                          RAG: retrieve defect knowledge chunks
                          (embeddings + class-metadata boost; TF-IDF fallback)
                                                       │
                          Report: LLM grounded on retrieved text
                                  (auto-fallback to deterministic template)
                                                       │
                     JSON response + annotated image ◄─┘
                     + metrics counters + predictions.jsonl + review_queue/
```

Containers (`docker-compose.yml`): `api` (FastAPI, port 8000), `ui` (Streamlit, 8501), `ollama` (local model server, 11434), `ollama-pull` (one-shot model download). The `ollama*` services sit behind the `ollama` compose profile so the app can also run without them.

## 3. Components

| Component | File | Responsibility |
|---|---|---|
| Config | `src/config.py` | one frozen settings object, env-overridable, read lazily so tests can override |
| Data prep | `src/data/prepare_dataset.py` | layout-agnostic scan, XML -> YOLO, class-name normalisation, leakage-safe filtering, stratified split, manifests |
| Trainer | `src/training/train.py` | Ultralytics fine-tuning with PCB-appropriate augmentation; writes weights + `model_meta.json` |
| Evaluator | `src/training/evaluate.py` | standard mAP plus our own IoU matcher for error analysis |
| Ablation | `src/training/evaluate_pipeline.py` | measures the VLM's real contribution (F1 with/without) |
| Optimizer | `src/optimization/*` | ONNX export, optional int8, latency benchmark |
| Detector | `src/inference/detector.py` | loads `.onnx` or `.pt`, returns typed `Detection` objects; lazy imports |
| VLM clients | `src/inference/vlm.py` | Ollama / Gemini / mock behind one interface, plain HTTP, robust JSON parsing |
| RAG | `src/inference/rag.py` | chunking by section, embedding retrieval, metadata boost, TF-IDF fallback |
| Report | `src/inference/report.py` | verdict rules, grounded LLM prompt, template fallback |
| Pipeline | `src/inference/pipeline.py` | orchestration, gating, timings, thread lock |
| API | `src/api/main.py` | validation, size limits, threadpool execution, typed responses |
| Monitoring | `src/api/monitoring.py` | in-memory metrics, JSONL prediction log, review-queue images |
| UI | `app/streamlit_app.py` | upload, settings, results, RAG transparency |

## 4. Key design decisions and trade-offs

**Two-stage cascade (detector first, VLM only when unsure).** A small detector is fast and cheap but overconfident on ambiguous cases; a VLM understands context but is slow on CPU. Running the VLM only on low-confidence detections (capped per image) gets its judgment where it matters without paying its latency on every defect. Trade-off: a wrong VLM opinion can hurt, so the effect is *measured* (`evaluate_pipeline.py`) and the default policy is `flag`, which never removes a detection.

**`flag` vs `apply` policy.** `flag` keeps every detection and marks disputed ones for human review (human-in-the-loop). `apply` lets the VLM drop or relabel. `apply` is only justified if the ablation shows an F1 gain.

**Resolution 1024 instead of 640.** PCB defects are a few dozen pixels on multi-megapixel boards; 640 px downsampling erases them. Costs more GPU time; cheap on a T4 for a dataset of this size. The evaluation reports recall by defect size to show if this is still insufficient (then: tiling).

**ONNX Runtime for serving.** No GPU on the target machine, so export once and run on CPU. Fixed input size for speed; `model_meta.json` carries the image size so the serving code can't drift from training. The benchmark script quantifies the speed-up instead of assuming it.

**Ollama in Docker, plain HTTP instead of vendor SDKs.** Free, offline after first pull, no rate limits. Because providers are plain `requests` calls behind `BaseVLM`, swapping to Gemini or another model is a config change and the project has no vendor lock-in. Model weights live in a named Docker volume, so they don't pollute the host and are removable with `docker compose down -v`.

**Graceful degradation everywhere.** Missing VLM -> detections still returned. LLM slow/failing/short answer -> deterministic template report from the same knowledge base. Embedding model unavailable -> TF-IDF retrieval. Missing weights -> API stays up and `/health` explains why. The service never returns a 500 because an optional component is down.

**RAG kept deliberately small.** About 30 chunks from 7 documents: an in-memory numpy index is enough (a vector database would be over-engineering). Retrieval is hybrid: semantic similarity plus a metadata boost for chunks belonging to the detected class. The LLM prompt says "use only the provided context" and receives the structured detections, which limits hallucination; the retrieved chunk ids are returned in the API response for transparency.

**Leakage-safe data handling.** Rotated/augmented copies of the same board are excluded by default so near-duplicates can't appear in both train and test; the split is stratified by dominant class and seeded; a manifest records exact membership.

**Thread safety.** The Ultralytics model is not guaranteed thread-safe, so inference runs behind a lock and off the event loop (`run_in_threadpool`). For higher throughput, run multiple uvicorn workers or replicas.

## 5. Evaluation strategy

1. **Detector quality:** mAP@0.5 and mAP@0.5:0.95 on the held-out test split (Ultralytics validator), with per-class AP.
2. **Operating-point quality:** our own IoU matcher at a fixed confidence gives TP/FP/FN, precision/recall per class, a confusion matrix including a background row/column (hallucinated vs missed defects), and recall by defect size.
3. **Worst-case inspection:** the most error-heavy test images are rendered with ground truth (green) and predictions (red) into `reports/error_cases/`.
4. **Component value:** the ablation compares detector-only vs detector + VLM variants on F1; the ONNX benchmark compares latency.
5. **Production feedback loop:** `logs/predictions.jsonl` records every request; REVIEW images are saved to `logs/review_queue/` so uncertain real-world images can be labelled and added to the next training run.

Everything is logged to `reports/experiments.jsonl` and rendered to `docs/EXPERIMENT_LOG.md`.

## 6. Failure modes and mitigations

| Failure mode | Effect | Mitigation |
|---|---|---|
| Tiny defects missed | low recall in the small-size bucket | higher `imgsz`; next step: tiled inference |
| Similar classes confused (spur vs mouse_bite) | off-diagonal confusion cells | more data / hard-negative mining; VLM `relabel` (only if the ablation approves) |
| Small VLM misjudges tiny crops | wrong reject/relabel | `flag` default; ablation gate; `VLM_MAX_CALLS` cap |
| VLM slow on CPU | latency | only uncertain detections; timeout; `template` report mode |
| Domain shift (real factory images) | precision drop | review queue -> relabel -> retrain |
| Oversized / corrupt uploads | crashes | size limit (413), decode check (400), typed validation (422) |
| Optional service down | degraded features | fallbacks listed in section 4 |

## 7. Security and operational notes

- Upload size is capped (`MAX_UPLOAD_MB`); images are decoded with OpenCV and never executed.
- The API has no authentication: put it behind a gateway before exposing it beyond localhost.
- The Gemini key (if used) is read from the environment and sent in a header, never logged or written to files.
- Logs store an image hash, not the image, except REVIEW images (switch off with `LOG_REVIEW_IMAGES=false`).

## 8. Future work

Tiled/SAHI inference for tiny defects; active learning from the review queue; calibrated int8 quantization or TensorRT on GPU hosts; a labelled subset to evaluate report quality; CI running `pytest` and a Docker build; Prometheus-format metrics.
