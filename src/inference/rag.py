"""Small retrieval-augmented-generation (RAG) knowledge base for defect guidance.

* Documents: knowledge_base/*.md, split into chunks by '## ' sections.
* Retrieval: sentence-transformers embeddings + cosine similarity, with a light
  metadata boost for chunks that belong to the detected defect class.
* Fallback: if the embedding model cannot be loaded (offline, no download), a
  pure-python TF-IDF retriever is used, so the app never fails because of RAG.
* The KB is tiny (about 30 chunks), so an in-memory numpy index is enough.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", text.lower().replace("_", " "))


def load_chunks(kb_dir: str) -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []
    for f in sorted(Path(kb_dir).glob("*.md")):
        title, section, buf = f.stem, "Overview", []

        def flush():
            body = " ".join(x.strip() for x in buf if x.strip())
            if body:
                chunks.append({"doc": f.stem, "title": title, "section": section, "body": body,
                               "text": f"{title} - {section}: {body}"})

        for line in f.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                flush()
                section, buf = line[3:].strip(), []
            else:
                buf.append(line)
        flush()
    return chunks


class KnowledgeBase:
    def __init__(self, kb_dir: str, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 use_embeddings: bool = True):
        self.chunks = load_chunks(kb_dir)
        self.backend = "tfidf"
        self._embedder = None
        self._matrix = None
        if use_embeddings and self.chunks:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(embed_model)
                self._matrix = self._embedder.encode(
                    [c["text"] for c in self.chunks], normalize_embeddings=True, show_progress_bar=False
                )
                self.backend = "embeddings"
            except Exception as e:  # offline / not installed
                print(f"[rag] embeddings unavailable ({type(e).__name__}); using TF-IDF fallback")
        self._build_tfidf()

    # -- TF-IDF fallback --------------------------------------------------
    def _build_tfidf(self) -> None:
        docs = [_tokens(c["text"]) for c in self.chunks]
        df = Counter(t for d in docs for t in set(d))
        n = max(len(docs), 1)
        self._idf = {t: math.log((1 + n) / (1 + v)) + 1 for t, v in df.items()}
        self._doc_vecs = [self._vec(d) for d in docs]

    def _vec(self, toks: List[str]) -> Dict[str, float]:
        tf = Counter(toks)
        v = {t: c * self._idf.get(t, 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    # -- search --------------------------------------------------------------
    def search(self, query: str, k: int = 4, defect: Optional[str] = None, boost: float = 0.15) -> List[Dict]:
        if not self.chunks:
            return []
        if self.backend == "embeddings":
            q = self._embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            scores = [float(x) for x in (self._matrix @ q)]
        else:
            qv = self._vec(_tokens(query))
            scores = [sum(w * dv.get(t, 0.0) for t, w in qv.items()) for dv in self._doc_vecs]
        if defect:
            scores = [s + (boost if c["doc"] == defect else 0.0) for s, c in zip(scores, self.chunks)]
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [{**self.chunks[i], "score": round(scores[i], 4)} for i in top]

    def section(self, doc: str, section: str) -> Optional[str]:
        for c in self.chunks:
            if c["doc"] == doc and c["section"].lower().startswith(section.lower()):
                return c["body"]
        return None

    def retrieve_for_label(self, label: str, k: int = 4) -> List[Dict]:
        q = f"{label.replace('_', ' ')} defect on a PCB: causes, severity and recommended action"
        return self.search(q, k=k, defect=label)
