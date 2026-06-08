"""Task 4: Chunking and local indexing for the Day 8 RAG pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
INDEX_PATH = PROJECT_DIR / "data" / "vector_index.json"

# Recursive paragraph chunking is robust for mixed legal markdown and news.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
CHUNKING_METHOD = "recursive"

# User-provided Vietnamese legal embedding model:
# https://huggingface.co/Quockhanh05/Vietnam_legal_embeddings
EMBEDDING_MODEL = "Quockhanh05/Vietnam_legal_embeddings"
EMBEDDING_DIM = 768
VECTOR_STORE = "local_json"


def load_documents() -> list[dict]:
    """Read all markdown files from data/standardized."""
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news" if "news" in md_file.parts else "unknown"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": str(md_file.relative_to(STANDARDIZED_DIR)),
                    "type": doc_type,
                },
            }
        )
    return documents


def _recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        while len(paragraph) > chunk_size:
            chunks.append(paragraph[:chunk_size].strip())
            paragraph = paragraph[max(0, chunk_size - chunk_overlap):].strip()
        current = paragraph

    if current:
        chunks.append(current)
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents into bounded chunks with metadata preserved."""
    chunks: list[dict] = []
    for doc in documents:
        for i, chunk_text in enumerate(_recursive_split(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)):
            chunks.append({"content": chunk_text, "metadata": {**doc["metadata"], "chunk_index": i}})
    return chunks


def text_to_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Deterministic hashed embedding fallback for offline tests."""
    vector = [0.0] * dim
    for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        vector[idx] += 1.0 if digest[4] % 2 == 0 else -1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed chunks with the HF model when available; otherwise use a local fallback."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = model.encode([c["content"] for c in chunks], show_progress_bar=False)
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb.tolist()
    except Exception:
        for chunk in chunks:
            chunk["embedding"] = text_to_embedding(chunk["content"])
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> Path:
    """Persist the local vector index as JSON for repeatable offline retrieval."""
    INDEX_PATH.write_text(
        json.dumps({"model": EMBEDDING_MODEL, "dim": EMBEDDING_DIM, "chunks": chunks}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return INDEX_PATH


def get_chunks(with_embeddings: bool = False) -> list[dict]:
    """Load indexed chunks, or build them from markdown files on demand."""
    if INDEX_PATH.exists():
        data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        chunks = data.get("chunks", [])
        if with_embeddings and chunks and "embedding" not in chunks[0]:
            return embed_chunks(chunks)
        return chunks
    chunks = chunk_documents(load_documents())
    return embed_chunks(chunks) if with_embeddings else chunks


def run_pipeline() -> Path:
    docs = load_documents()
    chunks = embed_chunks(chunk_documents(docs))
    return index_to_vectorstore(chunks)


if __name__ == "__main__":
    path = run_pipeline()
    print(f"Indexed documents to {path}")
