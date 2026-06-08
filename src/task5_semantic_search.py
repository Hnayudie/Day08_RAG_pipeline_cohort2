"""Task 5: semantic search over the local chunk index."""

from __future__ import annotations

import math

from .task4_chunking_indexing import get_chunks, text_to_embedding


def _cosine(a: list[float], b: list[float]) -> float:
    denom = (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0
    return sum(x * y for x, y in zip(a, b)) / denom


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Return top chunks by cosine similarity."""
    query_embedding = text_to_embedding(query)
    results: list[dict] = []
    for chunk in get_chunks(with_embeddings=True):
        score = _cosine(query_embedding, chunk.get("embedding", []))
        results.append(
            {
                "content": chunk["content"],
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in semantic_search("hinh phat ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
