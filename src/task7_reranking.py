"""Task 7: reranking helpers."""

from __future__ import annotations

import re


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Offline lexical-overlap reranker that mimics cross-encoder re-scoring."""
    query_tokens = _tokens(query)
    reranked: list[dict] = []
    for candidate in candidates:
        doc_tokens = _tokens(candidate.get("content", ""))
        overlap = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
        base = float(candidate.get("score", 0.0))
        item = candidate.copy()
        item["score"] = 0.7 * overlap + 0.3 * base
        reranked.append(item)
    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def rerank_mmr(query_embedding: list[float], candidates: list[dict], top_k: int = 5, lambda_param: float = 0.7) -> list[dict]:
    return sorted(candidates, key=lambda item: item.get("score", 0), reverse=True)[:top_k]


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item.get("content", "")
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items[key] = item
    results = []
    for key, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]:
        merged = items[key].copy()
        merged["score"] = float(score)
        results.append(merged)
    return results


def rerank(query: str, candidates: list[dict], top_k: int = 5, method: str = "cross_encoder") -> list[dict]:
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    return rerank_cross_encoder(query, candidates, top_k=top_k)


if __name__ == "__main__":
    sample = [{"content": "Dieu 248 ma tuy", "score": 0.5, "metadata": {}}]
    print(rerank("hinh phat ma tuy", sample))
