"""Task 6: BM25-style lexical search."""

from __future__ import annotations

import math
import re
from collections import Counter

from .task4_chunking_indexing import get_chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def build_bm25_index(corpus: list[dict]):
    tokenized = [_tokenize(doc["content"]) for doc in corpus]
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))
    avgdl = sum(len(tokens) for tokens in tokenized) / max(len(tokenized), 1)
    return {"tokenized": tokenized, "doc_freq": doc_freq, "avgdl": avgdl, "n_docs": len(corpus)}


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Return chunks sorted by a local BM25 implementation."""
    corpus = get_chunks()
    if not corpus:
        return []

    index = build_bm25_index(corpus)
    query_tokens = _tokenize(query)
    k1 = 1.5
    b = 0.75
    scored: list[dict] = []

    for i, tokens in enumerate(index["tokenized"]):
        counts = Counter(tokens)
        doc_len = len(tokens) or 1
        score = 0.0
        for term in query_tokens:
            if term not in counts:
                continue
            df = index["doc_freq"].get(term, 0)
            idf = math.log(1 + (index["n_docs"] - df + 0.5) / (df + 0.5))
            tf = counts[term]
            denom = tf + k1 * (1 - b + b * doc_len / (index["avgdl"] or 1))
            score += idf * (tf * (k1 + 1)) / denom
        if score > 0:
            scored.append({"content": corpus[i]["content"], "score": float(score), "metadata": corpus[i].get("metadata", {})})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    for result in lexical_search("Dieu 248 ma tuy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
