"""Task 10: RAG generation with citations."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve

load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

SYSTEM_PROMPT = """Answer in Vietnamese using only the provided context.
Every factual claim must include a citation in brackets.
If evidence is insufficient, say: Toi khong the xac minh thong tin nay tu nguon hien co."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Keep the best chunk first and move the second-best toward the end."""
    if len(chunks) <= 2:
        return chunks
    front = chunks[0::2]
    back = list(reversed(chunks[1::2]))
    return front + back


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"source-{i}")
        doc_type = metadata.get("type", "unknown")
        parts.append(f"[Document {i} | Source: {source} | Type: {doc_type}]\n{chunk.get('content', '')}")
    return "\n\n---\n\n".join(parts)


def _openrouter_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key or key == "sk-or-v1-xxx":
        raise RuntimeError("Missing OPENROUTER_API_KEY in .env")
    return key


def _build_user_prompt(query: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {query}"


def _generate_with_openrouter(query: str, context: str) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=_openrouter_api_key(),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    response = client.chat.completions.create(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(query, context)},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    answer = response.choices[0].message.content or ""
    if not answer:
        raise RuntimeError("OpenRouter returned an empty answer")
    return answer


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    answer = _generate_with_openrouter(query, context)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
    }


if __name__ == "__main__":
    print(generate_with_citation("Hinh phat tang tru ma tuy?")["answer"])
