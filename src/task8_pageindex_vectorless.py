"""Task 8: PageIndex vectorless retrieval using the real PageIndex API."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

PROJECT_DIR = Path(__file__).parent.parent
LANDING_LEGAL_DIR = PROJECT_DIR / "data" / "landing" / "legal"
PAGEINDEX_DOCS_PATH = PROJECT_DIR / "data" / "pageindex_doc_ids.json"

API_BASE_URL = "https://api.pageindex.ai"
POLL_SECONDS = 5
MAX_WAIT_SECONDS = 300
CHAT_TIMEOUT_SECONDS = 300


def _api_key() -> str:
    key = os.getenv("PAGEINDEX_API_KEY", "").strip()
    if not key or key == "pi_xxx":
        raise RuntimeError("Missing PAGEINDEX_API_KEY in .env")
    return key


def _headers() -> dict[str, str]:
    return {"api_key": _api_key()}


def _raise_for_api_error(response: requests.Response, action: str) -> None:
    if response.ok:
        return
    detail = response.text[:500]
    raise RuntimeError(f"PageIndex {action} failed: HTTP {response.status_code}: {detail}")


def _load_doc_records() -> list[dict[str, str]]:
    if not PAGEINDEX_DOCS_PATH.exists():
        env_doc_ids = [doc_id.strip() for doc_id in os.getenv("PAGEINDEX_DOC_IDS", "").split(",") if doc_id.strip()]
        return [{"doc_id": doc_id, "source": doc_id} for doc_id in env_doc_ids]
    return json.loads(PAGEINDEX_DOCS_PATH.read_text(encoding="utf-8"))


def _save_doc_records(records: list[dict[str, str]]) -> None:
    PAGEINDEX_DOCS_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=True, indent=2))


def list_remote_documents(limit: int = 50) -> list[dict[str, Any]]:
    """List documents already uploaded to the PageIndex account."""
    response = requests.get(
        f"{API_BASE_URL}/docs",
        headers=_headers(),
        params={"limit": limit, "offset": 0},
        timeout=30,
    )
    _raise_for_api_error(response, "document list")
    return response.json().get("documents", [])


def sync_uploaded_documents() -> list[dict[str, str]]:
    """Persist completed/processing PageIndex document IDs from the remote account."""
    records = [
        {"doc_id": doc.get("id", ""), "source": doc.get("name", doc.get("id", ""))}
        for doc in list_remote_documents()
        if doc.get("id")
    ]
    if not records:
        raise RuntimeError("No PageIndex documents found in the remote account")
    _save_doc_records(records)
    return records


def _wait_until_retrieval_ready(doc_id: str) -> None:
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    while time.monotonic() < deadline:
        response = requests.get(
            f"{API_BASE_URL}/doc/{doc_id}/",
            headers=_headers(),
            params={"type": "tree"},
            timeout=30,
        )
        _raise_for_api_error(response, "document status check")
        data = response.json()
        if data.get("status") == "completed" and data.get("retrieval_ready"):
            return
        time.sleep(POLL_SECONDS)
    raise RuntimeError(f"PageIndex document {doc_id} was not retrieval-ready after {MAX_WAIT_SECONDS} seconds")


def upload_documents(file_paths: list[str] | None = None, wait: bool = True) -> list[dict[str, str]]:
    """Upload local PDF files to PageIndex and persist returned doc IDs."""
    paths = [Path(path) for path in file_paths] if file_paths else sorted(LANDING_LEGAL_DIR.glob("*.pdf"))
    if not paths:
        raise RuntimeError(f"No PDF files found to upload in {LANDING_LEGAL_DIR}")

    records = _load_doc_records()
    uploaded_sources = {record.get("source") for record in records}
    for path in paths:
        if path.name in uploaded_sources:
            continue
        with path.open("rb") as file:
            response = requests.post(
                f"{API_BASE_URL}/doc/",
                headers=_headers(),
                files={"file": file},
                timeout=120,
            )
        _raise_for_api_error(response, f"upload for {path.name}")
        doc_id = response.json().get("doc_id")
        if not doc_id:
            raise RuntimeError(f"PageIndex upload for {path.name} did not return doc_id")
        records.append({"doc_id": doc_id, "source": path.name})
        uploaded_sources.add(path.name)
        _save_doc_records(records)
        if wait:
            _wait_until_retrieval_ready(doc_id)

    _save_doc_records(records)
    return records


def check_uploaded_documents() -> list[dict[str, str]]:
    records = _load_doc_records()
    if not records:
        raise RuntimeError("No PageIndex doc IDs found. Run upload_documents() first.")
    for record in records:
        _wait_until_retrieval_ready(record["doc_id"])
    return records


def get_uploaded_document_statuses() -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for record in _load_doc_records():
        response = requests.get(
            f"{API_BASE_URL}/doc/{record['doc_id']}/",
            headers=_headers(),
            params={"type": "tree"},
            timeout=30,
        )
        _raise_for_api_error(response, "document status check")
        data = response.json()
        statuses.append(
            {
                "doc_id": record["doc_id"],
                "source": record.get("source"),
                "status": data.get("status"),
                "retrieval_ready": data.get("retrieval_ready"),
                "available_keys": sorted(data.keys()),
            }
        )
    return statuses


def _get_document_tree(doc_id: str) -> dict[str, Any]:
    response = requests.get(
        f"{API_BASE_URL}/doc/{doc_id}/",
        headers=_headers(),
        params={"type": "tree"},
        timeout=30,
    )
    _raise_for_api_error(response, "document tree fetch")
    return response.json()


def _walk_tree(nodes: list[dict[str, Any]], source: str, doc_id: str) -> list[dict]:
    flattened: list[dict] = []
    stack = list(nodes)
    while stack:
        node = stack.pop(0)
        title = node.get("title", "")
        text = node.get("text", "")
        content = "\n".join(part for part in [title, text] if part).strip()
        if content:
            flattened.append(
                {
                    "content": content,
                    "score": 0.0,
                    "metadata": {
                        "source": source,
                        "type": "pageindex_tree",
                        "doc_id": doc_id,
                        "node_id": node.get("node_id"),
                        "title": title,
                        "page_index": node.get("page_index"),
                    },
                    "source": "pageindex",
                }
            )
        stack.extend(node.get("nodes", []))
    return flattened


def _token_overlap_score(query: str, content: str) -> float:
    query_tokens = set(re.findall(r"\w+", query.lower(), flags=re.UNICODE))
    content_tokens = set(re.findall(r"\w+", content.lower(), flags=re.UNICODE))
    if not query_tokens:
        return 0.0
    return len(query_tokens & content_tokens) / len(query_tokens)


def _tree_search(query: str, records: list[dict[str, str]], top_k: int) -> list[dict]:
    results: list[dict] = []
    for record in records:
        tree = _get_document_tree(record["doc_id"])
        nodes = _walk_tree(tree.get("result", []), source=record.get("source", record["doc_id"]), doc_id=record["doc_id"])
        for node in nodes:
            node["score"] = _token_overlap_score(query, node["content"])
        results.extend(node for node in nodes if node["score"] > 0)
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _submit_retrieval(doc_id: str, query: str) -> str:
    response = requests.post(
        f"{API_BASE_URL}/retrieval/",
        headers=_headers(),
        json={"doc_id": doc_id, "query": query, "thinking": False},
        timeout=60,
    )
    _raise_for_api_error(response, f"retrieval submit for {doc_id}")
    retrieval_id = response.json().get("retrieval_id")
    if not retrieval_id:
        raise RuntimeError(f"PageIndex retrieval for {doc_id} did not return retrieval_id")
    return retrieval_id


def _wait_for_retrieval(retrieval_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    while time.monotonic() < deadline:
        response = requests.get(f"{API_BASE_URL}/retrieval/{retrieval_id}/", headers=_headers(), timeout=30)
        _raise_for_api_error(response, f"retrieval status for {retrieval_id}")
        data = response.json()
        if data.get("status") == "completed":
            return data
        time.sleep(POLL_SECONDS)
    raise RuntimeError(f"PageIndex retrieval {retrieval_id} did not complete after {MAX_WAIT_SECONDS} seconds")


def _flatten_retrieved_nodes(data: dict[str, Any], source: str) -> list[dict]:
    results: list[dict] = []
    for node_index, node in enumerate(data.get("retrieved_nodes", []), 1):
        title = node.get("title", "")
        node_id = node.get("node_id", "")
        contents = node.get("relevant_contents") or [{"relevant_content": node.get("text", ""), "page_index": None}]
        for content_index, item in enumerate(contents, 1):
            text = item.get("relevant_content") or item.get("text") or ""
            if not text.strip():
                continue
            results.append(
                {
                    "content": text.strip(),
                    "score": 1.0 / (len(results) + 1),
                    "metadata": {
                        "source": source,
                        "type": "pageindex",
                        "doc_id": data.get("doc_id"),
                        "node_id": node_id,
                        "title": title,
                        "page_index": item.get("page_index"),
                        "rank": f"{node_index}.{content_index}",
                    },
                    "source": "pageindex",
                }
            )
    return results


def _chat_retrieve(query: str, doc_ids: list[str], top_k: int) -> list[dict]:
    prompt = (
        "Retrieve the most relevant raw source excerpts for this question. "
        "Do not answer from general knowledge. Return concise Vietnamese excerpts "
        "grounded only in the selected PageIndex document(s), with citations when available.\n\n"
        f"Question: {query}"
    )
    response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        headers={**_headers(), "Content-Type": "application/json"},
        json={
            "doc_id": doc_ids if len(doc_ids) > 1 else doc_ids[0],
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0,
            "enable_citations": True,
        },
        timeout=CHAT_TIMEOUT_SECONDS,
    )
    _raise_for_api_error(response, "chat retrieval")
    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        return []
    content = choices[0].get("message", {}).get("content", "").strip()
    if not content:
        return []
    return [
        {
            "content": content,
            "score": 1.0,
            "metadata": {
                "source": "PageIndex Chat API",
                "type": "pageindex_chat",
                "doc_id": ",".join(doc_ids),
            },
            "source": "pageindex",
        }
    ][:top_k]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve evidence from uploaded PageIndex documents via the real API."""
    records = _load_doc_records()
    if not records:
        raise RuntimeError("No PageIndex doc IDs found. Run upload_documents() first.")

    if not all(_get_document_tree(record["doc_id"]).get("retrieval_ready") for record in records):
        return _tree_search(query, records, top_k=top_k)

    results: list[dict] = []
    for record in records:
        retrieval_id = _submit_retrieval(record["doc_id"], query)
        data = _wait_for_retrieval(retrieval_id)
        results.extend(_flatten_retrieved_nodes(data, source=record.get("source", record["doc_id"])))

    if not results:
        results = _chat_retrieve(query, [record["doc_id"] for record in records], top_k=top_k)

    return results[:top_k]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload PDFs or query PageIndex.")
    parser.add_argument("--upload", action="store_true", help="Upload PDFs from data/landing/legal to PageIndex.")
    parser.add_argument("--no-wait", action="store_true", help="Save doc IDs immediately without waiting for retrieval readiness.")
    parser.add_argument("--sync", action="store_true", help="Save doc IDs from documents already uploaded in PageIndex.")
    parser.add_argument("--check", action="store_true", help="Check whether saved PageIndex doc IDs are retrieval-ready.")
    parser.add_argument("--status", action="store_true", help="Show saved PageIndex document status fields.")
    parser.add_argument("--query", default="ma tuy", help="Query text for PageIndex retrieval.")
    args = parser.parse_args()

    if args.upload:
        _print_json(upload_documents(wait=not args.no_wait))
    elif args.sync:
        _print_json(sync_uploaded_documents())
    elif args.check:
        _print_json(check_uploaded_documents())
    elif args.status:
        _print_json(get_uploaded_document_statuses())
    else:
        _print_json(pageindex_search(args.query, top_k=3))
