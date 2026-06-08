"""Task 2: crawl Vietnamese news articles related to artists and drugs."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://lifestyle.znews.vn/chi-dan-an-tay-la-nhung-mat-xich-cuoi-trong-duong-day-ma-tuy-post1510988.html",
    "https://tuoitre.vn/nguoi-mau-nhikolai-dinh-bi-bat-trong-chuyen-an-ma-tuy-o-khu-ma-lang-quan-1-20240625230004986.htm",
    "https://tuoitre.vn/cong-an-dua-ca-si-chu-bin-ve-tru-so-de-lam-ro-hanh-vi-nghi-lien-quan-ma-tuy-20240606194450472.htm",
    "https://thanhnien.vn/ca-si-son-ngoc-minh-vua-bi-bat-vi-lien-quan-den-ma-tuy-la-ai-18526052012481811.htm",
    "https://tuoitre.vn/ma-tuy-tan-cong-nao-bo-cuoc-chiem-doat-tham-lang-ma-nhieu-ban-tre-dang-quen-20260512084442037.htm",
]


def setup_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _extract_title(html: str, fallback: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return fallback
    title = re.sub(r"\s+", " ", match.group(1)).strip()
    return re.sub(r"\s*[-|].*$", "", title) or fallback


def _html_to_text(html: str) -> str:
    html = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


async def crawl_article(url: str) -> dict:
    """Crawl one article and return metadata plus markdown-like content."""
    response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    html = response.text
    title = _extract_title(html, url.rsplit("/", 1)[-1])
    text = _html_to_text(html)
    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().date().isoformat(),
        "content_markdown": text[:4000],
    }


async def crawl_all():
    setup_directory()
    for i, url in enumerate(ARTICLE_URLS, 1):
        article = await crawl_article(url)
        filepath = DATA_DIR / f"article_{i:02d}.json"
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {filepath}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
