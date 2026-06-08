"""Task 3: convert landing files into markdown."""

from __future__ import annotations

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

md_converter = MarkItDown()

LEGAL_EXTENSION_PRIORITY = {".pdf": 0, ".docx": 1, ".doc": 2, ".txt": 3}


def _convert_file_to_markdown(filepath: Path) -> str:
    """Convert source documents with Microsoft MarkItDown."""
    result = md_converter.convert(str(filepath))
    content = result.text_content.strip()
    if not content:
        raise ValueError(f"MarkItDown returned empty content for {filepath}")
    return content


def convert_legal_docs() -> list[Path]:
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    legal_dir = LANDING_DIR / "legal"
    if not legal_dir.exists():
        return saved

    source_files = [
        filepath
        for filepath in legal_dir.iterdir()
        if filepath.suffix.lower() in LEGAL_EXTENSION_PRIORITY
    ]
    preferred_files: dict[str, Path] = {}
    for filepath in sorted(source_files, key=lambda path: (path.stem, LEGAL_EXTENSION_PRIORITY[path.suffix.lower()])):
        preferred_files.setdefault(filepath.stem, filepath)

    for filepath in preferred_files.values():
        content = _convert_file_to_markdown(filepath)
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(f"# {filepath.stem}\n\n{content.strip()}\n", encoding="utf-8")
        saved.append(output_path)
    return saved


def convert_news_articles() -> list[Path]:
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    news_dir = LANDING_DIR / "news"
    if not news_dir.exists():
        return saved

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue
        data = json.loads(filepath.read_text(encoding="utf-8"))
        output_path = output_dir / f"{filepath.stem}.md"
        header = (
            f"# {data.get('title', filepath.stem)}\n\n"
            f"Source: {data.get('url', 'N/A')}\n\n"
            f"Crawled: {data.get('date_crawled', 'N/A')}\n\n"
        )
        output_path.write_text(header + data.get("content_markdown", data.get("content", "")), encoding="utf-8")
        saved.append(output_path)
    return saved


def convert_all() -> list[Path]:
    return convert_legal_docs() + convert_news_articles()


if __name__ == "__main__":
    for path in convert_all():
        print(path)
