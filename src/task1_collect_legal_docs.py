"""Task 1: record legal-document sources for the RAG corpus."""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

LEGAL_SOURCES = [
    {
        "filename": "luat-phong-chong-ma-tuy-2021.doc",
        "title": "Luật Phòng, chống ma túy 2021",
        "source_url": "https://chinhphu.vn/?docid=204940&pageid=27160",
        "attachment_url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/01/73luat.pdf",
    },
    {
        "filename": "nghi-dinh-105-2021.doc",
        "title": "Nghị định 105/2021/NĐ-CP",
        "source_url": "https://congbao.chinhphu.vn/van-ban/nghi-dinh-so-105-2021-nd-cp-34944/37821.htm",
        "attachment_url": "https://congbao.chinhphu.vn/van-ban/nghi-dinh-so-105-2021-nd-cp-34944/37821.htm",
    },
    {
        "filename": "bo-luat-hinh-su-chuong-ma-tuy.doc",
        "title": "Bộ luật Hình sự 2015 sửa đổi 2017 - chương các tội phạm về ma túy",
        "source_url": "https://thuvienphapluat.vn/van-ban/Bo-may-hanh-chinh/Van-ban-hop-nhat-01-VBHN-VPQH-2017-Bo-luat-Hinh-su-363655.aspx",
        "attachment_url": "",
    },
]


def setup_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def list_collected_files() -> list[dict]:
    setup_directory()
    collected = []
    for source in LEGAL_SOURCES:
        path = DATA_DIR / source["filename"]
        collected.append({**source, "path": str(path), "exists": path.exists(), "size": path.stat().st_size if path.exists() else 0})
    return collected


if __name__ == "__main__":
    for item in list_collected_files():
        print(f"{item['filename']}: exists={item['exists']} size={item['size']} source={item['source_url']}")
