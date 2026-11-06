"""Check every https link in the repo's Markdown files. Exits 1 if any look broken."""

from __future__ import annotations

import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).parents[1]
LINK = re.compile(r"\]\((https://[^)\s]+)\)")
HEADERS = {"User-Agent": "planagents-link-check/0.1"}


def check(url: str) -> tuple[str, str]:
    try:
        response = httpx.get(url, headers=HEADERS, follow_redirects=True, timeout=20)
        return url, str(response.status_code)
    except httpx.HTTPError as exc:
        return url, type(exc).__name__


def main() -> int:
    files = [p for p in ROOT.rglob("*.md") if ".venv" not in p.parts]
    urls = sorted({url for path in files for url in LINK.findall(path.read_text())})
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(check, urls))
    # Some sites refuse scripts while serving browsers fine, so these need a manual look.
    blocked = [(url, status) for url, status in results if status in {"401", "403", "429"}]
    broken = [(url, status) for url, status in results if status != "200" and (url, status) not in blocked]
    for url, status in blocked:
        print(f"{status}  {url}  (refused this script; check in a browser)")
    for url, status in broken:
        print(f"{status}  {url}  (BROKEN)")
    print(f"{len(urls)} links: {len(broken)} broken, {len(blocked)} refused the script")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
