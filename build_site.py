"""Build the website as static HTML files in docs/, for GitHub Pages.

Usage:
    python build_site.py

Then commit and push. GitHub Pages serves the docs/ folder at
https://emrozin.github.io/epl-probabilities/

Pages are found by following links, starting from the home page, so a new page is
included automatically as long as another page links to it.
"""

import os
import re
import shutil
import time
from pathlib import Path

BASE = "/epl-probabilities"   # the repository name: GitHub Pages serves the site under it
os.environ["SITE_BASE"] = BASE

from fastapi.testclient import TestClient  # noqa: E402  (must import after SITE_BASE is set)

import app as site  # noqa: E402

OUT = Path("docs")
# Internal links: href="..." or a season picker's option value="...", starting with the base path.
LINK = re.compile(r'(?:href|value)="' + re.escape(BASE) + r'(/[^"#?]*)"')


def save(path: str, html: str) -> None:
    """'/results/2026-09-21/' -> docs/results/2026-09-21/index.html"""
    target = OUT / path.strip("/") / "index.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")


def main() -> None:
    started = time.time()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    client = TestClient(site.app)

    queue, seen, built, broken = ["/"], {"/"}, 0, []
    while queue:
        path = queue.pop()
        response = client.get(path)
        if response.status_code != 200:
            broken.append(f"{path} ({response.status_code})")
            continue
        save(path, response.text)
        built += 1
        for link in LINK.findall(response.text):
            if not link.startswith("/static/") and link not in seen:
                seen.add(link)
                queue.append(link)

    # GitHub Pages shows 404.html for any address that doesn't exist.
    (OUT / "404.html").write_text(client.get("/this-page-does-not-exist/").text, encoding="utf-8")
    shutil.copytree("static", OUT / "static")
    (OUT / ".nojekyll").touch()  # serve files exactly as they are, without GitHub's Jekyll processing

    print(f"Built {built} pages into {OUT}/ in {time.time() - started:.0f} seconds.")
    if broken:
        print(f"\n{len(broken)} linked page(s) failed to build:")
        for line in broken:
            print("  " + line)


if __name__ == "__main__":
    main()
