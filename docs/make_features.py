"""Generate the Feature Gallery from ``features_manifest.md``.

Renders every manifest entry twice — light and dark, in styles matched to
the docs palette — and writes the thumbnail PNGs plus the gallery pages:

- ``_static/features/<id>-{light,dark}.png``  (the figures)
- ``features/<id>.md``                         (detail page: figure + code)
- ``features/index.md``                        (categorized thumbnail grid)

Run locally (this is not part of the Sphinx build; outputs are committed):

    cd docs && python make_features.py

Figure styles come from the package's own annotation palettes —
``publication`` for light mode and ``denim`` for dark (whose warm
charcoal matches the docs dark palette in ``_static/custom.css``) — with
the dual-mode ``uranometria`` data-color cycle in both, so the gallery
doubles as a demo of skyplothelper's styling. The mode-aware display
(and the navbar plot-color toggle) is implemented in
``_static/plot-theme.js`` + the ``plot-light``/``plot-dark`` CSS rules
in ``custom.css``.
"""

from __future__ import annotations

import os
import re
import sys
import traceback
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
# Import the local package source, not any same-named module elsewhere on
# the environment path (same shadow-guard as conf.py).
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import skyplothelper as sph  # noqa: E402
IMG_DIR = DOCS / "_static" / "features"
PAGE_DIR = DOCS / "features"
MANIFEST = DOCS / "features_manifest.md"

DPI = 110

# Figure styles drawn from the package's own annotation palettes, so the
# gallery demos skyplothelper styling and stays in sync with the package.
# 'denim' is the warm charcoal that the docs dark palette is based on.
def _theme_from_annotation(name: str) -> dict:
    p = sph.ANNOTATION_PALETTES[name]
    return {
        "figure.facecolor": p["fig_bg"],
        "axes.facecolor": p["ax_bg"],
        "axes.edgecolor": p["frame"],
        "axes.labelcolor": p["text"],
        "xtick.color": p["frame"],
        "ytick.color": p["frame"],
        "text.color": p["text"],
        "grid.color": p["grid"],
        # In-house default image colormap; individual entries override it.
        "image.cmap": "sph.deepsky",
    }


MODES: dict[str, dict] = {
    "light": {"palette": "uranometria",
              "theme": _theme_from_annotation("publication")},
    "dark": {"palette": "uranometria",
             "theme": _theme_from_annotation("denim")},
}

DATA_NOTE = (
    "```{note}\nThis example uses a file from the repository's "
    "[`examples/data/`](https://github.com/pjcigan/skyplothelper/tree/main/"
    "examples/data) directory (not bundled with the pip install) — see the "
    "README there for provenance and credits.\n```\n"
)


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def parse_manifest(text: str) -> list[dict]:
    """Parse categories/entries out of the manifest markdown."""
    categories: list[dict] = []
    # Drop everything before the first category heading.
    parts = re.split(r"^## +", text, flags=re.M)[1:]
    for part in parts:
        cat_name, _, body = part.partition("\n")
        cat = {"name": cat_name.strip(), "entries": []}
        for chunk in re.split(r"^### +", body, flags=re.M)[1:]:
            title, _, rest = chunk.partition("\n")
            entry = {"title": title.strip(), "meta": {}, "prose": "", "code": ""}
            m = re.search(r"```python\n(.*?)```", rest, re.S)
            if not m:
                raise ValueError(f"No python block in entry {title!r}")
            entry["code"] = m.group(1).rstrip()
            head = rest[: m.start()]
            prose_lines = []
            for line in head.splitlines():
                meta = re.match(r"- (\w+):\s*(.+)", line)
                if meta:
                    entry["meta"][meta.group(1)] = meta.group(2).strip()
                elif line.strip():
                    prose_lines.append(line.rstrip())
            entry["prose"] = "\n".join(prose_lines)
            entry["id"] = entry["meta"].get("id", slugify(entry["title"]))
            cat["entries"].append(entry)
        if cat["entries"]:
            categories.append(cat)
    return categories


def render(entry: dict, mode: str) -> None:
    spec = MODES[mode]
    plt.close("all")
    plt.rcdefaults()
    sph.set_style(base="standard", theme=spec["theme"], palette=spec["palette"])
    ns: dict = {"__name__": "__sph_features__"}
    exec(compile(entry["code"], f"<{entry['id']}>", "exec"), ns)  # noqa: S102
    fig = ns.get("fig") or plt.gcf()
    out = IMG_DIR / f"{entry['id']}-{mode}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close("all")


def image_block(entry_id: str, mode: str, title: str, from_root: bool) -> str:
    path = f"/_static/features/{entry_id}-{mode}.png"
    return (f"```{{image}} {path}\n"
            f":class: sph-plot plot-{mode} dark-light\n"
            f":alt: {title} ({mode} mode)\n"
            "```\n\n")


def api_links(meta: dict) -> str:
    names = [n.strip() for n in meta.get("api", "").split(",") if n.strip()]
    return " · ".join(f"{{py:obj}}`~skyplothelper.{n}`" for n in names)


def write_detail_page(entry: dict) -> None:
    parts = [f"# {entry['title']}\n"]
    parts.append(image_block(entry["id"], "light", entry["title"], True))
    parts.append(image_block(entry["id"], "dark", entry["title"], True))
    if entry["prose"]:
        parts.append(entry["prose"] + "\n")
    refs = []
    if "guide" in entry["meta"]:
        refs.append(f"Guide: {{doc}}`/guide/{entry['meta']['guide']}`")
    links = api_links(entry["meta"])
    if links:
        refs.append(f"API: {links}")
    if refs:
        parts.append(" — ".join(refs) + "\n")
    parts.append("## Code\n")
    parts.append(f"```python\n{entry['code']}\n```\n")
    if entry["meta"].get("data") == "examples":
        parts.append(DATA_NOTE)
    (PAGE_DIR / f"{entry['id']}.md").write_text("\n".join(parts))


def write_index(categories: list[dict]) -> None:
    parts = ["""# Feature Gallery

A visual index of what skyplothelper draws. Click any figure for a page
with the full example code. Figures follow the site's light/dark mode by
default — the plot-colors button in the top bar (next to the site theme
toggle) overrides them independently, e.g. to preview publication-style
light figures while reading in dark mode.

"""]
    # One captioned toctree per category, so the left sidebar shows the
    # categories as headers with their plots nested beneath (a tree) rather
    # than one flat list of every plot.
    for cat in categories:
        parts.append(f"```{{toctree}}\n:hidden:\n:caption: {cat['name']}\n\n")
        for e in cat["entries"]:
            parts.append(f"{e['id']}\n")
        parts.append("```\n\n")
    for cat in categories:
        parts.append(f"\n## {cat['name']}\n\n")
        parts.append("::::{grid} 1 2 3 3\n:gutter: 3\n\n")
        for e in cat["entries"]:
            parts.append(f":::{{grid-item-card}}\n:link: {e['id']}\n"
                         ":link-type: doc\n:text-align: center\n\n")
            parts.append(image_block(e["id"], "light", e["title"], False))
            parts.append(image_block(e["id"], "dark", e["title"], False))
            parts.append(f"**{e['title']}**\n:::\n\n")
        parts.append("::::\n")
    (PAGE_DIR / "index.md").write_text("".join(parts))


def main() -> int:
    categories = parse_manifest(MANIFEST.read_text())
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)  # snippet paths are repo-root-relative
    n_ok = n_fail = 0
    for cat in categories:
        for entry in cat["entries"]:
            try:
                for mode in MODES:
                    render(entry, mode)
                write_detail_page(entry)
                n_ok += 1
                print(f"  ok   {entry['id']}")
            except Exception:
                n_fail += 1
                print(f"  FAIL {entry['id']}")
                traceback.print_exc()
    write_index(categories)
    print(f"\n{n_ok} rendered, {n_fail} failed; "
          f"index + {n_ok} detail pages written to {PAGE_DIR}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
