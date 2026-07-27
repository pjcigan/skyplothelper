"""Generate dark-mode figure variants for the tutorial notebooks.

The normal Sphinx/nbsphinx build executes each tutorial notebook once, in the
default light ``publication`` style, and embeds those figures in the committed
``.ipynb`` (ReadTheDocs cannot re-execute them — heavy optional deps are mocked
and the example rasters are local-only). This script produces the *dark*
counterparts so the docs can offer a light/dark plot toggle on the tutorials,
the same way the plot-types gallery does.

It re-executes each notebook under a dark style and writes the resulting PNGs
to ``docs/_static/nb_dark/<stem>__<slug>.png``, where ``<slug>`` is the
slugified nearest-preceding markdown header (so the name follows the *content*,
not the position — reordering figures renames nothing), or an explicit
``# fig-slug: <name>`` magic comment in the figure's code cell when the header
isn't distinctive enough. Multiple figures under one slug get ``-2``/``-3``
suffixes. Alongside them it emits
a JSONP-style manifest ``<stem>.dark.js`` giving the slugs in figure-output
order::

    (window.__SPH_DARK = window.__SPH_DARK || {})["<stem>"] = ["slug-a", "slug-b", ...];

``_static/plot-theme.js`` loads that manifest via a ``<script>`` tag (not
fetch/XHR — Phil views local ``file://`` builds, which browsers forbid from
fetching ``file://`` resources) and pairs the Nth notebook output image with
``<stem>__<slugs[N]>.png``. The naming is thus decoupled from the volatile
``_images/...`` names Sphinx generates; adding/removing a figure changes only
its own file plus the one-line manifest, instead of churning every downstream
image. The JS falls back to the legacy ``<stem>_<i>.png`` scheme, so notebooks
migrate independently (regenerate one and it flips to slug names on its own).

Usage (run locally in the dev env, NOT on ReadTheDocs)::

    python docs/make_tutorial_dark_figs.py                # all tutorial notebooks
    python docs/make_tutorial_dark_figs.py getting_started # one (or several) by stem

Then rebuild the docs and commit the new ``_static/nb_dark/*.png``. Notebooks
with no committed dark figures simply fall back to their light figures in dark
mode (handled in JS), so this step is optional per notebook.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import re
import sys

import nbformat
from nbclient import NotebookClient

REPO = pathlib.Path(__file__).resolve().parent.parent
TUT = REPO / "docs" / "tutorials"
OUT = REPO / "docs" / "_static" / "nb_dark"

# The notebook kernel imports `skyplothelper` — make sure it's the local
# editable package and not a same-named module elsewhere on the environment
# path (a stale single-file shim on PYTHONPATH otherwise shadows it). Prepend
# the repo root so the kernel subprocess (which inherits this env) finds the
# package first regardless of its working directory.
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(REPO), os.environ.get("PYTHONPATH", "")]
).rstrip(os.pathsep)

# Injected as the first cell before re-execution. Applies a dark style and makes
# the figure background transparent so the saved PNGs sit cleanly on the docs'
# dark page surface (any color the theme uses), while the axes/text/strokes come
# from the dark theme. Keep this in sync with the look we want the dark toggle
# to show.
DARK_SETUP = """
# NB: do not force a non-interactive backend here — the notebook kernel's
# inline backend is what captures each plt.show() figure as a PNG output.
import matplotlib.pyplot as plt
import skyplothelper as sph

sph.set_style(theme="dark_sky", palette="nightcap")
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.transparent": True,
})
"""


def _slugify(text: str) -> str:
    """Slugify contract (shared with the JS consumer, which never re-derives):
    lowercase, non-alphanumeric runs collapse to a single '-', trim ends."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _last_header(md_source: str) -> str | None:
    """Text of the last markdown header line in a cell (nearest to a following
    figure), or None if the cell has no header."""
    last = None
    for line in md_source.splitlines():
        m = re.match(r"\s*#+\s+(.*\S)", line)
        if m:
            last = m.group(1).strip()
    return last


def _fig_slug_override(code_source: str) -> str | None:
    """A code cell may name its figure(s) explicitly with a magic comment
    ``# fig-slug: <name>`` (on any line), overriding the header-derived slug for
    that cell only. The name is slugified with the same contract, so
    ``# fig-slug: The Frame Itself`` and ``# fig-slug: the-frame-itself`` are
    equivalent. Useful for ambiguous sections, or several distinct figures under
    one header. Returns None when absent or empty (→ header slug).
    """
    m = re.search(r"^\s*#\s*fig-slug:\s*(.+?)\s*$", code_source, re.MULTILINE)
    return (_slugify(m.group(1)) or None) if m else None


def _figures_with_slugs(nb) -> list[tuple[str, str]]:
    """``(base64_png, slug)`` for every code-cell image output, in document order.

    Each figure's slug is the slugified *nearest preceding markdown header*, so
    the name follows the content (a reorder renames nothing — only the manifest
    order changes). Multiple figures under one header get ``-2``/``-3``/… suffixes
    in output order. The slug is decoupled from figure position, which is what
    lets a single figure regenerate without churning its neighbors.
    """
    out: list[tuple[str, str]] = []
    current = "figure"                       # fallback if a figure precedes any header
    counts: dict[str, int] = {}
    for cell in nb.cells:
        ct = cell.get("cell_type")
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        if ct == "markdown":
            header = _last_header(src)
            if header:
                current = _slugify(header) or "figure"
        elif ct == "code":
            # Per-cell override: `# fig-slug: <name>` wins over the header-derived
            # slug for THIS cell's figure(s) only (the next cell reverts to the
            # header). Does not change `current`. A cell emitting several figures
            # shares one base name → `<slug>`, `<slug>-2`, …; split it if each
            # needs a distinct name.
            base = _fig_slug_override(src) or current
            for o in cell.get("outputs", []):
                if "image/png" in o.get("data", {}):
                    n = counts.get(base, 0) + 1
                    counts[base] = n
                    slug = base if n == 1 else f"{base}-{n}"
                    out.append((o["data"]["image/png"], slug))
    return out


def generate(stem: str) -> int:
    nb_path = TUT / f"{stem}.ipynb"
    if not nb_path.exists():
        raise SystemExit(f"notebook not found: {nb_path}")
    nb = nbformat.read(nb_path, as_version=4)
    nb.cells.insert(0, nbformat.v4.new_code_cell(DARK_SETUP))

    # Execute from the notebook's own directory so relative data paths
    # (e.g. ``../../examples/data/...``) resolve exactly as they do in the
    # normal nbsphinx/jupytext build (which also runs from the notebook dir).
    client = NotebookClient(
        nb, kernel_name="python3", timeout=600,
        resources={"metadata": {"path": str(TUT)}},
    )
    client.execute()

    figs = _figures_with_slugs(nb)
    OUT.mkdir(parents=True, exist_ok=True)

    # Clear stale dark figs + manifest for this stem. Two precise globs (NOT
    # "{stem}_*.png") so a shared-prefix stem can't clobber another notebook's
    # files (e.g. "overlay" vs "overlay_grids"): legacy order-based
    # "<stem>_<int>.png", and new slug "<stem>__<slug>.png".
    for old in [*OUT.glob(f"{stem}_[0-9]*.png"), *OUT.glob(f"{stem}__*.png")]:
        old.unlink()
    (OUT / f"{stem}.dark.js").unlink(missing_ok=True)

    slugs = [slug for _, slug in figs]
    for b64, slug in figs:
        (OUT / f"{stem}__{slug}.png").write_bytes(base64.b64decode(b64))
    # Ordered slug manifest for _static/plot-theme.js. Shipped as JS (NOT .json)
    # so it loads over file:// local builds via a <script> tag — fetch()/XHR of
    # file:// is blocked. Slugs are final (suffixes already applied); the JS
    # treats them as opaque and indexes by figure order.
    manifest = (
        "(window.__SPH_DARK = window.__SPH_DARK || {})"
        f"[{json.dumps(stem)}] = {json.dumps(slugs)};\n"
    )
    (OUT / f"{stem}.dark.js").write_text(manifest, encoding="utf-8")

    print(f"{stem}: wrote {len(figs)} dark figure(s) + manifest to "
          f"{OUT.relative_to(REPO)}")
    return len(figs)


# Notebooks intentionally shipped light-only (their figures are style specimens;
# a theme-aware dark pass would override the default-vs-styled comparisons, and
# make panel titles dark-on-dark). plot-theme.js degrades to the light figure in
# every mode when a notebook's dark manifest is absent, so the site stays
# correct. Skipped unconditionally — even on an explicit stem arg — so the
# choice can't be undone by an accidental batch run.
_LIGHT_ONLY = {"styling"}


def main(argv: list[str]) -> None:
    stems = argv or sorted(p.stem for p in TUT.glob("*.ipynb"))
    for stem in stems:
        if stem in _LIGHT_ONLY:
            print(f"{stem}: skipped — light-only by design")
            continue
        generate(stem)


if __name__ == "__main__":
    main(sys.argv[1:])
