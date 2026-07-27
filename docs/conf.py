# Configuration file for the Sphinx documentation builder.
#
# Full reference: https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import pathlib
import re
import sys

# Put the repo root first on the path so autodoc imports the local package
# source rather than any same-named module elsewhere on the environment path.
sys.path.insert(0, os.path.abspath(".."))
# The docs dir itself, so the custom Pygments styles module is importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Register the denim Pygments styles (ported from the gedit color schemes) so
# they're available to the theme's pygments_light_style / pygments_dark_style.
# get_all_styles() reads STYLES; get_style_by_name() reads the name->module map.
from pygments.styles import STYLES, _STYLE_NAME_TO_MODULE_MAP  # noqa: E402

STYLES["DenimDarkStyle"] = ("_pygments_denim", "denimdark", ())
STYLES["DenimLightStyle"] = ("_pygments_denim", "denimlight", ())
_STYLE_NAME_TO_MODULE_MAP["denimdark"] = ("_pygments_denim", "DenimDarkStyle")
_STYLE_NAME_TO_MODULE_MAP["denimlight"] = ("_pygments_denim", "DenimLightStyle")

# -- Project information -----------------------------------------------------

project = "skyplothelper"
author = "Phil Cigan"
copyright = "2026, Phil Cigan"

# Read the version straight from the source so the docs never drift from
# skyplothelper/_version.py (installed metadata can be stale for editable
# installs, and importing the package here would pull in optional deps).
_version_src = (pathlib.Path(__file__).parent.parent / "skyplothelper" / "_version.py").read_text()
release = re.search(r'__version__\s*=\s*"([^"]+)"', _version_src).group(1)
version = ".".join(release.split(".")[:2])

# Which switcher.json entry to highlight as "current". On ReadTheDocs this is
# the version slug (e.g. "stable", "latest", "v1.0.0"); local builds map to
# "latest" so the dev entry is selected.
switcher_version = os.environ.get("READTHEDOCS_VERSION") or "latest"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",       # NumPy-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",               # Markdown (MyST) narrative pages
    "sphinx_design",             # grids / cards / tabs on the landing page
    "sphinx_copybutton",         # copy button on code blocks
    "nbsphinx",                  # render tutorial notebooks inline
]

# Generate the autosummary stub pages at build time.
autosummary_generate = True

# Document members in source order, keep the signature on the object's own line.
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"  # render annotations into the param list

# Optional third-party dependencies are mocked for the doc build so autodoc can
# import every module to read signatures without needing the heavy stack
# (cartopy/healpy in particular are awkward to pip-install on the docs runner).
autodoc_mock_imports = [
    "healpy",
    "cartopy",
    "astroquery",
    "reproject",
    "shapely",
    "scipy",
    "pysymlog",
    "plotly",
    "dash",
    "PIL",
]

# napoleon (NumPy docstring) settings.
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False
# Render docstring "Attributes" sections as :ivar: field entries rather than
# standalone attribute directives, so they don't collide with the same names
# documented by autodoc ``:members:`` (e.g. property-backed attributes).
napoleon_use_ivar = True

# Notebooks are executed elsewhere (or pre-rendered); don't run them on RTD.
nbsphinx_execute = "never"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints",
                    # gallery source manifest (parsed by make_features.py,
                    # not a doc page itself)
                    "features_manifest.md",
                    # manim scene-source README (an asset generator, not a doc
                    # page). Scoped to *.md so it doesn't also exclude the
                    # rendered videos under _static/manim (exclude_patterns is
                    # applied to the static tree too, relative to _static).
                    "manim/*.md"]

# MyST: enable a few useful extensions (colon-fence for directives, dollar math).
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

# Astropy writes |SkyCoord|, |QTable|, … in its docstrings — RST substitutions
# it defines in its *own* docs build (docs/common_links.txt), not something the
# package ships. Autodoc pulls those docstrings into our generated API pages
# (inherited members of TiltedEarthFrame and friends), where the substitutions
# are undefined and docutils raises "Undefined substitution referenced". Define
# them here so they resolve to the intended cross-references instead. Only
# visible on a *clean* build, since it's the regenerated api/generated/ stubs
# that carry them.
rst_epilog = """
.. |SkyCoord| replace:: :class:`~astropy.coordinates.SkyCoord`
.. |BaseFrame| replace:: :class:`~astropy.coordinates.BaseCoordinateFrame`
.. |Angle| replace:: :class:`~astropy.coordinates.Angle`
.. |SpectralCoord| replace:: :class:`~astropy.coordinates.SpectralCoord`
.. |EarthLocation| replace:: :class:`~astropy.coordinates.EarthLocation`
.. |Table| replace:: :class:`~astropy.table.Table`
.. |QTable| replace:: :class:`~astropy.table.QTable`
.. |Column| replace:: :class:`~astropy.table.Column`
.. |Quantity| replace:: :class:`~astropy.units.Quantity`
.. |Unit| replace:: :class:`~astropy.units.UnitBase`
.. |Time| replace:: :class:`~astropy.time.Time`
.. |ndarray| replace:: :class:`~numpy.ndarray`
"""

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
# Serve the agent-facing map + recipe corpus at the site root, per the llms.txt
# convention (agents fetch https://<site>/llms.txt). Both are generated from
# skyplothelper/_overview.py by scripts/make_llms_txt.py — regenerate them,
# never hand-edit (see .claude/DOCS_RESWEEP_PLAYBOOK.md).
html_extra_path = ["../llms.txt", "../llms-full.txt"]
html_css_files = ["custom.css"]
# fontawesome-config.js disables FontAwesome's document-wide MutationObserver,
# which otherwise re-scans the whole document on every DOM change and makes the
# plotly tutorial take 227 s instead of 4 s (see the file's comment). It only has
# to beat the theme's `defer`red fontawesome.js, which any classic script does;
# the low priority just puts it first among Sphinx's scripts.
html_js_files = [
    ("fontawesome-config.js", {"priority": 100}),
    "navbar-priority.js",
    "plot-theme.js",
    "lightbox.js",
]
html_title = "skyplothelper"

# Branding is deliberately minimal: a small navbar glyph + a favicon, no page
# banners. Both are the tightly-cropped chrome derivatives from
# docs/make_logo.py (the eye line mark and the HEALPix rosette). The navbar
# glyph is transparent with theme-appropriate ink — image_light (dark ink) on
# the light navbar, image_dark (white ink) on the dark navbar.
html_favicon = "_static/logo/logo_favicon.png"

# All appearance knobs live here (and in _static/custom.css for colors/fonts).
# See https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/
html_theme_options = {
    "logo": {
        "image_light": "_static/logo/logo_navbar-light.png",
        "image_dark": "_static/logo/logo_navbar-dark.png",
        "text": "skyplothelper",
    },
    "github_url": "https://github.com/pjcigan/skyplothelper",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/skyplothelper/",
            "icon": "fa-brands fa-python",
        },
    ],
    "use_edit_page_button": False,
    "navbar_align": "left",
    # Syntax highlighting ported from the gedit denim color schemes.
    "pygments_light_style": "denimlight",
    "pygments_dark_style": "denimdark",
    # Show many nav links on wide screens; the theme moves the overflow into a
    # "More" dropdown dynamically as the window narrows. Set high enough that
    # the API reference link stays visible on a typical desktop.
    "header_links_before_dropdown": 10,
    "show_toc_level": 2,
    "navigation_with_keys": True,
}

# Version switcher. It needs the hosted switcher.json plus more than one
# published version, so it only activates on ReadTheDocs; local builds omit it
# (otherwise the widget renders an empty "Choose version" button). On RTD it
# sits between the search box and the light/dark toggle. switcher.json lives in
# _static and is fetched from the stable "latest" URL so every version sees the
# current list.
if os.environ.get("READTHEDOCS") == "True":
    html_theme_options["switcher"] = {
        "json_url": "https://skyplothelper.readthedocs.io/en/latest/_static/switcher.json",
        "version_match": switcher_version,
    }
    html_theme_options["show_version_warning_banner"] = True
    html_theme_options["navbar_end"] = [
        "version-switcher",
        "theme-switcher",
        "navbar-icon-links",
    ]

html_context = {
    "github_user": "pjcigan",
    "github_repo": "skyplothelper",
    "github_version": "main",
    "doc_path": "docs",
}

# Left sidebar: the theme's in-section nav, plus a persistent API-reference
# section pinned below it (api-nav.html hides itself on API pages).
html_sidebars = {
    "**": ["sidebar-nav-bs", "api-nav"],
}

# -- Custom roles ------------------------------------------------------------

# matplotlib's inherited Artist docstrings use a ``:mpltype:`` role that only
# exists in matplotlib's own Sphinx build. Register a no-op so documenting
# matplotlib-derived classes (e.g. Beam) doesn't error.
from docutils import nodes  # noqa: E402
from docutils.parsers.rst import roles  # noqa: E402


def _mpltype_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    return [nodes.literal(rawtext, text)], []


roles.register_local_role("mpltype", _mpltype_role)
