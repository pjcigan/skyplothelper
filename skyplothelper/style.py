"""Themes and matplotlib RC param presets.

``set_theme`` applies a coordinated color theme; ``set_base_style`` is a richer
publication-quality preset; ``style_context`` is a context manager that
temporarily applies a preset.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Any, cast

import matplotlib.pyplot as plt  # noqa: F401  (used by some _THEMES values)
from cycler import cycler
from matplotlib import rcParams
from matplotlib import style as mpl_style

from ._compat import coord_ticks


def _rc_update(params: dict[str, Any]) -> None:
    """Apply a plain ``{name: value}`` dict to the global ``rcParams``.

    matplotlib 3.11 types ``RcParams`` keys as a ``Literal`` of every valid rc
    name, so a ``dict[str, Any]`` no longer satisfies ``update``. Casting to
    ``Any`` keeps this version-neutral: on the baseline (mpl 3.9) stubs the dict
    is accepted directly, so a ``# type: ignore`` here would be flagged unused.
    """
    rcParams.update(cast("Any", params))


class style_context:
    """
    Context manager for temporarily applying matplotlib RC param presets.

    More composable than ``set_base_style()`` — automatically restores previous
    settings on exit.

    Parameters
    ----------
    style : str or dict
        Legacy single-preset form: a set_base_style style name, or a dict of
        RC params. Ignored when any of *base* / *theme* / *palette* /
        rc overrides is given (those select the composable form below).
    base, theme, palette, font : str, dict, or list, optional
        The composable style layers (see :func:`set_style`); each accepts
        the same preset names or custom dict/list inputs as its setter. When
        any is given, the block uses ``set_style(base=, theme=, palette=,
        font=, **rc_overrides)`` instead of the legacy *style* preset.
    **rc_overrides
        Extra rcParams applied last (composable form only).

    Examples
    --------
    >>> with sph.style_context('standard'):
    ...     fig, ax = sph.allsky_figure()
    ...     ax.scatter(...)
    ...     # RC params restored after this block
    >>> with sph.style_context(base='journal', theme='dark_sky',
    ...                        palette='nightcap', font='journal'):
    ...     fig, ax = sph.allsky_figure()
    """

    def __init__(self, style: str | dict[str, Any] = 'standard', *,
                 base: str | dict[str, Any] | None = None,
                 theme: str | dict[str, Any] | None = None,
                 palette: str | list[str] | None = None,
                 font: str | list[str] | dict[str, Any] | None = None,
                 **rc_overrides: Any) -> None:
        self.style = style
        self.base = base
        self.theme = theme
        self.palette = palette
        self.font = font
        self.rc_overrides = rc_overrides
        # Composable form whenever any layer kwarg (or rc override) is set;
        # otherwise fall back to the legacy single-preset behavior.
        self._use_triple = (base is not None or theme is not None
                            or palette is not None or font is not None
                            or bool(rc_overrides))
        self._old_params: dict[str, Any] | None = None

    def __enter__(self) -> style_context:
        self._old_params = dict(cast("Any", rcParams))
        if self._use_triple:
            set_style(base=self.base, theme=self.theme,
                      palette=self.palette, font=self.font,
                      **self.rc_overrides)
        elif isinstance(self.style, dict):
            _rc_update(self.style)
        else:
            set_base_style(self.style)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._old_params is not None:
            _rc_update(self._old_params)



# ---- Base structural presets ------------------------------------------
#
# Purpose-specific structural rcParam presets dispatched by
# :func:`set_base_style`. The refinements that read as "polished" are mostly
# line-weight hierarchy and chartjunk removal, not color: thin frame/ticks,
# a hairline grid drawn below the data, inward minor ticks, generous padding.
# Fonts are *stacks* (never a bare family) so the look degrades gracefully
# (TeX Gyre -> Nimbus/Liberation -> DejaVu) instead of silently falling back
# to DejaVu on machines lacking the first choice; cm/stixsans math is paired
# to the text font.

_FONT_SERIF = ['TeX Gyre Termes', 'Liberation Serif', 'Times New Roman',
               'DejaVu Serif']
# Membership constraint, not just taste: matplotlib picks ONE installed family
# for a whole string — there is no per-glyph fallback — so the DejaVu backstop
# at the end is unreachable whenever an earlier face is installed. Every entry
# must therefore carry the prime marks (′ ″), which sph emits in every
# arcmin/arcsec label.
#
# Carlito was dropped for that reason (no U+2032; it led the stack, so ordinary
# Linux boxes rendered every arcmin label as a tofu box). It is not replaced
# like for like: Carlito is Calibri-metric, and Calibri itself has no U+2032,
# so no Calibri-metric face can satisfy this. Nimbus Sans and FreeSans are
# Helvetica-metric, which is what the leading TeX Gyre Heros actually is — so
# the stack is now metrically consistent as well as glyph-complete.
# Ordered so each common platform reaches a Helvetica-metric face before the
# DejaVu backstop: TeX Gyre Heros (a TeX install), Nimbus Sans (ghostscript),
# Liberation Sans (most Linux), Arial (macOS / Windows, and Helvetica-metric).
# _coerce_font_stack enforces glyph coverage at apply time, so this ordering is
# about *look*, not correctness.
_FONT_SANS = ['TeX Gyre Heros', 'Nimbus Sans', 'Liberation Sans', 'Arial',
              'Helvetica', 'FreeSans', 'DejaVu Sans']
_FONT_MONO = ['IBM Plex Mono', 'DejaVu Sans Mono', 'monospace']

# Default cycles baked into the refined presets. set_palette still overrides,
# since the palette layer is applied after base in set_style.
_URANO = ['#46618A', '#B97C52', '#5E8C7E', '#8A4540', '#C29B3C', '#716A8E']
_SPEAK = ['#B98E3E', '#A35D4C', '#41736B', '#67809F', '#8A8B57', '#96718F']


def _core() -> dict[str, Any]:
    """Structural rcParams shared by every refined (non-agnostic) preset."""
    return {
        'text.usetex': False,
        'mathtext.fontset': 'cm',
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'xtick.top': True, 'ytick.right': True,
        'axes.axisbelow': True,
        'axes.formatter.use_mathtext': True,
        'axes.formatter.limits': [-4, 4],
        'path.simplify': True, 'path.simplify_threshold': 0.5,
        'agg.path.chunksize': 10000,
        'image.cmap': 'magma', 'image.interpolation': 'antialiased',
        'image.origin': 'lower',
        'contour.negative_linestyle': 'dashed',
        'legend.numpoints': 1, 'legend.scatterpoints': 1,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.prop_cycle': cycler(color=_URANO),
    }


def _merge(**kw: Any) -> dict[str, Any]:
    d = _core()
    d.update(kw)
    return d


BASE_PRESETS: dict[str, dict[str, Any]] = {

    'journal': _merge(
        **{'font.family': 'serif', 'font.serif': _FONT_SERIF,
           'font.size': 9.5, 'axes.titlesize': 10.5, 'axes.labelsize': 10,
           'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
           'legend.fontsize': 8.5,
           'axes.linewidth': 0.6,
           'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
           'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
           'xtick.major.size': 4, 'ytick.major.size': 4,
           'xtick.minor.size': 2.2, 'ytick.minor.size': 2.2,
           'lines.linewidth': 1.2, 'lines.markersize': 4,
           'patch.linewidth': 0.7,
           'grid.linewidth': 0.4, 'grid.alpha': 0.45,
           'grid.linestyle': '-', 'grid.color': '#BBBBBB',
           'legend.frameon': False, 'legend.handlelength': 1.6,
           'legend.handletextpad': 0.5, 'legend.borderpad': 0.3,
           'axes.labelpad': 3.5, 'axes.titlepad': 7,
           'errorbar.capsize': 2, 'savefig.dpi': 600}),

    'press': _merge(
        **{'font.family': 'sans-serif', 'font.sans-serif': _FONT_SANS,
           'mathtext.fontset': 'stixsans',
           'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12.5,
           'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
           'legend.fontsize': 10.5,
           'axes.linewidth': 0.9,
           'xtick.major.width': 0.9, 'ytick.major.width': 0.9,
           'xtick.minor.width': 0.7, 'ytick.minor.width': 0.7,
           'xtick.major.size': 5, 'ytick.major.size': 5,
           'xtick.minor.size': 2.8, 'ytick.minor.size': 2.8,
           'lines.linewidth': 2.2, 'lines.markersize': 7,
           'patch.linewidth': 1.0,
           'grid.linewidth': 0.7, 'grid.alpha': 0.4,
           'grid.linestyle': '-', 'grid.color': '#C8C8C8',
           'legend.frameon': True, 'legend.framealpha': 0.9,
           'axes.titleweight': 'bold', 'errorbar.capsize': 3,
           'savefig.dpi': 300, 'axes.prop_cycle': cycler(color=_SPEAK)}),

    'poster': _merge(
        **{'font.family': 'sans-serif', 'font.sans-serif': _FONT_SANS,
           'mathtext.fontset': 'stixsans',
           'font.size': 15, 'axes.titlesize': 19, 'axes.labelsize': 17,
           'xtick.labelsize': 13, 'ytick.labelsize': 13,
           'legend.fontsize': 13,
           'axes.linewidth': 1.3,
           'xtick.major.width': 1.3, 'ytick.major.width': 1.3,
           'xtick.minor.width': 1.0, 'ytick.minor.width': 1.0,
           'xtick.major.size': 7, 'ytick.major.size': 7,
           'xtick.minor.size': 4, 'ytick.minor.size': 4,
           'lines.linewidth': 3.0, 'lines.markersize': 10,
           'patch.linewidth': 1.4,
           'grid.linewidth': 1.0, 'grid.alpha': 0.35,
           'grid.linestyle': '-', 'grid.color': '#CCCCCC',
           'legend.frameon': True, 'axes.titleweight': 'bold',
           'errorbar.capsize': 4, 'savefig.dpi': 300,
           'axes.prop_cycle': cycler(color=_SPEAK)}),

    'tufte': _merge(
        **{'font.family': 'serif', 'font.serif': _FONT_SERIF,
           'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10.5,
           'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
           'axes.linewidth': 0.5,
           'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
           'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
           'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
           'xtick.minor.size': 2, 'ytick.minor.size': 2,
           'xtick.direction': 'out', 'ytick.direction': 'out',
           'xtick.top': False, 'ytick.right': False,
           'axes.spines.top': False, 'axes.spines.right': False,
           'lines.linewidth': 1.0, 'lines.markersize': 4,
           'axes.grid': False, 'legend.frameon': False,
           'axes.labelpad': 4.0, 'errorbar.capsize': 1.5,
           'savefig.dpi': 600,
           'axes.edgecolor': '#444444',
           'xtick.color': '#444444', 'ytick.color': '#444444',
           'axes.labelcolor': '#222222', 'text.color': '#222222'}),

    'screen': _merge(
        **{'font.family': 'sans-serif', 'font.sans-serif': _FONT_SANS,
           'mathtext.fontset': 'stixsans',
           'font.size': 11, 'axes.titlesize': 12.5, 'axes.labelsize': 11.5,
           'xtick.labelsize': 10, 'ytick.labelsize': 10,
           'legend.fontsize': 10,
           'axes.linewidth': 0.8,
           'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
           'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6,
           'xtick.major.size': 4.5, 'ytick.major.size': 4.5,
           'xtick.minor.size': 2.5, 'ytick.minor.size': 2.5,
           'lines.linewidth': 1.8, 'lines.markersize': 6,
           'grid.linewidth': 0.6, 'grid.alpha': 0.5,
           'grid.linestyle': '-', 'grid.color': '#C0C0C0',
           'legend.frameon': False,
           'figure.dpi': 110, 'savefig.dpi': 150,
           'errorbar.capsize': 2.5}),
}
"""Named structural base presets for :func:`set_base_style`.

Eight presets: ``standard`` (opinionated general default), ``structural``
(color/font-agnostic nudges over the mpl defaults), ``journal`` / ``tufte``
(thin, fine, 600 dpi print), ``press`` / ``poster`` (bolder, framed legend
for reproduction / distance), ``screen`` (slightly heavier for displays),
and ``minimalist`` (frameless splash / title images only — not data plots).
"""

# 'standard' = the opinionated general default (refined look + a default
# cycle). Formerly named 'pretty1'; reachable via _BASE_ALIASES below.
BASE_PRESETS['standard'] = _merge(
    **{'font.family': 'sans-serif', 'font.sans-serif': _FONT_SANS,
       'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
       'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
       'axes.linewidth': 0.7,
       'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
       'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6,
       'xtick.major.size': 5, 'ytick.major.size': 5,
       'xtick.minor.size': 2.5, 'ytick.minor.size': 2.5,
       'lines.linewidth': 1.5, 'grid.linewidth': 0.5, 'grid.alpha': 0.5,
       'grid.linestyle': ':', 'legend.frameon': True,
       'legend.handlelength': 2, 'legend.handletextpad': 0.6,
       'errorbar.capsize': 3, 'savefig.dpi': 200})

# Color/font-AGNOSTIC structural preset. Deliberately does NOT inherit
# _core(): no prop_cycle, no font.family, no image.cmap, no facecolors — just
# the structural nudges over the mpl defaults, for "improve the defaults but
# leave my colors and fonts exactly as they are." This standalone definition
# is load-bearing; keep it color/font-agnostic.
BASE_PRESETS['structural'] = {
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'xtick.top': True, 'ytick.right': True,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'xtick.minor.size': 2.5, 'ytick.minor.size': 2.5,
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6,
    'axes.linewidth': 0.7, 'axes.axisbelow': True,
    'grid.linewidth': 0.5, 'grid.alpha': 0.5, 'grid.linestyle': ':',
    'lines.linewidth': 1.5,
    'legend.numpoints': 1, 'legend.scatterpoints': 1,
    'legend.handlelength': 2, 'legend.handletextpad': 0.6,
    'errorbar.capsize': 3,
    'axes.formatter.use_mathtext': True, 'axes.formatter.limits': [-4, 4],
    'path.simplify': True,
    'savefig.dpi': 200, 'savefig.bbox': 'tight',
}

# Minimalist 'product-reveal' preset for SPLASH / TITLE / HERO images (a
# globe render, a single FITS cutout, a logo-like figure), NOT quantitative
# scatter/line plots — it strips the axis scaffolding a data plot needs.
# Frameless, no ticks, light sans-serif, generous whitespace. Also standalone
# (no _core()): it sets its own facecolors / spines / cmap.
#
# Font note: the airy look uses ``font.weight='light'``, which needs an
# installed light-capable sans (TeX Gyre Heros, a Noto/Roboto Light, ...).
# matplotlib's bundled DejaVu Sans has no light face, so where none of the
# stack's lighter fonts are installed it falls back to DejaVu Book — and on a
# system that has *some* partially-matching light font, matplotlib may route
# text to it and emit "Glyph N missing from font" warnings if that font lacks
# a symbol (°/′/″ etc.). DejaVu itself covers those symbols; the warning is a
# font-availability artifact of requesting a light weight, not a bad glyph
# default. Use a different base (e.g. 'screen'/'standard') or install a light
# font if you hit it.
BASE_PRESETS['minimalist'] = {
    'font.family': 'sans-serif',
    # Carlito dropped — no U+2032; see _FONT_SANS.
    'font.sans-serif': ['TeX Gyre Heros', 'Helvetica Neue', 'Nimbus Sans',
                        'Arial', 'DejaVu Sans'],
    'font.weight': 'light',
    'mathtext.fontset': 'stixsans',
    'figure.facecolor': '#FAFAFA', 'axes.facecolor': '#FAFAFA',
    'savefig.facecolor': '#FAFAFA',
    'axes.edgecolor': '#CFCFCF', 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False, 'axes.spines.bottom': False,
    'axes.grid': False,
    'xtick.bottom': False, 'ytick.left': False,
    'xtick.labelbottom': False, 'ytick.labelleft': False,
    'axes.titlesize': 22, 'axes.titleweight': 'light',
    'axes.titlecolor': '#1D1D1F', 'text.color': '#1D1D1F',
    'figure.dpi': 110, 'savefig.dpi': 220, 'savefig.bbox': 'tight',
    'image.cmap': 'magma', 'image.interpolation': 'antialiased',
    'image.origin': 'lower',
    'lines.linewidth': 2.5, 'lines.solid_capstyle': 'round',
    'axes.prop_cycle': cycler(color=['#0071E3', '#1D1D1F', '#86868B']),
}

#: Monospace font stack for fixed-width tabular annotations / coord readouts.
#: Not a base preset — apply per-artist via ``ax.text(..., family=sph.MONO_STACK)``.
MONO_STACK = _FONT_MONO

# Legacy base-style name -> current key. Kept working without a deprecation
# warning per the pre-release convention; resolved in set_base_style.
_BASE_ALIASES = {'pretty1': 'standard'}


# ---- Font presets (graceful, portable typeface stacks) ----------------
#
# Each stack LEADS with the prettiest target face and ENDS in a
# always-available face bundled with matplotlib (DejaVu Serif/Sans/Sans
# Mono) so a figure degrades gracefully instead of silently hard-falling
# when the target isn't installed. Applied by
# :func:`set_font`, which also pairs the math fontset to the text family.

# Palatino-metric warm serif (talks/posters); Avant-Garde geometric sans
# (the portable Gill Sans nod for Tufte-style charts); Cinzel Roman caps (a
# free Trajan substitute — display only, needs the face registered) over a
# strong serif body; and a hand-drawn stack that pairs with ``plt.xkcd()``.
_FONT_PAGELLA = ['TeX Gyre Pagella', 'Palatino', 'Palatino Linotype',
                 'Book Antiqua', 'DejaVu Serif']
_FONT_ADVENTOR = ['TeX Gyre Adventor', 'Gillius ADF', 'Gill Sans',
                  'Century Gothic', 'DejaVu Sans']
_FONT_CINZEL = ['Cinzel', 'Trajan Pro', 'TeX Gyre Bonum', 'TeX Gyre Schola',
                'DejaVu Serif']
_FONT_SKETCH = ['xkcd Script', 'Humor Sans', 'Comic Neue', 'Patrick Hand',
                'Caveat', 'DejaVu Sans']

#: Curated, portable font presets for :func:`set_font`, keyed by name. Each
#: value carries a ``stack`` (graceful family list, tier-1 fallback last), a
#: ``kind`` (``'serif'`` | ``'sans'`` | ``'display'`` | ``'handwriting'`` |
#: ``'mono'``) that selects the family generic + ``math='auto'`` pairing, the
#: paired ``math`` fontset, and a one-line ``desc`` for docs to table.
FONT_PRESETS: dict[str, dict[str, Any]] = {
    'journal': {'stack': _FONT_SERIF, 'kind': 'serif', 'math': 'cm',
                'desc': 'Times-metric serif (TeX Gyre Termes) — polished '
                        'ApJ/MNRAS journal look.'},
    'talk': {'stack': _FONT_PAGELLA, 'kind': 'serif', 'math': 'cm',
             'desc': 'Palatino-like serif (TeX Gyre Pagella) — warm and open; '
                     'talks & posters.'},
    'tufte': {'stack': _FONT_ADVENTOR, 'kind': 'sans', 'math': 'stixsans',
              'desc': 'Avant-Garde geometric sans (TeX Gyre Adventor) — the '
                      'portable Gill Sans nod for Tufte-style charts.'},
    'web': {'stack': _FONT_SANS, 'kind': 'sans', 'math': 'stixsans',
            'desc': 'Helvetica-metric sans (TeX Gyre Heros / Nimbus Sans) '
                    '— clean and modern for screens & docs.'},
    'classical': {'stack': _FONT_CINZEL, 'kind': 'display', 'math': 'cm',
                  'desc': 'Cinzel Roman monumental caps (register the face) '
                          'over a strong serif body — classical headers.'},
    'sketch': {'stack': _FONT_SKETCH, 'kind': 'handwriting',
               'math': 'dejavusans',
               'desc': 'Hand-drawn stack (xkcd Script / Patrick Hand / Caveat) '
                       '— informal explainers; pairs with plt.xkcd().'},
    'mono': {'stack': _FONT_MONO, 'kind': 'mono', 'math': 'dejavusans',
             'desc': 'Monospace stack (IBM Plex Mono) — tables, code, '
                     'fixed-width tick labels.'},
}
"""Curated, portable font presets for :func:`set_font` (see the registry's
inline note for the per-preset fields)."""

# Legacy / convenience font-preset aliases (no deprecation warning, per the
# pre-release convention; resolved in set_font). 'poster' shares the warm
# Pagella serif of 'talk'.
_FONT_ALIASES = {'poster': 'talk'}

# Per-kind family generic + the rcParams list key that holds the stack, and
# the tier-1 fallback that every stack must end in. Display falls under the
# serif generic (Cinzel-style caps), handwriting under sans.
_FONT_KIND_FAMILY = {
    'serif': ('serif', 'font.serif', 'DejaVu Serif'),
    'display': ('serif', 'font.serif', 'DejaVu Serif'),
    'sans': ('sans-serif', 'font.sans-serif', 'DejaVu Sans'),
    'handwriting': ('sans-serif', 'font.sans-serif', 'DejaVu Sans'),
    'mono': ('monospace', 'font.monospace', 'DejaVu Sans Mono'),
}

# math='auto' pairing per kind: cm with serif/display bodies, stixsans with
# sans, dejavusans for the always-available handwriting/mono cases.
_FONT_AUTO_MATH = {
    'serif': 'cm', 'display': 'cm', 'sans': 'stixsans',
    'handwriting': 'dejavusans', 'mono': 'dejavusans',
}


def set_base_style(style: str | dict[str, Any] = 'standard',
                   specific_RCs: dict[str, Any] | None = None) -> None:
    """
    Apply a structural matplotlib RC parameter preset.

    Parameters
    ----------
    style : str or dict
        A key in :data:`BASE_PRESETS`, a reset/dict directive, or a legacy
        alias. The presets, with a one-line use each:

        - ``'standard'`` (default) — opinionated, batteries-included general
          look (refined structure, sans-serif stack, a default cycle).
        - ``'structural'`` — color/font-*agnostic*: just structural nudges
          over the mpl defaults, leaving your colors and fonts untouched
          (good for bulk data-inspection plots).
        - ``'journal'`` — thin lines, fine ticks, serif/cm, 600 dpi (print).
        - ``'press'`` — bolder sans-serif with a framed legend (reproduction).
        - ``'poster'`` — large type and heavy lines for viewing at distance.
        - ``'tufte'`` — minimal-ink serif, no top/right spines (restraint).
        - ``'screen'`` — slightly heavier so hairlines don't vanish on
          displays; lower dpi.
        - ``'minimalist'`` — frameless, tickless splash / title / hero
          images **only** (strips the axis scaffolding a data plot needs).
          Its light type needs an installed light-capable sans; with none,
          matplotlib falls back to DejaVu Book and may warn about a missing
          glyph if it routes text to a partial light font (see the preset's
          source comment). Use ``'screen'`` / ``'standard'`` to avoid that.

        Also accepts ``'default'`` / ``'reset'`` / ``'mpl'`` (or any string
        containing ``'def'``) to reset rcParams via ``plt.rcdefaults()``,
        a dict of RC params (a fully custom base layer), and the legacy
        alias ``'pretty1'`` (resolves to ``'standard'``). Unknown names emit
        a ``UserWarning`` and leave rcParams unchanged.
    specific_RCs : dict, optional
        Extra RC params merged in last (override hook), for every branch.

    Examples
    --------
    >>> sph.set_base_style('standard')
    >>> sph.set_base_style('journal', {'axes.grid': False})
    >>> sph.set_base_style({'xtick.direction': 'out', 'font.size': 11})

    Notes
    -----
    Fonts are set as *stacks* so they degrade gracefully; serif presets pair
    ``mathtext.fontset='cm'`` and sans presets ``'stixsans'``. If you override
    the family via ``specific_RCs`` the math fontset is not re-coordinated
    automatically. For a self-restoring variant use :class:`style_context`.
    """
    specific_RCs = specific_RCs or {}

    # Custom base layer: a dict of rcParams applied as-is.
    if isinstance(style, dict):
        _rc_update({**style, **specific_RCs})
        return

    s = str(style).lower()

    # Reset branch (preserve existing 'def*' behavior; also 'reset'/'mpl').
    if 'def' in s or s in ('reset', 'mpl'):
        plt.rcdefaults()
        if specific_RCs:
            _rc_update(specific_RCs)
        return

    # Resolve legacy aliases (e.g. 'pretty1' -> 'standard'), then dispatch.
    key = _BASE_ALIASES.get(style, style)
    if key in BASE_PRESETS:
        _rc_update({**BASE_PRESETS[key], **specific_RCs})
        return

    warnings.warn(
        f"set_base_style: unknown style {style!r}. Supported: "
        f"{', '.join(BASE_PRESETS)}, 'default'/'reset', or a dict. "
        f"rcParams unchanged.",
        stacklevel=2)


# ---- Theme presets (coordinated colors for axes, grid, labels, background) ----

_THEMES: dict[str, dict[str, Any]] = {
    'publication': {
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'grid.color': '0.8',
        'figure.facecolor': 'white',
        'text.color': 'black',
    },
    # 'twilight' is the violet-tinged dark theme (a hint of afterglow);
    # 'dark_sky' below is the near-black night-sky variant. The cycle
    # palettes (CYCLE_PALETTES) are tuned to read well on either.
    'twilight': {
        'axes.facecolor': '#1a1a2e',
        'axes.edgecolor': '#e0e0e0',
        'axes.labelcolor': '#e0e0e0',
        'xtick.color': '#e0e0e0',
        'ytick.color': '#e0e0e0',
        'grid.color': '#333366',
        'figure.facecolor': '#0f0f23',
        'text.color': '#e0e0e0',
        'image.cmap': 'inferno',
    },
    'dark_sky': {
        'axes.facecolor': '#0d1117',
        'axes.edgecolor': '#8b949e',
        'axes.labelcolor': '#c9d1d9',
        'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',
        'grid.color': '#21262d',
        'figure.facecolor': '#010409',
        'text.color': '#c9d1d9',
        'image.cmap': 'magma',
    },
    'poster': {
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'grid.color': '0.85',
        'figure.facecolor': 'white',
        'text.color': 'black',
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
    },
}


def set_theme(theme: str | dict[str, Any] = 'publication') -> None:
    """
    Apply a coordinated color theme for sky plots.

    Parameters
    ----------
    theme : str or dict
        One of:

        - a skyplothelper preset name ('publication', 'twilight',
          'dark_sky', 'poster') — ``'twilight'`` is the violet-tinged
          dark theme; ``'dark_sky'`` is the near-black night-sky variant;
        - any matplotlib built-in style name (e.g. ``'ggplot'``,
          ``'bmh'``, ``'seaborn-v0_8-darkgrid'`` — see
          ``matplotlib.style.available``);
        - a dict of RC params — your own custom theme.

        skyplothelper preset names take precedence over matplotlib
        style names if they ever collide.

    Examples
    --------
    >>> sph.set_theme('dark_sky')
    >>> sph.set_theme('ggplot')                       # matplotlib built-in
    >>> sph.set_theme({'axes.facecolor': '#1d1c1a',
    ...                'figure.facecolor': '#1d1c1a',
    ...                'text.color': '#d9d5c5'})      # custom
    >>> fig, ax = sph.allsky_figure()
    """
    if isinstance(theme, dict):
        _rc_update(theme)
    elif theme in _THEMES:
        _rc_update(_THEMES[theme])
    elif theme in mpl_style.library:
        _rc_update(dict(cast("Any", mpl_style.library[theme])))
    else:
        raise ValueError(
            f"Unknown theme '{theme}'. Available skyplothelper presets: "
            f"{', '.join(_THEMES.keys())}; any matplotlib style name "
            "(see matplotlib.style.available) also works.")


# ---- Palette layer (axes.prop_cycle) ----------------------------------
#
# The palette layer is deliberately separate from the theme layer: a theme
# coordinates the *background/foreground* (facecolor, text, ticks, grid,
# default cmap), while a palette is purely the data-color cycle. Keeping
# them independent lets a user mix any palette with any theme (or apply a
# palette on top of their own rcParams). Palettes are opt-in — neither
# ``set_base_style`` nor ``set_theme`` touches ``axes.prop_cycle``, so existing
# plots keep matplotlib's default cycle until ``set_palette`` is called.
#
# Each entry carries a ``mode`` (where it's designed to read well): 'dual'
# works on both light and dark backgrounds; 'light'/'dark' are tuned for
# one. The hex values are the CVD-tweaked finals (every confusable pair
# separated by lightness so they also survive grayscale printing).

CYCLE_PALETTES: dict[str, dict[str, Any]] = {
    # Aged-brass / terracotta / dusty-teal mid-lightness set — the headline
    # dual-mode cycle and the documented tutorial default.
    'speakeasy': {'mode': 'dual', 'colors': [
        '#B98E3E', '#A35D4C', '#41736B', '#67809F',
        '#8A8B57', '#96718F', '#9C9285', '#B26B79']},
    # Quieter, more cartographic dual-mode set with wider hue separation —
    # suits 3-5-region tutorial figures.
    'atlas': {'mode': 'dual', 'colors': [
        '#5B8A9A', '#C08A4D', '#6E8A64', '#92566B',
        '#6E6A9E', '#9CA092']},
    # Antique star-atlas / finder-chart set (twilight navy, burnished
    # copper, antique gold, verdigris...); dual-mode.
    'uranometria': {'mode': 'dual', 'colors': [
        '#46618A', '#B97C52', '#C29B3C', '#9DA3AB',
        '#5E8C7E', '#8A4540', '#716A8E']},
    # Deep inky publication-on-white set; muddy on dark, so light-only.
    'letterpress': {'mode': 'light', 'colors': [
        '#2E5266', '#A14E32', '#2F6B5E', '#9C7C26',
        '#6E4A6B', '#77833F', '#7A3B47', '#444444']},
    # Warm lamplit-bar set for dark backgrounds (champagne, coral clay,
    # sea glass, periwinkle, rose quartz...).
    'nightcap': {'mode': 'dark', 'colors': [
        '#E2C275', '#DD8E70', '#8FBFAF', '#93A8D1',
        '#C99BB3', '#A4AE7D', '#D8C5A4', '#9C9890']},
    # Moodier jewel-toned dark set with more saturation for punch against
    # a night-sky background.
    'velvet': {'mode': 'dark', 'colors': [
        '#D4A24E', '#C96F5B', '#5FA796', '#8093C9',
        '#B07BA5', '#94A052', '#C2A189']},
}
"""Named data-color cycles for :func:`set_palette`, keyed by palette name.

Each value is a ``{'mode': 'dual'|'light'|'dark', 'colors': [...]}`` dict.
The palettes are CVD-/grayscale-tested (the colors are lightness-separated,
so they stay distinguishable under color-vision deficiency and in grayscale).
"""


def set_palette(palette: str | list[str]) -> None:
    """
    Set the axes color cycle (``axes.prop_cycle``) to a named or explicit
    palette.

    This is the third, independent style layer (see :func:`set_style`).
    It only changes the data-color cycle — backgrounds, ticks, and the
    default colormap come from :func:`set_theme`.

    Parameters
    ----------
    palette : str or list of color
        A key in :data:`CYCLE_PALETTES`, or an explicit list of matplotlib
        colors to cycle through.

    Examples
    --------
    >>> sph.set_palette('speakeasy')
    >>> sph.set_palette(['#264653', '#2A9D8F', '#E9C46A'])
    """
    if isinstance(palette, str):
        try:
            colors = list(CYCLE_PALETTES[palette]['colors'])
        except KeyError:
            raise ValueError(
                f"Unknown palette {palette!r}. Available: "
                f"{', '.join(CYCLE_PALETTES)}") from None
    else:
        colors = list(palette)
    rcParams['axes.prop_cycle'] = cycler(color=colors)


# ---- Font layer (graceful typeface stacks + paired math fontset) ------

def _register_fonts(register: str | list[str]) -> None:
    """``addfont`` every .ttf/.otf in *register* (a file, dir, or list)."""
    import os

    import matplotlib.font_manager as fm
    paths = [register] if isinstance(register, str) else list(register)
    for p in paths:
        if os.path.isdir(p):
            for f in fm.findSystemFonts(fontpaths=[p]):
                fm.fontManager.addfont(f)
        elif os.path.isfile(p):
            fm.fontManager.addfont(p)


def _font_available(name: str) -> bool:
    """True if matplotlib can resolve *name* to an actual installed face."""
    import matplotlib.font_manager as fm
    try:
        fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
        return True
    except Exception:
        return False


def is_dark_background(color: Any) -> bool:
    """Is *color* dark enough that ink drawn on it should be light?

    Relative luminance, so any spelling of a dark background is recognized —
    not just the handful of names a string check would catch.
    """
    from matplotlib.colors import to_rgb
    try:
        r, g, b = to_rgb(color)
    except (ValueError, TypeError):
        return False
    return (0.299 * r + 0.587 * g + 0.114 * b) < 0.4


def muted_ink(reference: Any = None, light: str = '0.4',
              dark: str | None = None) -> str:
    """A deliberately de-emphasized gray that reads against *reference*.

    For text and marks that are *meant* to sit quieter than the primary ink —
    sub-tick labels, secondary callouts, neutral swatches. Those must NOT be
    resolved to ``rcParams['text.color']``: that promotes them to full
    strength and flattens the hierarchy they were softened to create. The
    only thing wrong with a bare ``'0.35'`` is that it is a *light-theme*
    muted tone, invisible on a dark canvas.

    Parameters
    ----------
    reference : color or Axes, optional
        The background to read against. An Axes contributes its facecolor.
        Defaults to ``rcParams['axes.facecolor']``.
    light : str
        The tone to use on a light background — the existing literal, so
        light-theme renders are preserved exactly.
    dark : str, optional
        The tone on a dark background. Defaults to the mirrored gray level
        (``'0.35'`` -> ``'0.65'``), which keeps the same contrast against the
        canvas; pass it explicitly when a specific tone is wanted.
    """
    if reference is None:
        reference = rcParams['axes.facecolor']
    elif hasattr(reference, 'get_facecolor'):
        reference = reference.get_facecolor()
    if not is_dark_background(reference):
        return light
    if dark is not None:
        return dark
    try:
        return f'{max(0.0, min(1.0, 1.0 - float(light))):.2f}'
    except (TypeError, ValueError):
        # A named tone with no numeric mirror — fall back to a light gray.
        return '0.7'


#  ′ (arcmin) and ″ (arcsec) — emitted by every sexagesimal / offset label.
_REQUIRED_MARKS = (0x2032, 0x2033)


@lru_cache(maxsize=64)
def _face_has_marks(family: str) -> bool | None:
    """Can *family*, as installed here, render the prime marks?

    ``None`` means the family isn't installed, so it is not this stack entry's
    turn to answer. Uses matplotlib's own FT2Font (``get_char_index`` returns 0
    for an absent codepoint) rather than fontTools, to avoid leaning on a
    transitive dependency for something this load-bearing.
    """
    from matplotlib import font_manager as fm
    from matplotlib.ft2font import FT2Font
    try:
        path = fm.findfont(fm.FontProperties(family=family),
                           fallback_to_default=False)
    except Exception:
        return None
    try:
        face = FT2Font(path)
        return all(face.get_char_index(cp) != 0 for cp in _REQUIRED_MARKS)
    except Exception:
        return None


def _coerce_font_stack(stack: list[str], fallback: str) -> list[str]:
    """Warn when *stack* cannot render the marks sph emits — but honor it.

    Note what the appended *fallback* does and does not buy. matplotlib
    resolves a family list by picking the first **installed** entry and
    rendering the whole string with it — there is no per-glyph fallback — so a
    trailing face is unreachable whenever anything ahead of it exists. It is a
    net only for the case where *none* of the listed faces are installed, and
    it is kept for exactly that.

    It is deliberately NOT promoted to the front of the list when the winning
    face lacks ``′ ″``. Promoting it rewrote ``font.serif`` /
    ``font.sans-serif`` figure-wide, so a preset naming a specific face
    silently rendered in DejaVu instead: on a machine with TeX Gyre installed,
    ``'journal'`` advertised TeX Gyre Termes and drew the untouched-matplotlib
    look it exists to avoid. That is a disproportionate remedy — the whole
    figure loses its typeface over two glyphs that appear only in sexagesimal
    tick labels, which many figures never draw.

    So the missing coverage is reported and left to the caller. The failure it
    permits is *visible* (tofu boxes) rather than silent, and the warning names
    both fixes. This is the policy that already applied to the display /
    handwriting presets, where a decorative face is the entire point; the only
    change is that it is no longer scoped to them, since a preset naming
    TeX Gyre Termes means it just as much as one naming a script face.
    """
    out = list(stack)
    if fallback not in out:
        out.append(fallback)

    for family in out:
        has = _face_has_marks(family)
        if has is None:
            continue            # not installed — the next entry gets its turn
        if has:
            return out          # the winning face is fine, leave the stack be
        warnings.warn(
            f"font {family!r} cannot render the prime marks ′ ″ that "
            "skyplothelper uses in coordinate labels, so those labels will "
            "show boxes; set separator='hms_letter' / dms_letter, or pick "
            "another font, if that matters here. Faces that do carry them "
            "include 'Nimbus Sans', 'Liberation Sans', 'Arial' and "
            "'Helvetica'.",
            stacklevel=3)
        return out
    return out


def _apply_font_family(kind: str, stack: list[str]) -> None:
    """Set ``font.family`` generic + the matching stack list for *kind*."""
    generic, key, fallback = _FONT_KIND_FAMILY[kind]
    _rc_update({'font.family': generic,
                key: _coerce_font_stack(stack, fallback)})


def set_font(font: str | list[str] | dict[str, Any] = 'default', *,
             math: str | None = 'auto', mono: list[str] | None = None,
             register: str | list[str] | None = None,
             specific_RCs: dict[str, Any] | None = None) -> None:
    """
    Set the figure font family (as a graceful stack) and pair the math fontset.

    The optional fourth style layer, mirroring :func:`set_base_style` /
    :func:`set_theme` / :func:`set_palette`: it updates rcParams in place (so it
    composes with :class:`style_context`, which save/restores them) and enforces
    the two portability rules the docs preach — always specify a *stack* ending
    in a guaranteed tier-1 face (DejaVu) so a figure never silently hard-falls,
    and pair the math fontset to the text family.

    Parameters
    ----------
    font : str, list, or dict
        One of:

        - a key in :data:`FONT_PRESETS` (``'journal'``, ``'talk'``/``'poster'``,
          ``'tufte'``, ``'web'``, ``'classical'``, ``'sketch'``, ``'mono'``);
        - a family generic ``'serif'`` / ``'sans-serif'``;
        - a specific installed face (e.g. ``'TeX Gyre Termes'``), turned into a
          stack ending in the matching tier-1 fallback (defaults to the serif
          family — pass the generic, a list, or a preset for sans);
        - an explicit stack ``list`` (the tier-1 fallback is appended if
          absent);
        - a ``dict`` of ``font.*`` rcParams, applied as-is;
        - ``'default'`` / ``'reset'`` to restore the matplotlib default fonts.

        An unknown name that is neither a preset nor an available face emits a
        ``UserWarning`` (listing the presets) and leaves the family unchanged.
    math : str or None
        ``'auto'`` (default) pairs the fontset to the family — ``'cm'`` for
        serif/display, ``'stixsans'`` for sans, ``'dejavusans'`` for the
        handwriting/mono presets. Pass an explicit ``mathtext.fontset``
        (``'cm'`` / ``'stix'`` / ``'stixsans'`` / ``'dejavuserif'`` /
        ``'dejavusans'``) to override, or ``None`` to leave mathtext untouched.
    mono : list of str, optional
        Monospace stack for ``font.monospace``. Defaults to :data:`MONO_STACK`,
        so monospace text artists always have a good stack too.
    register : str or list of str, optional
        Path(s) — files or directories — of ``.ttf`` / ``.otf`` faces to
        ``addfont`` before applying, so opt-in faces (e.g. Cinzel for
        ``'classical'``, the handwriting faces for ``'sketch'``) resolve.
    specific_RCs : dict, optional
        Extra rcParams merged in last (override hook), matching
        :func:`set_base_style`.

    Examples
    --------
    >>> sph.set_font('journal')                 # Termes + cm
    >>> sph.set_font('web', math='stixsans')    # Heros/Nimbus + explicit math
    >>> sph.set_font('sketch', register='xkcd-script.ttf')
    >>> sph.set_font(['EB Garamond', 'TeX Gyre Termes'])   # explicit stack
    >>> sph.set_font('default')                 # restore mpl defaults

    See Also
    --------
    set_style : composes base → theme → palette → font → overrides.
    """
    specific_RCs = specific_RCs or {}
    if register is not None:
        _register_fonts(register)

    _FONT_KEYS = ('font.family', 'font.serif', 'font.sans-serif',
                  'font.monospace', 'mathtext.fontset')

    # Reset branch — restore the matplotlib default fonts (then honor any
    # explicit math / mono / specific_RCs the caller still passed).
    if isinstance(font, str) and ('def' in font.lower()
                                  or font.lower() in ('reset', 'mpl')):
        defaults = cast("Any", plt.rcParamsDefault)
        reset: dict[str, Any] = {k: defaults[k] for k in _FONT_KEYS}
        if math not in ('auto', None):
            reset['mathtext.fontset'] = math
        if mono is not None:
            reset['font.monospace'] = list(mono)
        reset.update(specific_RCs)
        _rc_update(reset)
        return

    kind = 'serif'   # default family for raw-face / list inputs

    if isinstance(font, dict):
        _rc_update(font)
        fam = str(font.get('font.family', rcParams.get('font.family', 'serif')))
        kind = 'sans' if 'sans' in fam else 'serif'
    elif isinstance(font, (list, tuple)):
        _apply_font_family(kind, list(font))
    elif font in ('serif', 'sans-serif'):
        rcParams['font.family'] = font
        kind = 'sans' if font == 'sans-serif' else 'serif'
    else:
        key = _FONT_ALIASES.get(font, font)
        if key in FONT_PRESETS:
            preset = FONT_PRESETS[key]
            kind = preset['kind']
            _apply_font_family(kind, preset['stack'])
            if math == 'auto':
                math = preset.get('math') or _FONT_AUTO_MATH[kind]
        elif _font_available(font):
            _apply_font_family('serif', [font])   # raw face → serif stack
        else:
            warnings.warn(
                f"set_font: unknown preset / unavailable font {font!r}. "
                f"Supported presets: {', '.join(FONT_PRESETS)}; or a 'serif'/"
                f"'sans-serif' generic, an installed face, a stack list, a "
                f"dict, or 'default'. Font family unchanged.",
                stacklevel=2)
            return

    # Pair the math fontset to the family unless told not to ('auto' resolves
    # by kind; an explicit fontset passes through; None leaves it untouched).
    if math == 'auto':
        rcParams['mathtext.fontset'] = _FONT_AUTO_MATH[kind]
    elif math is not None:
        rcParams['mathtext.fontset'] = math

    # Always give monospace text artists a good stack too.
    rcParams['font.monospace'] = list(mono) if mono is not None else list(MONO_STACK)

    if specific_RCs:
        _rc_update(specific_RCs)


# ---- Annotation / scaffolding palettes --------------------------------
#
# Distinct from the cycle CYCLE_PALETTES above: those are *data* colors; these
# coordinate the *figure scaffolding* (backgrounds, two tiers of text, two
# tiers of grid, frame, default star color, two accents, plus object-label
# and compass colors) for finder-chart / star-atlas style figures. The
# two-tier text + two-tier grid structure is what gives these the layered
# "composed" look. Applied via :func:`style_annotation`, which is
# a separate accessor from :func:`set_theme` — so a name shared between the
# two namespaces (e.g. 'publication') is unambiguous in practice.
#
# Roles:
#   fig_bg, ax_bg   figure & axes backgrounds
#   text, text2     primary & secondary (subdued) text colors
#   label           primary object-label ink (distinct from stars/text)
#   compass         compass rose / orientation indicators / leader lines
#   grid, grid2     foreground & background (e.g. overlay-frame) grids
#   frame           axes frame / spines / tick color
#   stars           default marker/star color
#   accent, accent2 target reticles, FOV boxes, emphasis

ANNOTATION_PALETTES: dict[str, dict[str, str]] = {
    # Cream cartographic paper: leather-brown labels, faded-denim compass,
    # pale graticule-blue grid, crimson + brass accents.
    'parchment': dict(
        fig_bg='#F4EFE3', ax_bg='#FAF6EB', text='#33302A',
        text2='#847D6E', label='#7C5234', compass='#5E7591',
        grid='#ADBDC9', grid2='#E3DCCB', frame='#56524A',
        stars='#3A372F', accent='#9D4B36', accent2='#B98E3E'),
    # Journal-ready: near-black charcoal axes (spine + ticks at '0.1'), with
    # ticklabels one step lighter to match the axis labels, whisper grids,
    # ink-blue labels, one sparing red accent. (The spine/tick/ticklabel set
    # is intentionally dark — a light gray frame reads as draft, not print.)
    'publication': dict(
        fig_bg='#FFFFFF', ax_bg='#FFFFFF', text='#333333',
        text2='#333333', label='#2E5266', compass='#6B7B8C',
        grid='#DCDCDC', grid2='#EFEFEF', frame='#1A1A1A',
        stars='#444444', accent='#A4452D', accent2='#2E5266'),
    # Midnight chart: parchment text, gold-leaf labels, verdigris compass,
    # silver frame, gold/ember accents.
    'dark': dict(
        fig_bg='#0E1117', ax_bg='#141925', text='#E6DFCE',
        text2='#9A958A', label='#D4B36A', compass='#7FA694',
        grid='#3D4658', grid2='#262C3A', frame='#8E96A3',
        stars='#E8E3D4', accent='#C9A23F', accent2='#C96F5B'),
    # Colder, quieter night mode: muted red-light reticle (preserves dark
    # adaptation), sea-glass labels, steel-blue compass.
    'night': dict(
        fig_bg='#0B0D10', ax_bg='#0B0D10', text='#C8CCD2',
        text2='#7E848D', label='#8FBFAF', compass='#8195A3',
        grid='#2E333B', grid2='#1D2126', frame='#5F6671',
        stars='#D5D9DE', accent='#B0564E', accent2='#E2C275'),
    # Warm-gray dark mode with no blue cast: warm charcoal field, parchment
    # text, denim-blue labels, sage compass, burnt-rust reticle, khaki-gold
    # accents. (DenimDark-editor-derived.)
    'denim': dict(
        fig_bg='#1D1C1A', ax_bg='#262522', text='#D9D5C5',
        text2='#7E736A', label='#7C9AB6', compass='#8F9D6A',
        grid='#403E39', grid2='#2D2B27', frame='#8A847A',
        stars='#E2DDCB', accent='#B45A33', accent2='#C5B777'),
}
"""Coordinated annotation color themes for :func:`style_annotation`, keyed by name.

Each value maps annotation roles (``fig_bg``, ``ax_bg``, ``text``, ``label``,
``compass``, ``grid``, ``frame``, ``accent``, ...) to colors.
"""

# Legacy annotation-palette name -> current key (no deprecation warning per
# the pre-release convention; resolved in style_annotation).
_ANNOTATION_ALIASES = {'coffee': 'denim'}


def _is_wcsaxes(ax: Any) -> bool:
    """Duck-type check for WCSAxes (covers subclasses, e.g. tilted globe)."""
    return hasattr(ax, 'coords')


def _apply_tick_stroke(coord: Any, stroke_lw: float | None,
                       stroke_color: Any, wrap_ends: bool = False) -> None:
    """Give a WCSAxes coord's tick marks a contrasting stroke (or remove one).

    Astropy's ``Ticks._draw_ticks`` builds its own ``GraphicsContext`` and
    calls ``renderer.draw_markers`` directly, so it never consults
    ``path_effects`` — ``ticks.set_path_effects(...)`` is silently ignored.
    Wrapping the *renderer* in a ``PathEffectRenderer`` also fails (the
    ``allow_rasterization`` decorator on ``Ticks.draw`` reads
    ``renderer._raster_depth``, which the wrapper doesn't proxy). So emulate
    ``withStroke`` with a two-pass draw via the public width/color: a wider
    stroke-colored underlay, then the normal ticks on top. ``stroke_lw`` is the
    total underlay line width (matching ``withStroke(linewidth=...)`` as used
    by :func:`style_grid` / :func:`format_ticklabels`).

    ``wrap_ends`` also lengthens the underlay ticks by the stroke half-width so
    the stroke wraps the tick *ends* (not just the sides) — a fuller stroke for
    inward ticks over imagery. Off by default so existing callers are unchanged.
    """
    ticks = coord_ticks(coord)
    # Undo any prior wrap first, so re-styling is idempotent (never stacks):
    # drop our marker and the instance-level ``draw`` override, re-exposing
    # astropy's class method.
    if ticks.__dict__.pop('_sph_orig_tick_draw', None) is not None:
        ticks.__dict__.pop('draw', None)
    if stroke_color is None or stroke_lw is None or float(stroke_lw) <= 0:
        return
    base_draw = ticks.draw  # astropy's (decorated) Ticks.draw class method
    ticks._sph_orig_tick_draw = base_draw
    sw = float(stroke_lw)

    def _draw(renderer: Any, *args: Any, **kwargs: Any) -> Any:
        lw0, c0 = ticks.get_linewidth(), ticks.get_color()
        grow = max(0.0, (sw - lw0) / 2.0) if wrap_ends else 0.0
        if grow:
            sz0, msz0 = ticks.get_ticksize(), ticks.get_minor_ticksize()
            ticks.set_ticksize(sz0 + grow)
            ticks.set_minor_ticksize(msz0 + grow)
        ticks.set_linewidth(sw)
        ticks.set_color(stroke_color)
        base_draw(renderer, *args, **kwargs)   # stroke underlay
        ticks.set_linewidth(lw0)
        ticks.set_color(c0)
        if grow:
            ticks.set_ticksize(sz0)
            ticks.set_minor_ticksize(msz0)
        base_draw(renderer, *args, **kwargs)   # normal ticks on top

    ticks.draw = _draw


def apply_frame_stroke(ax: Any, stroke_color: Any = 'white',
                       stroke_lw: float | None = 1.6) -> None:
    """Stroke an axes frame + tick marks as one unit, for legibility over
    imagery (dark *or* bright backgrounds).

    Keeps a frame and its tick marks readable over a filled image — e.g. an
    astronomical map on a colormap that runs through black — by drawing a
    contrasting stroke around them (a dark core with a light stroke, or vice
    versa, stays visible on any background). Tick *labels* are left untouched.

    On a **WCSAxes** the frame is stroked as one continuous path that follows
    the actual frame *shape* — rectangular (TAN/CAR/globe insets), elliptical
    (AIT/MOL all-sky), or any other WCSAxes frame — reusing the frame patch's
    own path (so corners never cut across themselves, unlike per-spine
    strokes). The native tick marks — which ignore ``path_effects`` — are
    stroked with the two-pass draw of :func:`style_wcs_axes`
    (``wrap_ends=True`` so the tick ends are wrapped too). On a **plain Axes**,
    spines and tick lines take the stroke via ``path_effects`` directly.
    Idempotent: re-calling replaces the prior stroke; ``stroke_color=None`` /
    ``stroke_lw<=0`` removes it. Call it after the frame / limits are
    finalized, as with other decorations.

    Parameters
    ----------
    ax : matplotlib Axes or WCSAxes
        The axes whose frame + tick marks to stroke.
    stroke_color : color or None
        Stroke color (default ``'white'``). ``None`` removes an existing
        stroke.
    stroke_lw : float
        Total stroke line width in points (default 1.6). The *visible* stroke
        on each side is ``(stroke_lw - line_lw) / 2`` — so with a typical ~0.8
        frame/tick line this shows ~0.4 pt each side: enough to separate the
        frame from the image without competing with it. Must exceed the line
        width to be visible.

    Examples
    --------
    >>> ax.imshow(data, cmap='inferno')          # runs through black
    >>> sph.apply_frame_stroke(ax)               # white stroke, stays visible
    >>> sph.apply_frame_stroke(ax, 'black', 3)   # dark stroke for a light frame
    """
    import matplotlib.patches as mpatches

    from ._stroke import _stroke_path_effects

    # Drop any prior frame-stroke rectangle so re-styling never stacks.
    for patch in [p for p in ax.patches if p.get_gid() == '_sph_frame_stroke']:
        patch.remove()
    coords = list(getattr(ax, 'coords', []) or [])
    disable = stroke_color is None or stroke_lw is None or float(stroke_lw) <= 0

    if coords:
        # WCSAxes: native ticks need the two-pass; clear or (re)apply it.
        if disable:
            for coord in coords:
                _apply_tick_stroke(coord, None, None)
            return
        assert stroke_lw is not None  # not disabled → set
        sw = float(stroke_lw)
        # Follow the actual frame shape — rectangular (TAN/CAR/globe insets),
        # elliptical (AIT/MOL all-sky), or any other WCSAxes frame — by reusing
        # the frame patch's own path + (live) transform, drawn just under the
        # frame and unclipped so its outer half shows. (Call after the frame /
        # limits are finalized, as with other decorations.)
        fpatch = getattr(getattr(ax, 'coords', None), 'frame', None)
        fpatch = getattr(fpatch, 'patch', None)
        if fpatch is not None:
            ax.add_patch(mpatches.PathPatch(
                fpatch.get_path(), transform=fpatch.get_transform(),
                fill=False, edgecolor=stroke_color, linewidth=sw,
                zorder=fpatch.get_zorder() - 0.01, clip_on=False,
                gid='_sph_frame_stroke'))
        else:  # fallback: the axes-boundary rectangle
            spine_z = min((s.get_zorder() for s in ax.spines.values()),
                          default=2.5)
            ax.add_patch(mpatches.Rectangle(
                (0, 0), 1, 1, transform=ax.transAxes, fill=False,
                edgecolor=stroke_color, linewidth=sw, zorder=spine_z - 0.01,
                clip_on=False, gid='_sph_frame_stroke'))
        for coord in coords:
            _apply_tick_stroke(coord, sw, stroke_color, wrap_ends=True)
        return

    # Plain Axes: spines and tick lines honor path_effects directly.
    pe = None if disable else _stroke_path_effects(stroke_color, stroke_lw)
    for spine in ax.spines.values():
        spine.set_path_effects(pe or [])
    for tl in (*ax.xaxis.get_ticklines(), *ax.yaxis.get_ticklines()):
        tl.set_path_effects(pe or [])


def style_wcs_axes(
    ax: Any, direction: str | None = None, major_size: float | None = None,
    minor_size: float | None = None, width: float | None = None,
    tick_color: Any = None, minor_ticks: bool | None = None,
    minor_frequency: int | None = None, labelcolor: Any = None,
    labelsize: Any = None, axislabel_color: Any = None,
    axislabel_size: Any = None, frame_color: Any = None,
    frame_linewidth: float | None = None, grid: bool = False,
    grid_color: Any = None, grid_linestyle: Any = None,
    grid_linewidth: float | None = None, grid_alpha: float | None = None,
    stroke_lw: float | None = None, stroke_color: Any = None,
    coords: list[Any] | None = None,
) -> Any:
    """
    Apply tick / label / frame / grid styling to a WCSAxes, translating
    the rc-based style conventions that WCSAxes otherwise ignores.

    WCSAxes ignores most ``xtick.*`` / ``ytick.*`` rcParams — tick
    direction, length, width, minor-tick visibility, and label styling
    are set through the ``ax.coords`` API instead. This helper bridges
    that gap so a single sph style call applies everywhere: the rc layers
    (:func:`set_base_style` / :func:`set_theme` / :func:`set_palette`) handle
    regular Axes, and ``style_wcs_axes(ax)`` carries the same intent onto
    a WCSAxes.

    Every parameter defaults to the corresponding *current* rcParam, so
    the canonical call after the rc layers is simply
    ``style_wcs_axes(ax)``.

    Parameters
    ----------
    ax : WCSAxes (or subclass)
        The axes to style. Plain Axes are passed through with a warning
        (they already obey rcParams).
    direction : {'in', 'out'}, optional
        Tick direction. Default ``rcParams['xtick.direction']``. WCSAxes
        does not support ``'inout'``; it is mapped to ``'in'``.
    major_size : float, optional
        Major tick length in points. Default ``rcParams['xtick.major.size']``.
    minor_size : float, optional
        Minor tick *length* in points. Opt-in: WCSAxes minor ticks otherwise
        inherit the major tick length, so this defaults to ``None`` (leave
        the inherited length untouched) rather than resolving an rcParam —
        passing it explicitly calls ``coord.ticks.set_minor_ticksize`` so
        minors render as short subdivisions instead of denser majors. (Tick
        *placement* is handled by ``minor_ticks`` below — independent of size.)
    width : float, optional
        Tick line width. Default ``rcParams['xtick.major.width']``.
    tick_color : color, optional
        Tick mark color. Default ``rcParams['xtick.color']``.
    minor_ticks : bool, optional
        Display minor ticks. Default ``rcParams['xtick.minor.visible']``.
        Minors subdivide whatever major ticks are present, leaving the major
        positions unchanged. This works even when the majors come from an
        explicit value list (the usual ``make_wcs_frame`` field-frame case),
        where astropy's native minor locator would otherwise place none — an
        interpolating minor locator is installed for those axes. Pass
        ``minor_ticks=False`` to opt out.
    minor_frequency : int, optional
        Minor-tick intervals per major interval (default 5, i.e. 4 minors
        between adjacent majors).
    labelcolor, labelsize : optional
        Tick-label color / size. Defaults ``rcParams['xtick.labelcolor']``
        (falling back to ``['xtick.color']``) and
        ``rcParams['xtick.labelsize']``.
    axislabel_color, axislabel_size : optional
        Axis-label (e.g. 'RA', 'Dec') color / size. Defaults
        ``rcParams['axes.labelcolor']`` / ``['axes.labelsize']``.
    frame_color, frame_linewidth : optional
        Color / linewidth of the (possibly curved) plot frame. Defaults
        ``rcParams['axes.edgecolor']`` / ``['axes.linewidth']``.
    grid : bool, default False
        Draw the coordinate grid with the styling below.
    grid_color, grid_linestyle, grid_linewidth, grid_alpha : optional
        Grid styling; default to the ``grid.*`` rcParams.
    stroke_lw, stroke_color : optional
        Add a contrasting stroke around the tick *marks* for legibility
        over imagery — the tick-mark counterpart of the stroke options on
        :func:`style_grid` (grid lines) and :func:`format_ticklabels` (tick
        labels). ``stroke_lw`` is the total stroke line width in points; both
        must be given for the stroke to apply (default ``None`` → no stroke).
        Implemented as a two-pass draw because astropy's tick renderer ignores
        ``path_effects``; set ``stroke_lw`` a little above the tick ``width``.
    coords : list of int or str, optional
        Which coordinates to style (e.g. ``[0, 1]`` or ``['ra', 'dec']``).
        Default: all coordinates on the axes.

    Returns
    -------
    ax
        The same axes, for chaining.

    Examples
    --------
    >>> sph.set_base_style(); sph.set_theme('dark_sky')
    >>> fig, ax = sph.make_wcs_frame(111, projection='AIT')
    >>> sph.style_wcs_axes(ax, grid=True)
    """
    if not _is_wcsaxes(ax):
        warnings.warn("style_wcs_axes: received a non-WCSAxes; regular "
                      "Axes already follow rcParams. Nothing applied.",
                      stacklevel=2)
        return ax

    # Resolve every unset parameter from the current rcParams, so the look
    # matches what regular Axes already inherited from the rc layers.
    if direction is None:
        direction = rcParams['xtick.direction']
    if direction == 'inout':   # not supported by WCSAxes
        direction = 'in'
    if major_size is None:
        major_size = rcParams['xtick.major.size']
    if width is None:
        width = rcParams['xtick.major.width']
    if tick_color is None:
        tick_color = rcParams['xtick.color']
    if minor_ticks is None:
        minor_ticks = rcParams['xtick.minor.visible']
    if labelcolor is None:
        labelcolor = rcParams.get('xtick.labelcolor', 'inherit')
        if labelcolor in ('inherit', None):
            labelcolor = rcParams['xtick.color']
    if labelsize is None:
        labelsize = rcParams['xtick.labelsize']
    if axislabel_color is None:
        axislabel_color = rcParams['axes.labelcolor']
    if axislabel_size is None:
        axislabel_size = rcParams['axes.labelsize']
    if frame_color is None:
        frame_color = rcParams['axes.edgecolor']
    if frame_linewidth is None:
        frame_linewidth = rcParams['axes.linewidth']
    if grid_color is None:
        grid_color = rcParams['grid.color']
    if grid_linestyle is None:
        grid_linestyle = rcParams['grid.linestyle']
    if grid_linewidth is None:
        grid_linewidth = rcParams['grid.linewidth']
    if grid_alpha is None:
        grid_alpha = rcParams['grid.alpha']

    if coords is None:
        coord_objs = list(ax.coords)
    else:
        coord_objs = [ax.coords[c] for c in coords]

    # Astropy's native minor-tick locator only subdivides a fixed major
    # spacing (``set_ticks(spacing=...)``); it returns nothing when the major
    # ticks come from an explicit value list — which is exactly what
    # make_wcs_frame's field frames use, so ``minor_ticks=True`` would silently
    # place no minors. This shared helper installs an interpolating minor
    # locator for value-list axes (and uses astropy's native path for
    # spacing-based ones), subdividing whatever majors are present WITHOUT
    # moving them. Lazy import to avoid any import-order coupling with ticks.
    from .ticks import _enable_minor_ticks_for_explicit_tick_values

    for coord in coord_objs:
        coord.set_ticks(size=major_size, width=width, color=tick_color,
                        direction=direction)
        if minor_ticks:
            freq = int(minor_frequency) if minor_frequency is not None else 5
            _enable_minor_ticks_for_explicit_tick_values(coord, frequency=freq)
        else:
            coord.display_minor_ticks(False)
            if minor_frequency is not None:
                coord.set_minor_frequency(int(minor_frequency))
        # Opt-in only: astropy's default minor length is the major length, so
        # set_minor_ticksize is called solely when minor_size is explicit —
        # an unconditional call would shrink minors for every existing caller.
        if minor_size is not None:
            coord_ticks(coord).set_minor_ticksize(minor_size)
        # Contrasting stroke on the tick marks (legibility over imagery),
        # matching the grid / tick-label stroke options. No-op unless both
        # stroke args are given.
        _apply_tick_stroke(coord, stroke_lw, stroke_color)
        coord.set_ticklabel(color=labelcolor, size=labelsize)
        # Restyle the existing axis-label text in place; set_axislabel is
        # the only public path to recolor/resize it, so re-apply the
        # current text with the new properties.
        try:
            lab = coord.get_axislabel()
            if lab:
                coord.set_axislabel(lab, color=axislabel_color,
                                    size=axislabel_size)
        except Exception:
            pass

    # Frame: handles curved / elliptical frames (AIT, MOL, ...) too.
    try:
        ax.coords.frame.set_color(frame_color)
        ax.coords.frame.set_linewidth(frame_linewidth)
    except Exception:
        pass

    if grid:
        ax.coords.grid(color=grid_color, linestyle=grid_linestyle,
                       linewidth=grid_linewidth, alpha=grid_alpha)

    # Overlay-drawn tick labels (all-sky in-frame / boundary labels from
    # add_overlay_ticks) are plain Text artists the coords API can't reach;
    # recolor any tagged by render_labels so a post-build restyle tracks the
    # native labels.
    for _txt in ax.texts:
        if getattr(_txt, '_sph_overlay_ticklabel', False):
            _txt.set_color(labelcolor)
    # Overlay tick *marks* (render_ticks Line2Ds) likewise — recolor to the
    # tick color, and additionally honor the resolved tick ``direction``.
    # astropy can't draw inward ticks on a curved spine, so on an all-sky
    # elliptical frame (AIT/MOL) these sph-drawn boundary ticks are the only
    # direction-controllable ones; redirect them from the stored anchor +
    # outward endpoint (set by render_ticks) so e.g. a 'structural' base
    # (xtick.direction='in') or an explicit direction='in' actually points
    # them inward. 'out' (the default) re-derives the identical segment.
    for _ln in ax.lines:
        if not getattr(_ln, '_sph_overlay_tick', False):
            continue
        _ln.set_color(tick_color)
        anchor = getattr(_ln, '_sph_tick_anchor', None)
        out_end = getattr(_ln, '_sph_tick_out_end', None)
        if anchor is None or out_end is None:
            continue
        ax_x, ax_y = anchor
        ox, oy = out_end
        if direction == 'in':
            _ln.set_data([ax_x, 2 * ax_x - ox], [ax_y, 2 * ax_y - oy])
        elif direction == 'both':
            _ln.set_data([2 * ax_x - ox, ox], [2 * ax_y - oy, oy])
        else:  # 'out'
            _ln.set_data([ax_x, ox], [ax_y, oy])

    return ax


def style_annotation(ax: Any, palette: str | dict[str, str],
                             grid: bool = True) -> dict[str, str]:
    """
    Apply an annotation / scaffolding palette to an Axes or WCSAxes:
    backgrounds, frame, tick & label colors, and the foreground grid.

    Dispatches to the right machinery for plain Axes vs. WCSAxes and
    returns the resolved palette dict, so callers can pull ``'accent'`` /
    ``'stars'`` / ``'label'`` / ``'compass'`` for subsequent elements.

    Parameters
    ----------
    ax : Axes or WCSAxes
    palette : str or dict
        A key in :data:`ANNOTATION_PALETTES`, or a dict with the same keys.
    grid : bool, default True
        Style/draw the foreground grid in ``palette['grid']``.

    Returns
    -------
    palette : dict
        The resolved palette (e.g. for ``palette['accent']`` afterward).
        A second overlay grid, if any, can be colored with
        ``palette['grid2']`` by the caller.

    Examples
    --------
    >>> pal = sph.style_annotation(ax, 'night')
    >>> ax.scatter(ra, dec, color=pal['stars'])
    """
    if isinstance(palette, str):
        key = _ANNOTATION_ALIASES.get(palette, palette)
        try:
            palette = ANNOTATION_PALETTES[key]
        except KeyError:
            raise ValueError(
                f"Unknown annotation palette {palette!r}. Available: "
                f"{', '.join(ANNOTATION_PALETTES)}") from None

    fig = ax.figure
    fig.patch.set_facecolor(palette['fig_bg'])
    ax.set_facecolor(palette['ax_bg'])

    if _is_wcsaxes(ax):
        style_wcs_axes(ax,
                       tick_color=palette['frame'],
                       labelcolor=palette['text2'],
                       axislabel_color=palette['text'],
                       frame_color=palette['frame'],
                       grid=grid, grid_color=palette['grid'])
    else:
        for s in ax.spines.values():
            s.set_color(palette['frame'])
        ax.tick_params(colors=palette['frame'],
                       labelcolor=palette['text2'], which='both')
        ax.xaxis.label.set_color(palette['text'])
        ax.yaxis.label.set_color(palette['text'])
        ax.title.set_color(palette['text'])
        if grid:
            ax.grid(True, color=palette['grid'])
            ax.set_axisbelow(True)

    return palette


# ---- Composable entry point -------------------------------------------

def set_style(base: str | dict[str, Any] | None = None,
              theme: str | dict[str, Any] | None = None,
              palette: str | list[str] | None = None,
              font: str | list[str] | dict[str, Any] | None = None,
              **rc_overrides: Any) -> None:
    """
    Apply any combination of the four style layers in one call.

    Each argument is independently optional, so you can compose the full
    look or apply just one layer on top of your own rcParams:

    * ``base`` — structural rcParams via :func:`set_base_style` (e.g.
      ``'standard'``, ``'journal'``, or your own dict of RC params).
    * ``theme`` — background/foreground coordination via
      :func:`set_theme` (e.g. ``'dark_sky'``, ``'twilight'``, any
      matplotlib built-in style name, or your own dict).
    * ``palette`` — the data-color cycle via :func:`set_palette` (e.g.
      ``'speakeasy'``, or an explicit list of colors).
    * ``font`` — the typeface stack + paired math fontset via
      :func:`set_font` (e.g. ``'journal'``, ``'web'``, a face, or a stack).
    * ``rc_overrides`` — any extra rcParams, applied last.

    Layers are applied base → theme → palette → font → overrides, so a
    later layer wins on any shared key.

    Note that WCSAxes ignore most tick rcParams; after building an all-sky
    frame, also call :func:`style_wcs_axes` to carry the look onto it.

    Examples
    --------
    >>> sph.set_style(base='standard', theme='dark_sky', palette='nightcap')
    >>> sph.set_style(palette='atlas')          # palette only
    >>> sph.set_style(theme='twilight', **{'axes.grid': True})
    """
    if base is not None:
        set_base_style(base)
    if theme is not None:
        set_theme(theme)
    if palette is not None:
        set_palette(palette)
    if font is not None:
        set_font(font)
    if rc_overrides:
        _rc_update(rc_overrides)
