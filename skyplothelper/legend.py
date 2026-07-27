"""Multi-dimensional legends — one compact block per visual channel.

Sky-plot maps routinely encode several data dimensions at once: marker
*shape*, *color*, *size*, *fill*, *line style*, and more. :class:`MultiLegend`
builds a legend as a stack of independent **channel blocks**, each mapping one
visual channel to one data meaning, and places the result anywhere on the
canvas — including off the axes frame, the common all-sky case that
:meth:`matplotlib.axes.Axes.legend` handles awkwardly.

The core is a single generic :class:`LegendBlock` whose entries carry
arbitrary matplotlib style dicts plus a ``swatch_kind`` describing how each
entry is drawn. The named block classes (:class:`ColorBlock`,
:class:`ShapeBlock`, :class:`LineBlock`, ...) and the fluent ``MultiLegend.add_*``
methods are thin wrappers over it, so any encoding channel is expressible
without a new class::

    (sph.MultiLegend(ax, loc='lower right')
        .add_color('Target', {'DDO 69': 'purple', 'DDO 70': 'C0'})
        .add_shape('Sample', {'DGS': 'o', 'KINGFISH': 'D'})   # auto-grayscale
        .add_line('Fit', {'MW': ':', 'Z-dep': '-.'})
        .draw())

The legend is built from :mod:`matplotlib.offsetbox` primitives (the same
machinery the native legend uses internally) and attached via
``ax.add_artist``, so it coexists with any other legend on the axes and works
on any :class:`~matplotlib.axes.Axes`, not only WCSAxes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import rcParams

from ._stroke import _stroke_path_effects

__all__ = [
    "MultiLegend",
    "LegendBlock",
    "ColorBlock",
    "ShapeBlock",
    "LineBlock",
    "SizeBlock",
    "EdgeBlock",
    "FillBlock",
    "AlphaBlock",
    "OrientBlock",
    "RegionBlock",
    "TextBlock",
    "ColorbarBlock",
    "GlyphBlock",
    "register_glyph",
    "list_glyphs",
]

# Swatch kinds. Each maps to a renderer in _make_swatch (except 'text', which
# renders label-only in _render_entry). New channels are new property keys on
# existing kinds, not new kinds — the set rarely needs to grow.
_SWATCH_KINDS = ("marker", "patch", "line", "region", "text", "custom", "glyph")

# Location names shared with the native legend, passed straight to
# AnchoredOffsetbox's ``loc``.
_LOC_NAMES: frozenset[str] = frozenset({
    "upper right", "upper left", "lower left", "lower right", "right",
    "center left", "center right", "lower center", "upper center", "center",
})

# "outside <side>" presets → (anchor point in axes fraction, loc name of the
# box corner pinned to that point). Anchoring in axes fraction just outside
# [0, 1] places the legend in the figure margin without needing a resolved
# layout (unlike reading ax.get_position()), and it tracks the axes if it
# later moves.
_OUTSIDE_PRESETS: dict[str, tuple[tuple[float, float], str]] = {
    "outside right":        ((1.02, 0.5), "center left"),
    "outside left":         ((-0.02, 0.5), "center right"),
    "outside top":          ((0.5, 1.02), "lower center"),
    "outside bottom":       ((0.5, -0.02), "upper center"),
    "outside upper right":  ((1.02, 1.0), "upper left"),
    "outside lower right":  ((1.02, 0.0), "lower left"),
    "outside upper left":   ((-0.02, 1.0), "upper right"),
    "outside lower left":   ((-0.02, 0.0), "lower right"),
    "outside top right":    ((1.0, 1.02), "lower right"),
    "outside top left":     ((0.0, 1.02), "lower left"),
    "outside bottom right": ((1.0, -0.02), "upper right"),
    "outside bottom left":  ((0.0, -0.02), "upper left"),
}

# Neutral grays used when a shape/fill block shares a legend with a color
# block, so the shape dimension does not read as "another color" (the
# convention plot_catalog's size legend already follows).
_NEUTRAL_SWATCH = "0.4"

# Default geometry, in points. Swatch box height tracks the label font size.
_DEFAULT_FONTSIZE = 10.0
_SWATCH_SCALE = 1.3      # swatch box height = fontsize * this
_LINE_ASPECT = 2.2       # line-swatch width = box height * this
_MARKER_FRAC = 0.72      # default marker diameter = box height * this


# ---------------------------------------------------------------------------
# Named-glyph registry — draw sph's domain glyphs as legend swatches by name.
#
# A builder is ``fn(center, size, color, lw) -> list[Artist]`` drawing the
# glyph centered at ``center`` within radius ~``size`` (point coords). The
# registry is the single place both a GlyphBlock and (future, cluster "B")
# the plotting side would resolve a glyph name from — so a plotted glyph and
# its legend swatch stay identical. Seeded with the reticle shapes (the
# cleanly reusable single-artist glyphs); register your own with
# register_glyph(). The instrument icons (antenna/telescope/dome) are built in
# a placement-dependent world frame and are deferred to the shared-registry
# refactor.
# ---------------------------------------------------------------------------

_GLYPH_REGISTRY: dict[str, Any] = {}


def register_glyph(name: str, builder: Any) -> None:
    """Register a named glyph for use in a :class:`GlyphBlock` / ``add_glyph``.

    ``builder`` is a callable ``fn(center, size, color, lw) -> list`` of
    matplotlib artists drawing the glyph centered at ``center`` (an ``(x, y)``
    point) within roughly ``size`` points, in the given ``color`` / linewidth.
    """
    _GLYPH_REGISTRY[str(name)] = builder


def list_glyphs() -> list[str]:
    """Names of the registered legend glyphs."""
    return sorted(_GLYPH_REGISTRY)


def _reticle_glyph_builder(style: str) -> Any:
    """A glyph builder that draws a reticle ``style`` (plus/x/L/circle)."""
    def build(center: Any, size: float, color: Any, lw: float) -> list[Any]:
        from matplotlib.lines import Line2D

        from .overlays.reticle import _reticle_segments
        cx, cy = center
        segs = _reticle_segments(style, size, size * 0.32, 0.0, 48, 0.0)
        return [Line2D([cx + x for x, y in seg], [cy + y for x, y in seg],
                       color=color, linewidth=lw, solid_capstyle="round")
                for seg in segs]
    return build


for _style, _alias in [("plus", "crosshair"), ("x", "crosshair_x"),
                       ("L", "corner"), ("circle", "target")]:
    _GLYPH_REGISTRY[f"reticle_{_style}"] = _reticle_glyph_builder(_style)
    _GLYPH_REGISTRY[_alias] = _GLYPH_REGISTRY[f"reticle_{_style}"]


def _normalize_entries(
    entries: Any, vary: str | None
) -> list[tuple[str, dict[str, Any]]]:
    """Return ``entries`` as an ordered ``[(label, style_dict), ...]`` list.

    ``entries`` may be a ``{label: value}`` dict or a list of ``(label, value)``
    pairs. Each ``value`` is either a full style dict, or a scalar that is
    wrapped as ``{vary: value}`` when a ``vary`` prop name is given (the path
    the named wrappers use). A scalar with no ``vary`` is an error — the
    generic block cannot know which property it sets.
    """
    if isinstance(entries, dict):
        items: list[tuple[str, Any]] = list(entries.items())
    else:
        items = [tuple(pair) for pair in entries]

    out: list[tuple[str, dict[str, Any]]] = []
    for label, value in items:
        if isinstance(value, dict):
            style = dict(value)
        elif vary is not None:
            style = {vary: value}
        else:
            raise TypeError(
                f"entry {label!r}: value must be a style dict for a generic "
                f"LegendBlock (got {value!r}); use a named add_* wrapper, or "
                "pass vary= to name the property this scalar sets")
        out.append((str(label), style))
    return out


class LegendBlock:
    """One channel block: a titled group of entries drawn as swatches.

    This is the generic core the named blocks and ``MultiLegend.add_*`` methods
    build on. Entries are ``{label: style_dict}`` (or a list of pairs), the
    block carries a ``swatch_kind`` describing how each entry is rendered, and
    ``base_style`` holds the properties shared by every entry (the block's
    constant channels).

    Parameters
    ----------
    title : str or None
        Block heading; ``None`` draws no title.
    entries : dict or list of (label, value)
        Per-entry style. ``value`` is a style dict, or a scalar wrapped as
        ``{vary: value}`` when ``vary`` is given.
    swatch_kind : {'marker', 'patch', 'line'}
        How each entry is drawn.
    vary : str, optional
        Property name that a scalar entry value sets (used by the wrappers).
    base_style : dict, optional
        Style shared by all entries; per-entry style overrides it.
    ncol : int, default 1
        Number of entry columns.
    orientation : {'vertical', 'horizontal'}
        Lay entries out in a column (default) or a single row.
    """

    def __init__(
        self,
        title: str | None,
        entries: Any,
        *,
        swatch_kind: str = "marker",
        vary: str | None = None,
        base_style: dict[str, Any] | None = None,
        ncol: int = 1,
        orientation: str = "vertical",
    ) -> None:
        if swatch_kind not in _SWATCH_KINDS:
            raise ValueError(
                f"swatch_kind must be one of {_SWATCH_KINDS!r} "
                f"(got {swatch_kind!r})")
        if orientation not in ("vertical", "horizontal"):
            raise ValueError(
                f"orientation must be 'vertical' or 'horizontal' "
                f"(got {orientation!r})")
        self.title = title
        self.entries = _normalize_entries(entries, vary)
        self.swatch_kind = swatch_kind
        self.base_style = dict(base_style or {})
        self.ncol = max(1, int(ncol))
        self.orientation = orientation
        # Whether this block's face color was left to the caller — a
        # color-varying sibling turns an unset shape/fill swatch neutral gray.
        self._accepts_neutral = False

    # -- rendering -------------------------------------------------------
    def _resolved_style(self, entry_style: dict[str, Any]) -> dict[str, Any]:
        """Merge base + entry style for one entry (entry wins)."""
        return {**self.base_style, **entry_style}

    def _render(
        self,
        *,
        text_color: Any,
        title_color: Any,
        fontsize: float,
        title_fontsize: float,
        stroke: list[Any] | None,
        swatch_h: float,
        entry_sep: float,
        title_sep: float,
    ) -> Any:
        """Build this block's :class:`~matplotlib.offsetbox.OffsetBox`."""
        from matplotlib.offsetbox import TextArea, VPacker

        entry_boxes = [
            self._render_entry(label, style, text_color=text_color,
                               fontsize=fontsize, stroke=stroke,
                               swatch_h=swatch_h)
            for label, style in self.entries
        ]
        entries_box = self._pack_entries(entry_boxes, entry_sep)

        if self.title is None:
            return entries_box
        title_props = dict(color=title_color, size=title_fontsize,
                           weight="bold")
        if stroke is not None:
            title_props["path_effects"] = stroke
        title_area = TextArea(self.title, textprops=title_props)
        return VPacker(children=[title_area, entries_box],
                       align="left", pad=0, sep=title_sep)

    def _pack_entries(self, entry_boxes: list[Any], sep: float) -> Any:
        """Arrange rendered entries by orientation / ncol."""
        from matplotlib.offsetbox import HPacker, VPacker

        if self.orientation == "horizontal":
            return HPacker(children=entry_boxes, align="center", pad=0, sep=sep)
        if self.ncol <= 1:
            return VPacker(children=entry_boxes, align="left", pad=0, sep=sep)
        # Grid: fill column-major so entries read top-to-bottom then across,
        # matching how a reader scans a multi-column key.
        n = len(entry_boxes)
        per_col = int(np.ceil(n / self.ncol))
        columns: list[Any] = [
            VPacker(children=entry_boxes[c * per_col:(c + 1) * per_col],
                    align="left", pad=0, sep=sep)
            for c in range(self.ncol)
            if entry_boxes[c * per_col:(c + 1) * per_col]
        ]
        return HPacker(children=columns, align="top", pad=0, sep=sep * 2.5)

    def _render_entry(
        self,
        label: str,
        entry_style: dict[str, Any],
        *,
        text_color: Any,
        fontsize: float,
        stroke: list[Any] | None,
        swatch_h: float,
    ) -> Any:
        """One entry = swatch + label, packed side by side.

        A ``text`` entry is label-only (a free note), so it renders as just the
        text with no swatch.
        """
        from matplotlib.offsetbox import HPacker, TextArea

        text_props = dict(color=text_color, size=fontsize)
        if stroke is not None:
            text_props["path_effects"] = stroke
        text = TextArea(label, textprops=text_props)
        if self.swatch_kind == "text":
            return text
        swatch = _make_swatch(self.swatch_kind, self._resolved_style(entry_style),
                              swatch_h, stroke)
        return HPacker(children=[swatch, text], align="center",
                       pad=0, sep=0.4 * swatch_h)


def _make_swatch(
    kind: str, style: dict[str, Any], swatch_h: float, stroke: list[Any] | None
) -> Any:
    """Draw one swatch of ``kind`` into a sized :class:`DrawingArea`.

    Style keys are matplotlib artist properties: ``marker``, ``facecolor``,
    ``edgecolor``, ``markersize`` (diameter, points), ``linewidth``,
    ``linestyle``, ``hatch``, ``alpha``, ``color``.
    """
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import DrawingArea
    from matplotlib.patches import Rectangle

    alpha = style.get("alpha")
    art: Any

    if kind == "glyph":
        # A named sph glyph (reticle shapes, or any register_glyph'd builder),
        # drawn from the shared registry so it matches the plotted glyph.
        name = style.get("glyph")
        builder = _GLYPH_REGISTRY.get(name) if isinstance(name, str) else None
        if builder is None:
            raise ValueError(
                f"unknown glyph {name!r}; register one with register_glyph() "
                f"or use one of {list_glyphs()}")
        color = style.get("color", style.get("facecolor", "C0"))
        lw = float(style.get("linewidth", 1.6))
        radius = float(style.get("markersize", swatch_h * 0.42))
        box = max(swatch_h, radius * 2.4)
        da = DrawingArea(box, box, 0.0, 0.0)
        for art in builder((box / 2.0, box / 2.0), radius, color, lw):
            if stroke is not None:
                art.set_path_effects(stroke)
            da.add_artist(art)
        return da
    if kind == "custom":
        # Escape hatch: the caller hands in a ready matplotlib artist, drawn in
        # a swatch_h-sized box in point coords (also covers sph Path glyphs).
        handle = style.get("handle")
        if handle is None:
            raise ValueError("custom swatch entry needs a 'handle' artist")
        da = DrawingArea(swatch_h, swatch_h, 0.0, 0.0)
        art = handle
    elif kind == "marker":
        from matplotlib.markers import MarkerStyle
        mk = style.get("marker", "o")
        angle = style.get("angle")
        if angle is not None:
            from matplotlib.transforms import Affine2D
            mk = MarkerStyle(mk, transform=Affine2D().rotate_deg(float(angle)))
        face = style.get("facecolor", style.get("color", "C0"))
        edge = style.get("edgecolor", "none")
        # Unfilled glyphs (+ x | _ 1 2 3 4) draw with the edge/line color and
        # ignore the face, so an edge of 'none' would make them invisible —
        # color them with the face color instead.
        if edge in ("none", None) and not MarkerStyle(mk).is_filled():
            edge = face
        ms = float(style.get("markersize", swatch_h * _MARKER_FRAC))
        # Grow the box so a large graduated marker (BVID-style size legend)
        # neither clips nor overlaps its neighbors; small markers keep the
        # default box so ordinary blocks stay compact.
        box = max(swatch_h, ms * 1.3)
        da = DrawingArea(box, box, 0.0, 0.0)
        art = Line2D(
            [box / 2.0], [box / 2.0], linestyle="none",
            marker=mk, markersize=ms,
            markerfacecolor=face, markeredgecolor=edge,
            markeredgewidth=float(style.get("linewidth", 1.0)), alpha=alpha)
    elif kind in ("patch", "region"):
        da = DrawingArea(swatch_h, swatch_h, 0.0, 0.0)
        pad = swatch_h * 0.12
        face = style.get("facecolor", style.get("color", "C0"))
        # A region swatch reads as a translucent footprint with a visible
        # border (edge defaults to the fill color); a patch is a solid chip.
        if kind == "region":
            edge = style.get("edgecolor", face)
            lw = float(style.get("linewidth", 1.2))
            fill_alpha = alpha if alpha is not None else 0.35
        else:
            edge = style.get("edgecolor", "none")
            lw = float(style.get("linewidth", 0.0))
            fill_alpha = alpha
        art = Rectangle(
            (pad, pad), swatch_h - 2 * pad, swatch_h - 2 * pad,
            facecolor=face, edgecolor=edge, linewidth=lw,
            hatch=style.get("hatch"), alpha=fill_alpha)
    elif kind == "line":
        box_w = swatch_h * _LINE_ASPECT
        da = DrawingArea(box_w, swatch_h, 0.0, 0.0)
        y = swatch_h / 2.0
        color = style.get("color", style.get("facecolor", "C0"))
        art = Line2D(
            [box_w * 0.05, box_w * 0.95], [y, y],
            linestyle=style.get("linestyle", "-"),
            linewidth=float(style.get("linewidth", 1.8)),
            color=color, alpha=alpha,
            marker=style.get("marker", ""),
            markersize=float(style.get("markersize", swatch_h * _MARKER_FRAC)),
            markerfacecolor=color, markeredgecolor=color)
    else:  # pragma: no cover - guarded by LegendBlock.__init__
        raise ValueError(f"unknown swatch kind {kind!r}")

    if stroke is not None:
        art.set_path_effects(stroke)
    da.add_artist(art)
    return da


def _neutral_gray(facecolor: Any) -> str:
    """Pick a neutral swatch gray that reads against ``facecolor``.

    Shape/size swatches use a neutral gray so they don't imply a color
    category; the tone is mode-aware — a lighter gray on a dark legend
    background, the standard dark gray on a light one.
    """
    from matplotlib.colors import to_rgb
    try:
        r, g, b = to_rgb(facecolor)
    except (ValueError, TypeError):
        return _NEUTRAL_SWATCH
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "0.7" if luminance < 0.4 else _NEUTRAL_SWATCH


# ---------------------------------------------------------------------------
# Named block wrappers
# ---------------------------------------------------------------------------

class ColorBlock(LegendBlock):
    """A block where color is the varied channel.

    ``swatch`` selects how each color entry is drawn: ``'patch'`` (default) a
    filled color chip that reads as its own dimension, ``'line'`` a thick color
    segment, or ``'marker'`` a colored marker glyph (for when the data really
    are that marker, e.g. circles). ``target`` colors the marker face
    (default) or edge.
    """

    def __init__(
        self, title: str | None, entries: Any, *,
        swatch: str = "patch", target: str = "face", marker: str = "o",
        size: float | None = None, lw: float | None = None, ncol: int = 1,
        **base: Any,
    ) -> None:
        if swatch not in ("patch", "marker", "line"):
            raise ValueError(
                f"swatch must be 'patch', 'marker' or 'line' (got {swatch!r})")
        if target not in ("face", "edge"):
            raise ValueError(f"target must be 'face' or 'edge' (got {target!r})")
        vary = "edgecolor" if (swatch == "marker" and target == "edge") else \
               ("facecolor" if swatch != "line" else "color")
        base_style = dict(base)
        if swatch == "marker":
            base_style.setdefault("marker", marker)
            if size is not None:
                base_style["markersize"] = float(np.sqrt(size))
            if target == "edge":
                base_style.setdefault("facecolor", "none")
        elif swatch == "line" and lw is not None:
            base_style["linewidth"] = lw
        super().__init__(title, entries,
                         swatch_kind=("marker" if swatch == "marker" else
                                      ("line" if swatch == "line" else "patch")),
                         vary=vary, base_style=base_style, ncol=ncol)


class ShapeBlock(LegendBlock):
    """A block where marker shape is the varied channel.

    ``color`` is the shared swatch color. Left unset, the swatches turn neutral
    gray whenever the legend also carries a color block, so shape reads as its
    own dimension rather than as another color.
    """

    def __init__(
        self, title: str | None, entries: Any, *,
        color: Any = None, size: float | None = None, ncol: int = 1,
        **base: Any,
    ) -> None:
        base_style = dict(base)
        base_style.setdefault("facecolor", color if color is not None else "C0")
        base_style.setdefault("edgecolor", "none")
        if size is not None:
            base_style["markersize"] = float(np.sqrt(size))
        super().__init__(title, entries, swatch_kind="marker", vary="marker",
                         base_style=base_style, ncol=ncol)
        self._accepts_neutral = color is None


class LineBlock(LegendBlock):
    """A block of line entries, varying dash style (default) or width.

    ``vary='linestyle'`` (default) maps scalar entry values to line dash
    patterns; ``vary='lw'`` (or ``'linewidth'``) maps them to line widths for a
    weight-encoding legend. Pass ``marker=`` to put a marker on each segment.
    """

    def __init__(
        self, title: str | None, entries: Any, *,
        vary: str = "linestyle", color: Any = None, lw: float | None = None,
        marker: Any = None, ncol: int = 1,
    ) -> None:
        prop = "linewidth" if vary in ("lw", "linewidth") else "linestyle"
        base_style: dict[str, Any] = {}
        if color is not None:
            base_style["color"] = color
        if lw is not None and prop != "linewidth":
            base_style["linewidth"] = lw
        if marker is not None:
            base_style["marker"] = marker
        super().__init__(title, entries, swatch_kind="line", vary=prop,
                         base_style=base_style, ncol=ncol)


class SizeBlock(LegendBlock):
    """A graduated-marker block: marker size encodes a quantity.

    Representative markers are sized by the *same* scaling a scatter used, so
    the legend can't disagree with the plot. Supply the raw ``values`` to
    show (labeled in the data's own units), or a ``num`` to auto-pick evenly
    spaced representatives from a plot's ``size_map``. The mapping params
    (``smin``/``smax`` output area, ``rmin``/``rmax`` scaled bounds,
    ``scale``) come from :meth:`from_catalog` in the usual case; provide them
    explicitly for a standalone legend.

    Marker area follows ``s = smin + (smax-smin)*(scale(v)-rmin)/(rmax-rmin)``
    and the swatch draws it at ``markersize = sqrt(s)`` (scatter ``s`` is an
    area in points², a Line2D marker is a diameter) — matching plotted sizes.
    """

    def __init__(
        self, title: str | None, *,
        values: Any = None, num: int = 4, nice: bool = True,
        size_map: tuple[Any, Any, float, float] | None = None,
        smin: float = 10.0, smax: float = 200.0,
        rmin: float | None = None, rmax: float | None = None,
        scale: Any = "linear", color: Any = None, marker: str = "o",
        fmt: Any = None, ncol: int = 1, orientation: str = "vertical",
    ) -> None:
        from .data_plots import _apply_size_scale

        if values is None:
            if size_map is None:
                raise ValueError(
                    "SizeBlock needs either values= or a size_map "
                    "(use SizeBlock.from_catalog to pull it off a plot)")
            raw, scaled, m0, m1 = size_map
            raw = np.asarray(raw, dtype=float)
            scaled = np.asarray(scaled, dtype=float)
            rlo, rhi = float(np.nanmin(raw)), float(np.nanmax(raw))
            if nice and (candidates := _nice_values(rlo, rhi, max(2, num))):
                # Zero-config: round 1/2/5-decade values across the data range,
                # so the key reads 1/5/10/50 rather than raw min/mean/max.
                values = candidates
            else:
                # Evenly spaced in the scaled domain, labels recovered by
                # interpolation on the actual (scaled, raw) pairs — inverts any
                # monotonic scale, incl. callables, with no analytic inverse.
                order = np.argsort(scaled)
                targets = np.linspace(m0, m1, max(1, num))
                values = np.interp(targets, scaled[order], raw[order])
            if rmin is None:
                rmin = m0
            if rmax is None:
                rmax = m1

        vals = np.atleast_1d(np.asarray(values, dtype=float))
        scaled_vals = _apply_size_scale(vals, scale)
        lo = rmin if rmin is not None else float(np.min(scaled_vals))
        hi = rmax if rmax is not None else float(np.max(scaled_vals))
        if hi > lo:
            areas = smin + (smax - smin) * (scaled_vals - lo) / (hi - lo)
        else:
            areas = np.full_like(scaled_vals, (smin + smax) / 2.0)
        diameters = np.sqrt(np.clip(areas, 0.0, None))

        def _label(v: float) -> str:
            if callable(fmt):
                return str(fmt(v))
            return format(v, fmt if isinstance(fmt, str) else "g")

        swatch_color = color if color is not None else _NEUTRAL_SWATCH
        entries = [
            (_label(float(v)),
             {"markersize": float(d), "marker": marker,
              "facecolor": swatch_color, "edgecolor": "none"})
            for v, d in zip(vals, diameters)
        ]
        super().__init__(title, entries, swatch_kind="marker",
                         base_style={}, ncol=ncol, orientation=orientation)
        # Size is its own dimension → its neutral gray is mode-aware unless the
        # caller chose a color.
        self._size_neutral = color is None

    @classmethod
    def from_catalog(
        cls, result: Any, *, values: Any = None, num: int = 4, nice: bool = True,
        title: str | None = None, color: Any = None, marker: str = "o",
        fmt: Any = None, ncol: int = 1, orientation: str = "vertical",
    ) -> "SizeBlock":
        """Build a :class:`SizeBlock` from a :func:`~skyplothelper.plot_catalog`
        result, reproducing that plot's exact marker sizes.

        ``result`` is the object ``plot_catalog`` returned (a bare
        ``PathCollection`` or a ``CatalogPlot``) — it must have been called
        with ``sizeby=``. With no ``values=``, round 1/2/5-decade
        representatives are auto-picked (``nice``).
        """
        info = _extract_size_info(result)
        _, _, rmin, rmax = info.size_map
        return cls(title, values=values, num=num, nice=nice,
                   size_map=info.size_map, smin=info.smin, smax=info.smax,
                   rmin=rmin, rmax=rmax, scale=info.size_scale, color=color,
                   marker=marker, fmt=fmt, ncol=ncol, orientation=orientation)


def _extract_size_info(result: Any) -> Any:
    """Pull the stashed ``_sph_size_info`` off a plot_catalog result."""
    scatter = getattr(result, "scatter", result)
    info = getattr(scatter, "_sph_size_info", None)
    if info is None:
        raise ValueError(
            "no size scaling found on this plot_catalog result — call "
            "plot_catalog(..., sizeby=<column>) so a size legend can match it")
    return info


def _format_value(v: float, fmt: Any) -> str:
    """Format a numeric entry label (callable, format string, or default 'g')."""
    if callable(fmt):
        return str(fmt(v))
    return format(v, fmt if isinstance(fmt, str) else "g")


def _nice_values(lo: float, hi: float, n: int = 5) -> list[float]:
    """Round 1/2/5-decade values spanning ``[lo, hi]`` for a graduated key.

    Returns the "nice" numbers (1, 2, 5 × 10^k) inside the data range, so a
    size/alpha legend reads 1/5/10/20/50 rather than the raw min/mean/max.
    Falls back to the plain endpoints when the range is degenerate or not
    strictly positive (log-decade steps need lo > 0).
    """
    import math
    if not (hi > lo) or lo <= 0:
        return [lo, hi] if hi > lo else [lo]
    k0 = math.floor(math.log10(lo))
    k1 = math.ceil(math.log10(hi))
    cands = [m * 10.0 ** k for k in range(k0, k1 + 1) for m in (1, 2, 5)]
    cands = sorted({v for v in cands if lo <= v <= hi})
    if not cands:
        return [lo, hi]
    if len(cands) > n:
        idx = sorted({int(round(i)) for i in np.linspace(0, len(cands) - 1, n)})
        cands = [cands[i] for i in idx]
    return cands


class EdgeBlock(LegendBlock):
    """A block where marker *edge* color is the varied channel.

    Lets a single marker encode a second category on its rim (face = one
    dimension, edge = another). Swatches are neutral-filled rings so the edge
    color reads.
    """

    def __init__(
        self, title: str | None, entries: Any, *, marker: str = "o",
        facecolor: Any = "0.85", size: float | None = None, lw: float = 1.6,
        ncol: int = 1, **base: Any,
    ) -> None:
        base_style = dict(base)
        base_style.setdefault("marker", marker)
        base_style.setdefault("facecolor", facecolor)
        base_style.setdefault("linewidth", lw)
        if size is not None:
            base_style["markersize"] = float(np.sqrt(size))
        super().__init__(title, entries, swatch_kind="marker", vary="edgecolor",
                         base_style=base_style, ncol=ncol)


_HATCH_CHARS = set("/\\|-+xoO.")


def _is_hatch(spec: Any) -> bool:
    """A fill spec is a hatch pattern if it's a string of hatch glyphs."""
    return (isinstance(spec, str) and spec.lower() not in ("filled", "solid",
            "open") and bool(spec) and all(c in _HATCH_CHARS for c in spec))


def _fill_style(spec: Any, color: Any) -> dict[str, Any]:
    """Translate a fill spec ('filled'/'open'/hatch) to a style dict."""
    if isinstance(spec, str) and spec.lower() in ("filled", "solid"):
        return {"facecolor": color}
    if isinstance(spec, str) and spec.lower() == "open":
        return {"facecolor": "none", "edgecolor": color, "linewidth": 1.5}
    if _is_hatch(spec):
        return {"facecolor": "none", "edgecolor": color, "hatch": spec,
                "linewidth": 0.8}
    raise ValueError(
        f"fill spec must be 'filled', 'open', or a hatch string; got {spec!r}")


class FillBlock(LegendBlock):
    """A block where fill state / texture is the varied channel.

    Each entry value is ``'filled'``, ``'open'``, or a hatch string (e.g.
    ``'///'``). ``kind='marker'`` draws filled/open markers (the DGR
    "Dust-Gas vs Dust-HI" case); ``kind='patch'`` draws chips and is required
    for hatch patterns (matplotlib markers don't hatch).
    """

    def __init__(
        self, title: str | None, entries: Any, *, kind: str = "marker",
        marker: str = "o", color: Any = "0.4", size: float | None = None,
        ncol: int = 1,
    ) -> None:
        items = list(entries.items()) if isinstance(entries, dict) else list(entries)
        if any(_is_hatch(spec) for _lbl, spec in items):
            kind = "patch"                       # hatch only renders on patches
        built: list[tuple[str, dict[str, Any]]] = []
        for label, spec in items:
            style = _fill_style(spec, color)
            if kind == "marker":
                style.setdefault("marker", marker)
                if size is not None:
                    style["markersize"] = float(np.sqrt(size))
            built.append((str(label), style))
        super().__init__(title, built,
                         swatch_kind="patch" if kind == "patch" else "marker",
                         ncol=ncol)


class AlphaBlock(LegendBlock):
    """A block where opacity encodes a quantity (or a category).

    Give ``values=`` (mapped onto ``[amin, amax]`` and labeled in data units)
    for a graduated-opacity key, or ``entries={label: alpha}`` for explicit
    levels.
    """

    def __init__(
        self, title: str | None, *, values: Any = None, entries: Any = None,
        amin: float = 0.2, amax: float = 1.0, color: Any = "0.4",
        marker: str = "o", fmt: Any = None, ncol: int = 1,
    ) -> None:
        built: list[tuple[str, dict[str, Any]]] = []
        if entries is not None:
            items = list(entries.items()) if isinstance(entries, dict) else list(entries)
            for label, a in items:
                built.append((str(label),
                              {"alpha": float(a), "marker": marker,
                               "facecolor": color, "edgecolor": "none"}))
        elif values is not None:
            vals = np.atleast_1d(np.asarray(values, dtype=float))
            lo, hi = float(np.min(vals)), float(np.max(vals))
            for v in vals:
                a = amin if hi <= lo else amin + (amax - amin) * (v - lo) / (hi - lo)
                built.append((_format_value(float(v), fmt),
                              {"alpha": float(a), "marker": marker,
                               "facecolor": color, "edgecolor": "none"}))
        else:
            raise ValueError("AlphaBlock needs values= or entries=")
        super().__init__(title, built, swatch_kind="marker", ncol=ncol)


class OrientBlock(LegendBlock):
    """A block where marker rotation encodes an angle (position angle,
    polarization). Each entry value is an angle in degrees."""

    def __init__(
        self, title: str | None, entries: Any, *, marker: str = "^",
        color: Any = "0.4", size: float | None = None, ncol: int = 1,
    ) -> None:
        base_style: dict[str, Any] = {"marker": marker, "facecolor": color,
                                      "edgecolor": "none"}
        if size is not None:
            base_style["markersize"] = float(np.sqrt(size))
        super().__init__(title, entries, swatch_kind="marker", vary="angle",
                         base_style=base_style, ncol=ncol)


_REGION_KEY_ALIASES = {"fc": "facecolor", "ec": "edgecolor"}


class RegionBlock(LegendBlock):
    """A block of translucent filled-region swatches (survey footprints,
    exclusion zones). Each entry value is a color, or a style dict with
    ``facecolor``/``fc``, ``edgecolor``/``ec``, ``alpha``, ``hatch``."""

    def __init__(self, title: str | None, entries: Any, *, ncol: int = 1) -> None:
        items = list(entries.items()) if isinstance(entries, dict) else list(entries)
        built: list[tuple[str, dict[str, Any]]] = []
        for label, val in items:
            style: dict[str, Any]
            if isinstance(val, dict):
                style = {str(_REGION_KEY_ALIASES.get(k, k)): v
                         for k, v in val.items()}
            else:
                style = {"facecolor": val}
            built.append((str(label), style))
        super().__init__(title, built, swatch_kind="region", ncol=ncol)


class TextBlock(LegendBlock):
    """A block of free-text notes (no swatches) — e.g. "dashed = model"."""

    def __init__(self, title: str | None, notes: Any, *, ncol: int = 1) -> None:
        if isinstance(notes, str):
            notes = [notes]
        entries: list[tuple[str, dict[str, Any]]] = [(str(n), {}) for n in notes]
        super().__init__(title, entries, swatch_kind="text", ncol=ncol)


class ColorbarBlock(LegendBlock):
    """A compact continuous-color gradient strip, for when color encodes a
    continuous quantity and should sit *in* the legend stack rather than as a
    separate axes colorbar. Renders a horizontal gradient with end labels.

    Parameters
    ----------
    title : str or None
    cmap : str or Colormap
    vmin, vmax : float
        Range the strip spans (drives the end labels).
    length : float, optional
        Bar length in points (default ~8× the swatch height).
    fmt : str or callable, optional
        End-label format (default ``'g'``).
    """

    def __init__(
        self, title: str | None, *, cmap: Any = "viridis",
        vmin: float = 0.0, vmax: float = 1.0, n: int = 64,
        length: float | None = None, fmt: Any = None,
    ) -> None:
        super().__init__(title, {}, swatch_kind="patch")
        self.cmap = cmap
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.n = max(2, int(n))
        self.length = length
        self.fmt = fmt

    def _render(
        self, *, text_color: Any, title_color: Any, fontsize: float,
        title_fontsize: float, stroke: list[Any] | None, swatch_h: float,
        entry_sep: float, title_sep: float,
    ) -> Any:
        import matplotlib.pyplot as plt
        from matplotlib.offsetbox import DrawingArea, HPacker, TextArea, VPacker
        from matplotlib.patches import Rectangle

        cmap = plt.get_cmap(self.cmap) if isinstance(self.cmap, str) else self.cmap
        bar_w = self.length if self.length is not None else swatch_h * 8.0
        da = DrawingArea(bar_w, swatch_h, 0.0, 0.0)
        step = bar_w / self.n
        for i in range(self.n):
            da.add_artist(Rectangle(
                (i * step, 0.0), step + 0.5, swatch_h,
                facecolor=cmap(i / (self.n - 1)), edgecolor="none"))
        border = Rectangle(
            (0.0, 0.0), bar_w, swatch_h, facecolor="none",
            edgecolor=text_color, linewidth=0.6)
        # The bar border is a swatch-equivalent artist — stroke it too when a
        # stroke is set, so it matches the outlined swatches other blocks draw.
        if stroke is not None:
            border.set_path_effects(stroke)
        da.add_artist(border)

        tp = dict(color=text_color, size=fontsize * 0.9)
        if stroke is not None:
            tp["path_effects"] = stroke
        lo = TextArea(_format_value(self.vmin, self.fmt), textprops=tp)
        hi = TextArea(_format_value(self.vmax, self.fmt), textprops=tp)
        bar_row = HPacker(children=[lo, da, hi], align="center", pad=0,
                          sep=0.3 * swatch_h)
        if self.title is None:
            return bar_row
        title_props = dict(color=title_color, size=title_fontsize, weight="bold")
        if stroke is not None:
            title_props["path_effects"] = stroke
        return VPacker(children=[TextArea(self.title, textprops=title_props),
                                 bar_row], align="left", pad=0, sep=title_sep)


class GlyphBlock(LegendBlock):
    """A block whose swatches are named sph glyphs (reticle shapes, or any
    :func:`register_glyph`'d builder), resolved from the shared glyph registry
    so the legend swatch matches the plotted glyph.

    Each entry value is a glyph name (see :func:`list_glyphs`), e.g.
    ``{'target': 'reticle_circle', 'candidate': 'crosshair'}``. ``color`` /
    ``lw`` / ``size`` set the shared swatch style.
    """

    def __init__(
        self, title: str | None, entries: Any, *, color: Any = None,
        lw: float = 1.6, size: float | None = None, ncol: int = 1,
    ) -> None:
        base_style: dict[str, Any] = {"linewidth": lw}
        if color is not None:
            base_style["color"] = color
        if size is not None:
            base_style["markersize"] = float(size)
        super().__init__(title, entries, swatch_kind="glyph", vary="glyph",
                         base_style=base_style, ncol=ncol)
        # Like ShapeBlock, an unset color goes neutral next to a color block.
        self._accepts_neutral = color is None

    def _resolved_style(self, entry_style: dict[str, Any]) -> dict[str, Any]:
        # The neutral-shape pass writes 'facecolor'; glyphs read 'color'.
        style = super()._resolved_style(entry_style)
        if "color" not in style and "facecolor" in style:
            style["color"] = style["facecolor"]
        return style


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

class MultiLegend:
    """A canvas-placeable legend built from one block per visual channel.

    Add channel blocks with the fluent ``add_*`` methods (or ``add_block`` for a
    pre-built :class:`LegendBlock`), then :meth:`draw`. Blocks stack vertically
    by default; pass ``orientation='horizontal'`` to lay them side by side.

    Parameters
    ----------
    ax : matplotlib Axes or WCSAxes
    loc : str or (float, float)
        A preset — any of the nine matplotlib corner names (``'lower right'``
        …) for inside the axes, or an ``'outside <side>'`` name (e.g.
        ``'outside lower right'``, ``'outside bottom'``) for the figure margin
        — or an explicit ``(x, y)`` anchor read in ``coords``.
    coords : {'axes', 'figure'}, optional
        Coordinate system for a tuple ``loc`` (default ``'axes'``).
    orientation : {'vertical', 'horizontal'}
        How the blocks stack.
    palette : str or dict, optional
        An annotation palette (name or dict); its ``text``/``text2``/``frame``
        colors drive the legend text and frame for light/dark-aware styling.
        Explicit ``text_color`` / ``frame_color`` / ``facecolor`` override it.
    frameon : bool, default True
        Draw the surrounding frame.
    stroke_color, stroke_lw :
        Optional outline for legend text and swatches (same convention as the
        other decorations); off by default.
    fontsize : float, optional
        Label font size in points (title is bold at the same size).
    reserve : bool, default False
        For an ``'outside ...'`` placement, shrink the host axes to open
        figure-margin room for the legend (approximate). No-op inside.

    Examples
    --------
    >>> import skyplothelper as sph
    >>> (sph.MultiLegend(ax, loc='lower right', title='Sources')
    ...     .add_color('type', {'quasar': 'red', 'star': 'blue'})
    ...     .add_size('flux', values=[1, 10, 100], smin=10, smax=200)
    ...     .draw())
    """

    def __init__(
        self, ax: Any, *,
        loc: str | tuple[float, float] = "lower right",
        coords: str | None = None,
        orientation: str = "vertical",
        title: str | None = None,
        palette: Any = None,
        text_color: Any = None,
        frame_color: Any = None,
        facecolor: Any = None,
        frameon: bool = True,
        framealpha: float = 1.0,
        borderpad: float = 0.6,
        block_sep: float | None = None,
        stroke_color: Any = None,
        stroke_lw: float = 0.0,
        fontsize: float | None = None,
        reserve: bool = False,
    ) -> None:
        if orientation not in ("vertical", "horizontal"):
            raise ValueError(
                f"orientation must be 'vertical' or 'horizontal' "
                f"(got {orientation!r})")
        self.ax = ax
        self.loc = loc
        self.reserve = reserve
        self.coords = coords
        self.orientation = orientation
        self.title = title
        self.palette = palette
        self._text_color = text_color
        self._frame_color = frame_color
        self._facecolor = facecolor
        self.frameon = frameon
        self.framealpha = framealpha
        self.borderpad = borderpad
        self.block_sep = block_sep
        self._stroke = _stroke_path_effects(stroke_color, stroke_lw)
        self.fontsize = fontsize
        self.blocks: list[LegendBlock] = []
        self.artist: Any = None

    # -- fluent block adders --------------------------------------------
    def add_block(self, block: LegendBlock) -> "MultiLegend":
        """Append a pre-built :class:`LegendBlock` (or wrapper)."""
        self.blocks.append(block)
        return self

    def add_color(self, title: str | None, entries: Any, *,
                  swatch: str = "patch", target: str = "face",
                  marker: str = "o", size: float | None = None,
                  lw: float | None = None, ncol: int = 1,
                  **base: Any) -> "MultiLegend":
        """Add a color-encoding block (see :class:`ColorBlock`)."""
        return self.add_block(ColorBlock(
            title, entries, swatch=swatch, target=target, marker=marker,
            size=size, lw=lw, ncol=ncol, **base))

    def add_shape(self, title: str | None, entries: Any, *,
                  color: Any = None, size: float | None = None,
                  ncol: int = 1, **base: Any) -> "MultiLegend":
        """Add a marker-shape block (see :class:`ShapeBlock`)."""
        return self.add_block(ShapeBlock(
            title, entries, color=color, size=size, ncol=ncol, **base))

    def add_line(self, title: str | None, entries: Any, *,
                 vary: str = "linestyle", color: Any = None,
                 lw: float | None = None, marker: Any = None,
                 ncol: int = 1) -> "MultiLegend":
        """Add a line-style / line-width block (see :class:`LineBlock`)."""
        return self.add_block(LineBlock(
            title, entries, vary=vary, color=color, lw=lw, marker=marker,
            ncol=ncol))

    def add_size(self, title: str | None, *, values: Any = None, num: int = 4,
                 nice: bool = True, size_map: Any = None, smin: float = 10.0,
                 smax: float = 200.0, rmin: float | None = None,
                 rmax: float | None = None, scale: Any = "linear",
                 color: Any = None, marker: str = "o", fmt: Any = None,
                 ncol: int = 1, orientation: str = "vertical") -> "MultiLegend":
        """Add a graduated marker-size block (see :class:`SizeBlock`)."""
        return self.add_block(SizeBlock(
            title, values=values, num=num, nice=nice, size_map=size_map,
            smin=smin, smax=smax, rmin=rmin, rmax=rmax, scale=scale,
            color=color, marker=marker, fmt=fmt, ncol=ncol,
            orientation=orientation))

    def add_size_from(self, result: Any, *, values: Any = None, num: int = 4,
                      nice: bool = True, title: str | None = None,
                      color: Any = None, marker: str = "o", fmt: Any = None,
                      ncol: int = 1,
                      orientation: str = "vertical") -> "MultiLegend":
        """Add a size block matching a :func:`~skyplothelper.plot_catalog`
        result's ``sizeby`` scaling (see :meth:`SizeBlock.from_catalog`)."""
        return self.add_block(SizeBlock.from_catalog(
            result, values=values, num=num, nice=nice, title=title,
            color=color, marker=marker, fmt=fmt, ncol=ncol,
            orientation=orientation))

    def add_edge(self, title: str | None, entries: Any, *, marker: str = "o",
                 facecolor: Any = "0.85", size: float | None = None,
                 lw: float = 1.6, ncol: int = 1, **base: Any) -> "MultiLegend":
        """Add a marker edge-color block (see :class:`EdgeBlock`)."""
        return self.add_block(EdgeBlock(
            title, entries, marker=marker, facecolor=facecolor, size=size,
            lw=lw, ncol=ncol, **base))

    # color defaults to None in these three wrappers rather than restating the
    # block classes' "0.4": forwarding it unconditionally would shadow the
    # class default, so changing the swatch color in one place and not the
    # other would silently do nothing for anyone using the wrapper.
    def add_fill(self, title: str | None, entries: Any, *, kind: str = "marker",
                 marker: str = "o", color: Any = None, size: float | None = None,
                 ncol: int = 1) -> "MultiLegend":
        """Add a fill-state / hatch block (see :class:`FillBlock`)."""
        return self.add_block(FillBlock(
            title, entries, kind=kind, marker=marker, size=size, ncol=ncol,
            **({} if color is None else {"color": color})))

    def add_alpha(self, title: str | None, *, values: Any = None,
                  entries: Any = None, amin: float = 0.2, amax: float = 1.0,
                  color: Any = None, marker: str = "o", fmt: Any = None,
                  ncol: int = 1) -> "MultiLegend":
        """Add a graduated-opacity block (see :class:`AlphaBlock`)."""
        return self.add_block(AlphaBlock(
            title, values=values, entries=entries, amin=amin, amax=amax,
            marker=marker, fmt=fmt, ncol=ncol,
            **({} if color is None else {"color": color})))

    def add_orientation(self, title: str | None, entries: Any, *,
                        marker: str = "^", color: Any = None,
                        size: float | None = None, ncol: int = 1) -> "MultiLegend":
        """Add a marker-rotation / angle block (see :class:`OrientBlock`)."""
        return self.add_block(OrientBlock(
            title, entries, marker=marker, size=size, ncol=ncol,
            **({} if color is None else {"color": color})))

    def add_region(self, title: str | None, entries: Any, *,
                   ncol: int = 1) -> "MultiLegend":
        """Add a translucent filled-region block (see :class:`RegionBlock`)."""
        return self.add_block(RegionBlock(title, entries, ncol=ncol))

    def add_text(self, title: str | None, notes: Any, *,
                 ncol: int = 1) -> "MultiLegend":
        """Add a free-text note block (see :class:`TextBlock`)."""
        return self.add_block(TextBlock(title, notes, ncol=ncol))

    def add_colorbar(self, title: str | None, *, cmap: Any = "viridis",
                     vmin: float = 0.0, vmax: float = 1.0, n: int = 64,
                     length: float | None = None,
                     fmt: Any = None) -> "MultiLegend":
        """Add a continuous-color gradient strip (see :class:`ColorbarBlock`)."""
        return self.add_block(ColorbarBlock(
            title, cmap=cmap, vmin=vmin, vmax=vmax, n=n, length=length, fmt=fmt))

    def add_glyph(self, title: str | None, entries: Any, *, color: Any = None,
                  lw: float = 1.6, size: float | None = None,
                  ncol: int = 1) -> "MultiLegend":
        """Add a named-sph-glyph block (see :class:`GlyphBlock`)."""
        return self.add_block(GlyphBlock(
            title, entries, color=color, lw=lw, size=size, ncol=ncol))

    def add_custom(self, title: str | None, entries: Any, *,
                   ncol: int = 1) -> "MultiLegend":
        """Add a block of custom swatches: each entry value is a matplotlib
        artist drawn as the swatch (the escape hatch for anything the built-in
        channels don't cover, including sph ``Path`` glyphs)."""
        items = list(entries.items()) if isinstance(entries, dict) else list(entries)
        built = [(str(label), {"handle": art}) for label, art in items]
        return self.add_block(LegendBlock(
            title, built, swatch_kind="custom", ncol=ncol))

    # -- rendering -------------------------------------------------------
    def draw(self) -> "MultiLegend":
        """Build the legend artist and attach it to the axes."""
        from matplotlib.font_manager import FontProperties
        from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker

        if not self.blocks:
            raise ValueError("MultiLegend has no blocks; add at least one "
                             "before draw()")

        pal = self._resolve_palette()
        # With no explicit palette, follow the rcParams a plain matplotlib
        # legend would follow rather than literal black-on-white — which
        # rendered as a white box with black text on any dark figure, and
        # additionally starved ``_apply_neutral_shapes`` below of the dark
        # background it needs to pick a legible neutral.
        _default_text = rcParams["text.color"]
        _default_face = rcParams["legend.facecolor"]
        if _default_face == "inherit":
            _default_face = self.ax.get_facecolor()
        _default_frame = rcParams["legend.edgecolor"]
        if _default_frame == "inherit":
            _default_frame = rcParams["axes.edgecolor"]

        text_color = self._text_color if self._text_color is not None else \
            (pal["text"] if pal else _default_text)
        title_color = self._text_color if self._text_color is not None else \
            (pal["text"] if pal else _default_text)
        frame_color = self._frame_color if self._frame_color is not None else \
            (pal["frame"] if pal else _default_frame)
        facecolor = self._facecolor if self._facecolor is not None else \
            (pal["ax_bg"] if pal else _default_face)

        self._apply_neutral_shapes(facecolor)

        fontsize = self.fontsize if self.fontsize is not None else _DEFAULT_FONTSIZE
        swatch_h = fontsize * _SWATCH_SCALE
        block_sep = self.block_sep if self.block_sep is not None else fontsize * 0.9

        block_boxes = [
            b._render(text_color=text_color, title_color=title_color,
                      fontsize=fontsize, title_fontsize=fontsize,
                      stroke=self._stroke, swatch_h=swatch_h,
                      entry_sep=fontsize * 0.35, title_sep=fontsize * 0.3)
            for b in self.blocks
        ]

        vertical = self.orientation == "vertical" or self.title is not None
        children = block_boxes
        if self.title is not None:
            master = TextArea(self.title, textprops=dict(
                color=title_color, size=fontsize * 1.05, weight="bold",
                **({"path_effects": self._stroke} if self._stroke else {})))
            children = [master, *block_boxes]
        packed: Any = (
            VPacker(children=children, align="left", pad=0, sep=block_sep)
            if vertical else
            HPacker(children=children, align="top", pad=0, sep=block_sep))

        loc_name, bbox_to_anchor, bbox_transform = self._resolve_placement()
        box = AnchoredOffsetbox(
            loc=loc_name, child=packed, pad=self.borderpad, borderpad=0.0,
            frameon=self.frameon, bbox_to_anchor=bbox_to_anchor,
            bbox_transform=bbox_transform,
            prop=FontProperties(size=fontsize))
        box.patch.set_facecolor(facecolor)
        box.patch.set_edgecolor(frame_color)
        box.patch.set_alpha(self.framealpha)
        # Draw above data but let it be clipped-free (legends sit on top).
        box.set_zorder(10)
        box.set_clip_on(False)
        self.ax.add_artist(box)
        self.artist = box
        if self.reserve:
            self._reserve_margin()
        return self

    def _reserve_margin(self) -> None:
        """Shrink the host axes to free figure-margin room for an off-frame
        legend, so it isn't clipped by the figure edge.

        Approximate (a fixed fraction, not measured from the drawn legend) —
        WCSAxes' fixed aspect makes an exact reservation fiddly; the goal is
        just to open margin space. No-op for an inside placement.
        """
        if not (isinstance(self.loc, str) and self.loc.lower().startswith("outside")):
            return
        frac = 0.16
        pos = self.ax.get_position()
        key = self.loc.lower()
        x0, y0, w, h = pos.x0, pos.y0, pos.width, pos.height
        if "bottom" in key:
            self.ax.set_position([x0, y0 + frac * h, w, h * (1 - frac)])
        elif "top" in key:
            self.ax.set_position([x0, y0, w, h * (1 - frac)])
        elif "right" in key:
            self.ax.set_position([x0, y0, w * (1 - frac), h])
        elif "left" in key:
            self.ax.set_position([x0 + frac * w, y0, w * (1 - frac), h])

    # -- helpers ---------------------------------------------------------
    def _resolve_palette(self) -> dict[str, str] | None:
        """Resolve ``palette`` (name or dict) to an annotation-palette dict."""
        if self.palette is None:
            return None
        if isinstance(self.palette, dict):
            return self.palette
        from .style import _ANNOTATION_ALIASES, ANNOTATION_PALETTES
        key = _ANNOTATION_ALIASES.get(self.palette, self.palette)
        try:
            return ANNOTATION_PALETTES[key]
        except KeyError:
            raise ValueError(
                f"Unknown annotation palette {self.palette!r}. Available: "
                f"{', '.join(ANNOTATION_PALETTES)}") from None

    def _apply_neutral_shapes(self, facecolor: Any) -> None:
        """Resolve neutral-gray swatches to a mode-aware tone.

        A shape block whose color the caller left unset reads as "another
        color" next to a real color block; forcing it neutral gray keeps the
        two dimensions visually distinct. Size blocks are always neutral (size
        is its own dimension). The gray tone is mode-aware — lighter on a dark
        legend background so the swatches stay legible (see _neutral_gray).
        """
        neutral = _neutral_gray(facecolor)
        has_color = any(
            isinstance(b, ColorBlock) or b.swatch_kind == "patch"
            for b in self.blocks)
        for b in self.blocks:
            # Size swatches are neutral regardless of a color sibling.
            if getattr(b, "_size_neutral", False):
                for _label, style in b.entries:
                    style["facecolor"] = neutral
            # Shape/fill swatches go neutral only to contrast a color block.
            elif has_color and getattr(b, "_accepts_neutral", False):
                b.base_style["facecolor"] = neutral

    def _resolve_placement(
        self,
    ) -> tuple[str, tuple[float, float] | tuple[float, float, float, float], Any]:
        """Resolve ``loc``/``coords`` to (loc name, bbox_to_anchor, transform)."""
        ax = self.ax
        if isinstance(self.loc, str):
            key = self.loc.lower()
            if key in _OUTSIDE_PRESETS:
                anchor, name = _OUTSIDE_PRESETS[key]
                return name, anchor, ax.transAxes
            if key in _LOC_NAMES:
                return key, (0.0, 0.0, 1.0, 1.0), ax.transAxes
            raise ValueError(
                f"Unknown loc preset {self.loc!r}. Use a corner name "
                f"({', '.join(sorted(_LOC_NAMES))}), an 'outside ...' preset "
                f"({', '.join(_OUTSIDE_PRESETS)}), or an (x, y) tuple.")
        # Explicit (x, y) anchor.
        x, y = self.loc
        transform = ax.figure.transFigure if self.coords == "figure" else ax.transAxes
        # Pin the box's lower-left corner at the point by default.
        return "lower left", (float(x), float(y)), transform
