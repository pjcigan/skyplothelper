"""Tests for skyplothelper.legend — the multi-dimensional MultiLegend tool."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Rectangle

import skyplothelper as sph
from skyplothelper.legend import (
    _NEUTRAL_SWATCH,
    AlphaBlock,
    ColorbarBlock,
    ColorBlock,
    EdgeBlock,
    FillBlock,
    LegendBlock,
    LineBlock,
    OrientBlock,
    RegionBlock,
    ShapeBlock,
    SizeBlock,
    TextBlock,
    _is_hatch,
    _make_swatch,
    _neutral_gray,
    _nice_values,
    _normalize_entries,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Entry parsing
# ---------------------------------------------------------------------------

def test_normalize_entries_dict_with_vary():
    out = _normalize_entries({"a": "o", "b": "s"}, vary="marker")
    assert out == [("a", {"marker": "o"}), ("b", {"marker": "s"})]


def test_normalize_entries_pairs_preserve_order():
    out = _normalize_entries([("z", "o"), ("a", "s")], vary="marker")
    assert [label for label, _ in out] == ["z", "a"]


def test_normalize_entries_dict_values_pass_through():
    out = _normalize_entries({"x": dict(marker="D", color="C1")}, vary=None)
    assert out == [("x", {"marker": "D", "color": "C1"})]


def test_normalize_entries_scalar_without_vary_errors():
    with pytest.raises(TypeError, match="style dict"):
        _normalize_entries({"x": "o"}, vary=None)


# ---------------------------------------------------------------------------
# LegendBlock validation
# ---------------------------------------------------------------------------

def test_legendblock_rejects_unknown_swatch_kind():
    with pytest.raises(ValueError, match="swatch_kind"):
        LegendBlock("t", {"a": {}}, swatch_kind="bogus")


def test_legendblock_rejects_bad_orientation():
    with pytest.raises(ValueError, match="orientation"):
        LegendBlock("t", {"a": {}}, orientation="sideways")


# ---------------------------------------------------------------------------
# Swatch rendering
# ---------------------------------------------------------------------------

def test_make_marker_swatch_uses_facecolor():
    da = _make_swatch("marker", {"marker": "D", "facecolor": "red"}, 13.0, None)
    assert isinstance(da, DrawingArea)
    line = da.get_children()[0]
    assert isinstance(line, Line2D)
    assert line.get_marker() == "D"
    assert line.get_markerfacecolor() == "red"


def test_make_patch_swatch_is_rectangle():
    da = _make_swatch("patch", {"facecolor": "C2", "hatch": "///"}, 13.0, None)
    rect = da.get_children()[0]
    assert isinstance(rect, Rectangle)
    assert rect.get_hatch() == "///"


def test_make_line_swatch_carries_linestyle():
    da = _make_swatch("line", {"linestyle": "-.", "color": "C3"}, 13.0, None)
    line = da.get_children()[0]
    assert isinstance(line, Line2D)
    assert line.get_linestyle() == "-."


def test_swatch_applies_stroke_path_effects():
    from skyplothelper._stroke import _stroke_path_effects
    stroke = _stroke_path_effects("white", 2.0)
    da = _make_swatch("marker", {"marker": "o"}, 13.0, stroke)
    assert da.get_children()[0].get_path_effects()


# ---------------------------------------------------------------------------
# Block wrappers
# ---------------------------------------------------------------------------

def test_colorblock_default_is_patch():
    b = ColorBlock("C", {"a": "red"})
    assert b.swatch_kind == "patch"
    assert b.entries[0] == ("a", {"facecolor": "red"})


def test_colorblock_marker_swatch():
    b = ColorBlock("C", {"a": "red"}, swatch="marker", marker="s")
    assert b.swatch_kind == "marker"
    assert b.base_style["marker"] == "s"


def test_colorblock_edge_target():
    b = ColorBlock("C", {"a": "red"}, swatch="marker", target="edge")
    assert b.entries[0] == ("a", {"edgecolor": "red"})
    assert b.base_style["facecolor"] == "none"


def test_colorblock_rejects_bad_swatch_and_target():
    with pytest.raises(ValueError, match="swatch"):
        ColorBlock("C", {"a": "red"}, swatch="bogus")
    with pytest.raises(ValueError, match="target"):
        ColorBlock("C", {"a": "red"}, target="bogus")


def test_shapeblock_marks_neutral_when_color_unset():
    b = ShapeBlock("S", {"a": "o"})
    assert b._accepts_neutral is True
    assert b.entries[0] == ("a", {"marker": "o"})


def test_shapeblock_explicit_color_not_neutral():
    b = ShapeBlock("S", {"a": "o"}, color="C1")
    assert b._accepts_neutral is False
    assert b.base_style["facecolor"] == "C1"


def test_lineblock_vary_linewidth():
    b = LineBlock("L", {"thin": 1, "thick": 4}, vary="lw")
    assert b.entries[0] == ("thin", {"linewidth": 1})


def test_lineblock_default_vary_linestyle():
    b = LineBlock("L", {"a": ":"})
    assert b.entries[0] == ("a", {"linestyle": ":"})


# ---------------------------------------------------------------------------
# MultiLegend container
# ---------------------------------------------------------------------------

def test_multilegend_draw_attaches_artist():
    fig, ax = plt.subplots()
    leg = sph.MultiLegend(ax, loc="lower right").add_color(
        "C", {"a": "red", "b": "C0"}).draw()
    assert isinstance(leg.artist, AnchoredOffsetbox)
    assert leg.artist in ax.get_children()
    fig.canvas.draw()


def test_multilegend_empty_draw_errors():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="no blocks"):
        sph.MultiLegend(ax).draw()


def test_multilegend_fluent_returns_self():
    fig, ax = plt.subplots()
    m = sph.MultiLegend(ax)
    assert m.add_color("C", {"a": "red"}) is m
    assert m.add_shape("S", {"a": "o"}) is m
    assert m.add_line("L", {"a": ":"}) is m


def test_grayscale_shape_auto_with_color_block():
    # Shape swatch with unset color turns neutral gray when a color block
    # shares the legend (so shape reads as its own dimension).
    fig, ax = plt.subplots()
    leg = (sph.MultiLegend(ax)
           .add_color("C", {"a": "red"})
           .add_shape("S", {"x": "o"})
           .draw())
    shape_block = leg.blocks[1]
    assert shape_block.base_style["facecolor"] == _NEUTRAL_SWATCH


def test_shape_alone_keeps_default_color():
    # No color block -> shape swatch is not forced neutral.
    fig, ax = plt.subplots()
    leg = sph.MultiLegend(ax).add_shape("S", {"x": "o"}).draw()
    assert leg.blocks[0].base_style["facecolor"] == "C0"


def test_explicit_shape_color_survives_color_sibling():
    fig, ax = plt.subplots()
    leg = (sph.MultiLegend(ax)
           .add_color("C", {"a": "red"})
           .add_shape("S", {"x": "o"}, color="black")
           .draw())
    assert leg.blocks[1].base_style["facecolor"] == "black"


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("loc", [
    "upper left", "lower right", "center",
    "outside lower right", "outside bottom", "outside right",
])
def test_placement_presets_resolve(loc):
    fig, ax = plt.subplots()
    m = sph.MultiLegend(ax, loc=loc).add_color("C", {"a": "red"})
    name, anchor, transform = m._resolve_placement()
    assert isinstance(name, str)
    assert len(anchor) in (2, 4)
    m.draw()


def test_outside_preset_anchors_beyond_axes():
    fig, ax = plt.subplots()
    m = sph.MultiLegend(ax, loc="outside right").add_color("C", {"a": "red"})
    _, anchor, transform = m._resolve_placement()
    assert anchor[0] > 1.0                 # to the right of the axes
    assert transform is ax.transAxes


def test_free_coords_figure_transform():
    fig, ax = plt.subplots()
    m = sph.MultiLegend(ax, loc=(0.1, 0.2), coords="figure").add_color(
        "C", {"a": "red"})
    name, anchor, transform = m._resolve_placement()
    assert anchor == (0.1, 0.2)
    assert transform is ax.figure.transFigure


def test_unknown_loc_preset_errors():
    fig, ax = plt.subplots()
    m = sph.MultiLegend(ax, loc="nowhere").add_color("C", {"a": "red"})
    with pytest.raises(ValueError, match="Unknown loc preset"):
        m._resolve_placement()


# ---------------------------------------------------------------------------
# Mode-aware styling
# ---------------------------------------------------------------------------

def test_palette_drives_text_and_frame_colors():
    from matplotlib.colors import to_rgba

    from skyplothelper.style import ANNOTATION_PALETTES
    fig, ax = plt.subplots()
    leg = sph.MultiLegend(ax, palette="dark").add_color("C", {"a": "red"}).draw()
    pal = ANNOTATION_PALETTES["dark"]
    # Frame edge + face come from the palette's frame / ax_bg roles.
    assert leg.artist.patch.get_edgecolor() == to_rgba(pal["frame"])
    assert leg.artist.patch.get_facecolor() == to_rgba(pal["ax_bg"])


def test_unknown_palette_errors():
    fig, ax = plt.subplots()
    m = sph.MultiLegend(ax, palette="not-a-palette").add_color("C", {"a": "red"})
    with pytest.raises(ValueError, match="Unknown annotation palette"):
        m.draw()


def test_stroke_reaches_swatches_and_text():
    fig, ax = plt.subplots()
    leg = sph.MultiLegend(ax, stroke_color="white", stroke_lw=2.0).add_shape(
        "S", {"a": "o"}).draw()
    assert leg._stroke is not None


# ---------------------------------------------------------------------------
# SizeBlock — graduated marker sizes
# ---------------------------------------------------------------------------

def test_sizeblock_explicit_values_increasing_sizes():
    b = SizeBlock("N", values=[1, 10, 100], smin=10, smax=200,
                  rmin=1.0, rmax=100.0, scale="linear")
    sizes = [st["markersize"] for _lbl, st in b.entries]
    assert sizes == sorted(sizes)                 # monotonic in value
    # markersize is a diameter = sqrt(area); area spans [smin, smax]
    assert abs(sizes[0] - (10 ** 0.5)) < 1e-6     # value 1 -> area smin=10
    assert abs(sizes[-1] - (200 ** 0.5)) < 1e-6   # value 100 -> area smax=200


def test_sizeblock_labels_use_fmt():
    b = SizeBlock("N", values=[1234.5], smin=10, smax=20, rmin=0, rmax=2000,
                  fmt=".0f")
    assert b.entries[0][0] == "1234"


def test_sizeblock_requires_values_or_size_map():
    with pytest.raises(ValueError, match="values= or a size_map"):
        SizeBlock("N")


def test_sizeblock_sqrt_scale_matches_formula():
    # value v -> area = smin + (smax-smin)*(sqrt(v)-rmin)/(rmax-rmin)
    from skyplothelper.data_plots import _apply_size_scale
    smin, smax = 8.0, 400.0
    rmin, rmax = 1.0, np.sqrt(600.0)              # scaled bounds of vlim (1,600)
    b = SizeBlock("N", values=[100], smin=smin, smax=smax,
                  rmin=rmin, rmax=rmax, scale="sqrt")
    sc100 = _apply_size_scale(np.array([100.0]), "sqrt")[0]
    area = smin + (smax - smin) * (sc100 - rmin) / (rmax - rmin)
    assert abs(b.entries[0][1]["markersize"] - np.sqrt(area)) < 1e-6


def test_size_swatch_grows_box_for_large_marker():
    small = _make_swatch("marker", {"marker": "o", "markersize": 8}, 13.0, None)
    big = _make_swatch("marker", {"marker": "o", "markersize": 40}, 13.0, None)
    assert big.width > small.width                # box grew to fit


# ---------------------------------------------------------------------------
# plot_catalog enrichment + SizeBlock.from_catalog
# ---------------------------------------------------------------------------

def _allsky_ax():
    from skyplothelper.wcs_frame import make_wcs_frame
    fig = plt.figure()
    return make_wcs_frame(projection="AIT", fig=fig)


def test_plot_catalog_stashes_size_info():
    ax = _allsky_ax()
    cat = {"ra": [10.0, 20, 30], "dec": [0.0, 5, -5], "n": [1.0, 50, 500]}
    cp = sph.plot_catalog(ax, cat, sizeby="n", frame="icrs")
    info = cp._sph_size_info
    assert info.size_scale == "linear"
    assert info.size_map is not None


def test_from_catalog_reproduces_onplot_sizes():
    ax = _allsky_ax()
    rng = np.random.default_rng(0)
    n = 80
    cat = {"ra": rng.uniform(0, 360, n),
           "dec": np.degrees(np.arcsin(rng.uniform(-1, 1, n))),
           "n": rng.integers(1, 600, n).astype(float)}
    cp = sph.plot_catalog(ax, cat, sizeby="n", size_scale="sqrt",
                          smin=8, smax=400, frame="icrs")
    # A value present in the data must get the exact plotted marker area.
    v = float(cat["n"][10])
    block = SizeBlock.from_catalog(cp, values=[v])
    onplot_area = cp.get_sizes()[10]
    legend_ms = block.entries[0][1]["markersize"]
    assert abs(legend_ms - np.sqrt(onplot_area)) < 1e-6


def test_from_catalog_without_sizeby_errors():
    ax = _allsky_ax()
    cp = sph.plot_catalog(ax, {"ra": [10.0], "dec": [0.0]}, frame="icrs")
    with pytest.raises(ValueError, match="no size scaling"):
        SizeBlock.from_catalog(cp)


def test_size_vlim_shares_scale_across_calls():
    ax = _allsky_ax()
    kw = dict(sizeby="n", size_vlim=(1, 600), smin=8, smax=400, frame="icrs")
    a = sph.plot_catalog(ax, {"ra": [10.0], "dec": [0.0], "n": [100.0]}, **kw)
    b = sph.plot_catalog(ax, {"ra": [20.0], "dec": [0.0], "n": [100.0]},
                         marker="D", **kw)
    assert np.isclose(a.get_sizes()[0], b.get_sizes()[0])


def test_add_size_from_bvid_layout_renders():
    ax = _allsky_ax()
    rng = np.random.default_rng(1)
    n = 120
    cat = {"ra": rng.uniform(0, 360, n),
           "dec": np.degrees(np.arcsin(rng.uniform(-1, 1, n))),
           "n": rng.integers(1, 600, n).astype(float)}
    cp = sph.plot_catalog(ax, cat, sizeby="n", size_scale="sqrt",
                          smin=8, smax=400, frame="icrs")
    leg = (sph.MultiLegend(ax, loc="outside bottom", orientation="horizontal")
           .add_size_from(cp, values=[1, 50, 200, 500], title="Nb of Obs",
                          orientation="horizontal")
           .add_color("Cat", {"A": "orange", "B": "C0", "C": "red"},
                      swatch="marker")
           .draw())
    ax.figure.canvas.draw()
    assert len(leg.blocks) == 2


# ---------------------------------------------------------------------------
# Mode-aware neutral gray
# ---------------------------------------------------------------------------

def test_neutral_gray_mode_aware():
    assert _neutral_gray("white") == _NEUTRAL_SWATCH   # dark gray on light
    assert _neutral_gray("#101018") == "0.7"           # light gray on dark


def test_size_block_neutral_is_mode_aware_on_dark():
    fig, ax = plt.subplots()
    leg = (sph.MultiLegend(ax, palette="dark")
           .add_size("N", values=[1, 10, 100], smin=10, smax=200,
                     rmin=1, rmax=100)
           .draw())
    # On the dark palette the size swatches use the lighter neutral.
    face = leg.blocks[0].entries[0][1]["facecolor"]
    assert face == "0.7"


# ---------------------------------------------------------------------------
# Step-3 channels: edge / fill / hatch / alpha / orientation / region / text /
# custom
# ---------------------------------------------------------------------------

def test_edgeblock_varies_edgecolor():
    b = EdgeBlock("Flag", {"a": "C2", "b": "C3"})
    assert b.entries[0] == ("a", {"edgecolor": "C2"})
    assert b.base_style["facecolor"] == "0.85"


def test_fillblock_filled_open_specs():
    b = FillBlock("Ap", {"full": "filled", "reduced": "open"}, color="0.4")
    assert b.entries[0][1]["facecolor"] == "0.4"          # filled
    assert b.entries[1][1]["facecolor"] == "none"         # open
    assert b.entries[1][1]["edgecolor"] == "0.4"
    assert b.swatch_kind == "marker"


def test_fillblock_hatch_upgrades_to_patch():
    b = FillBlock("Survey", {"DES": "///", "LSST": "xxx"}, kind="marker",
                  color="C0")
    assert b.swatch_kind == "patch"                       # hatch forces patch
    assert b.entries[0][1]["hatch"] == "///"


def test_fillblock_rejects_bad_spec():
    with pytest.raises(ValueError, match="fill spec"):
        FillBlock("X", {"a": "bogus"})


def test_is_hatch_discriminates():
    assert _is_hatch("///") and _is_hatch("xxx")
    assert not _is_hatch("filled") and not _is_hatch("open")
    assert not _is_hatch("C0")


def test_alphablock_graduated_maps_to_range():
    b = AlphaBlock("Density", values=[0, 10], amin=0.2, amax=1.0)
    assert b.entries[0][1]["alpha"] == 0.2                # min value -> amin
    assert b.entries[-1][1]["alpha"] == 1.0               # max value -> amax


def test_alphablock_explicit_entries():
    b = AlphaBlock("D", entries={"faint": 0.3, "bright": 0.9})
    assert b.entries[0][1]["alpha"] == 0.3


def test_alphablock_requires_values_or_entries():
    with pytest.raises(ValueError, match="values= or entries="):
        AlphaBlock("D")


def test_orientblock_sets_angle():
    b = OrientBlock("PA", {"0": 0, "45": 45, "90": 90})
    assert b.entries[1] == ("45", {"angle": 45})
    assert b.swatch_kind == "marker"


def test_orient_swatch_applies_rotation():
    # A rotated marker swatch builds without error and carries the marker.
    da = _make_swatch("marker", {"marker": "^", "angle": 45.0}, 13.0, None)
    assert da.get_children()


def test_regionblock_color_and_dict():
    b = RegionBlock("Foot", {"DES": "C0",
                             "Euclid": dict(fc="C1", ec="C1", hatch="//")})
    assert b.swatch_kind == "region"
    assert b.entries[0][1] == {"facecolor": "C0"}
    # fc/ec aliases normalized to facecolor/edgecolor
    assert b.entries[1][1]["facecolor"] == "C1"
    assert b.entries[1][1]["edgecolor"] == "C1"


def test_region_swatch_is_translucent_rectangle():
    from matplotlib.patches import Rectangle
    da = _make_swatch("region", {"facecolor": "C0"}, 13.0, None)
    rect = da.get_children()[0]
    assert isinstance(rect, Rectangle)
    assert rect.get_alpha() == 0.35                        # default translucency


def test_textblock_is_label_only():
    b = TextBlock("Notes", ["dashed = model", "shaded = 1σ"])
    assert b.swatch_kind == "text"
    assert [lbl for lbl, _ in b.entries] == ["dashed = model", "shaded = 1σ"]


def test_text_entry_renders_without_swatch():
    fig, ax = plt.subplots()
    leg = sph.MultiLegend(ax).add_text("Notes", ["a note"]).draw()
    ax.figure.canvas.draw()
    assert leg.artist is not None


def test_custom_swatch_uses_handle():
    da = _make_swatch("custom", {"handle": Line2D([6], [6], marker="*")},
                      13.0, None)
    assert isinstance(da.get_children()[0], Line2D)


def test_custom_swatch_requires_handle():
    with pytest.raises(ValueError, match="handle"):
        _make_swatch("custom", {}, 13.0, None)


def test_all_step3_channels_render_together():
    fig, ax = plt.subplots()
    art = Line2D([6], [6], marker="*", markersize=10, linestyle="none",
                 markerfacecolor="gold")
    leg = (sph.MultiLegend(ax, loc="center")
           .add_edge("Flag", {"secure": "C2", "marginal": "C3"})
           .add_fill("Ap", {"full": "filled", "reduced": "open"})
           .add_fill("Survey", {"DES": "///"}, kind="patch", color="C0")
           .add_alpha("Density", values=[1, 5, 10])
           .add_orientation("PA", {"0": 0, "90": 90})
           .add_region("Footprint", {"DES": "C0"})
           .add_text("Notes", ["dashed = model"])
           .add_custom("Special", {"star": art})
           .draw())
    fig.canvas.draw()
    assert len(leg.blocks) == 8


# ---------------------------------------------------------------------------
# Phase 2 polish: auto-nice values / ColorbarBlock / reserve
# ---------------------------------------------------------------------------

def test_nice_values_round_decades():
    assert _nice_values(1, 600, 5) == [1, 5, 20, 100, 500]


def test_nice_values_degenerate_range():
    assert _nice_values(5, 5) == [5]                       # lo == hi
    assert _nice_values(-1, 1) == [-1, 1]                  # non-positive lo


def test_sizeblock_auto_nice_labels():
    raw = np.array([1.0, 3, 7, 50, 200, 590])
    sm = (raw, np.sqrt(raw), float(np.sqrt(raw).min()), float(np.sqrt(raw).max()))
    b = SizeBlock("N", size_map=sm, smin=8, smax=400, scale="sqrt")
    labels = [lbl for lbl, _ in b.entries]
    # Round numbers within the data range, not the raw min/mean/max.
    assert all(float(x) in (1, 2, 5, 10, 20, 50, 100, 200, 500) for x in labels)


def test_sizeblock_nice_false_uses_interp():
    raw = np.array([1.0, 3, 7, 50, 200, 590])
    sm = (raw, np.sqrt(raw), float(np.sqrt(raw).min()), float(np.sqrt(raw).max()))
    b = SizeBlock("N", size_map=sm, smin=8, smax=400, scale="sqrt",
                  nice=False, num=4)
    assert len(b.entries) == 4                             # linspace-interp path


def test_colorbarblock_renders_gradient():
    fig, ax = plt.subplots()
    leg = sph.MultiLegend(ax).add_colorbar(
        "z", cmap="plasma", vmin=0, vmax=3, fmt=".1f").draw()
    fig.canvas.draw()
    assert isinstance(leg.blocks[0], ColorbarBlock)
    assert leg.blocks[0].vmax == 3.0


def test_reserve_shrinks_axes_for_outside_legend():
    from skyplothelper.wcs_frame import make_wcs_frame
    fig = plt.figure()
    ax = make_wcs_frame(projection="AIT", fig=fig)
    y0_before = ax.get_position().y0
    sph.MultiLegend(ax, loc="outside bottom", reserve=True).add_color(
        "C", {"a": "red"}).draw()
    assert ax.get_position().y0 > y0_before                # bottom lifted


def test_reserve_noop_inside():
    fig, ax = plt.subplots()
    pos_before = tuple(ax.get_position().bounds)
    sph.MultiLegend(ax, loc="lower right", reserve=True).add_color(
        "C", {"a": "red"}).draw()
    assert tuple(ax.get_position().bounds) == pos_before   # unchanged


# ---------------------------------------------------------------------------
# Named-glyph swatches (A): register_glyph / GlyphBlock / add_glyph
# ---------------------------------------------------------------------------

def test_builtin_glyphs_registered():
    glyphs = sph.list_glyphs()
    for name in ("reticle_plus", "reticle_x", "reticle_L", "reticle_circle",
                 "crosshair", "target", "corner"):
        assert name in glyphs


def test_glyphblock_entries_and_neutral_flag():
    b = sph.GlyphBlock("Targets", {"a": "reticle_circle", "b": "crosshair"})
    assert b.swatch_kind == "glyph"
    assert b.entries[0] == ("a", {"glyph": "reticle_circle"})
    assert b._accepts_neutral is True                # color unset -> neutral


def test_glyph_swatch_renders_reticle():
    da = _make_swatch("glyph", {"glyph": "reticle_circle"}, 13.0, None)
    assert da.get_children()                          # drew line artists


def test_glyph_swatch_unknown_name_errors():
    with pytest.raises(ValueError, match="unknown glyph"):
        _make_swatch("glyph", {"glyph": "not-a-glyph"}, 13.0, None)


def test_register_custom_glyph_and_use():
    calls = {}

    def _builder(center, size, color, lw):
        calls["hit"] = (center, size, color, lw)
        return [Line2D([center[0]], [center[1]], marker="o")]

    sph.register_glyph("test_glyph_xyz", _builder)
    assert "test_glyph_xyz" in sph.list_glyphs()
    _make_swatch("glyph", {"glyph": "test_glyph_xyz", "color": "red"},
                 13.0, None)
    assert calls["hit"][2] == "red"                  # color threaded through


def test_glyph_neutral_gray_next_to_color_block():
    fig, ax = plt.subplots()
    leg = (sph.MultiLegend(ax)
           .add_color("C", {"a": "red"})
           .add_glyph("G", {"t": "reticle_circle"})
           .draw())
    # The neutral pass writes 'facecolor'; GlyphBlock maps it to 'color'.
    resolved = leg.blocks[1]._resolved_style(leg.blocks[1].entries[0][1])
    assert resolved["color"] == _NEUTRAL_SWATCH


# ---------------------------------------------------------------------------
# Integration — the DGR-style multi-channel figure
# ---------------------------------------------------------------------------

def test_dgr_style_multichannel_legend_renders():
    fig, ax = plt.subplots()
    leg = (sph.MultiLegend(ax, loc="lower right")
           .add_color("Target", {"DDO 69": "purple", "DDO 70": "C0",
                                  "DDO 75": "green", "DDO 210": "#b0a0d0"},
                      ncol=2)
           .add_shape("Sample", {"DGS": "o", "KINGFISH": "D",
                                 "Galametz+11": "+", "Galliano+08": "^"})
           .add_line("RR14 Fit", {r"$X_{CO,MWG}$": ":", r"$X_{CO,Z}$": "-."})
           .draw())
    fig.canvas.draw()                      # exercises the full layout
    assert len(leg.blocks) == 3
    assert isinstance(leg.artist, AnchoredOffsetbox)


def test_colorbarblock_border_is_stroked_when_stroke_set():
    """ColorbarBlock strokes its bar-border rectangle too (a swatch-equivalent),
    matching the outlined swatches other blocks draw."""
    import matplotlib.patheffects as pe
    from matplotlib.patches import Rectangle

    from skyplothelper.legend import ColorbarBlock
    blk = ColorbarBlock("cb", cmap="viridis", vmin=0.0, vmax=1.0)
    art = blk._render(text_color="k", title_color="k", fontsize=9,
                      title_fontsize=9,
                      stroke=[pe.withStroke(linewidth=2, foreground="w")],
                      swatch_h=12.0, entry_sep=2.0, title_sep=2.0)

    def _rects(o, acc):
        for c in getattr(o, "get_children", list)():
            if isinstance(c, Rectangle):
                acc.append(c)
            _rects(c, acc)
        return acc

    stroked = [r for r in _rects(art, []) if r.get_path_effects()]
    assert stroked, "ColorbarBlock border rectangle not stroked"
