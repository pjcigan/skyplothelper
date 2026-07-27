"""Smoke tests for skyplothelper.overlays.annotations."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from skyplothelper.overlays.annotations import (
    add_axis_inlay,
    add_bandlabels,
    add_compass,
    add_sizebar,
    style_ax_colors,
)
from skyplothelper.wcs_frame import make_wcs_frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _wcs_axes_with_beam():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(180.0, 0.0), fig=fig)
    return fig, ax


def test_add_compass_smoke():
    fig, ax = _wcs_axes_with_beam()
    add_compass(ax)
    fig.canvas.draw()


def _compass_head_span_pts(ax, art):
    """Rendered arrow-patch span (px → points) — proxy for head/arrow size."""
    ax.figure.canvas.draw()
    bb = art.arrow_patch.get_window_extent()
    return max(bb.width, bb.height) * 72.0 / ax.figure.dpi


def test_add_compass_head_scales_with_figure_size():
    """The arrowhead must scale with the figure (was a fixed mutation_scale=15,
    so it dominated a small/thumbnail axes)."""
    fig_s = plt.figure(figsize=(2, 2))
    ax_s = make_wcs_frame(111, projection="TAN", center=(180, 0), fig=fig_s)
    fig_s.canvas.draw()
    small = _compass_head_span_pts(ax_s, add_compass(ax_s)[0])

    fig_l = plt.figure(figsize=(8, 8))
    ax_l = make_wcs_frame(111, projection="TAN", center=(180, 0), fig=fig_l)
    fig_l.canvas.draw()
    large = _compass_head_span_pts(ax_l, add_compass(ax_l)[0])

    assert large > small * 1.5, (small, large)   # clearly scales up


def test_add_compass_head_length_is_honored():
    """head_length / head_width are live (were accepted but ignored): a tiny
    value yields a visibly smaller arrowhead than a large one."""
    fig, ax = _wcs_axes_with_beam()
    fig.canvas.draw()
    big = add_compass(ax, loc="lower left",
                      head_length=0.04, head_width=0.04)[0]
    small = add_compass(ax, loc="upper right",
                        head_length=0.002, head_width=0.002)[0]
    fig.canvas.draw()
    bw = big.arrow_patch.get_window_extent().width
    sw = small.arrow_patch.get_window_extent().width
    assert bw > sw, (sw, bw)


def test_add_compass_forwards_kwargs_to_arrowprops():
    """**kwargs reach arrowprops (were accepted but ignored) — e.g. an
    explicit mutation_scale override doesn't raise."""
    fig, ax = _wcs_axes_with_beam()
    fig.canvas.draw()
    arts = add_compass(ax, mutation_scale=2.0, linestyle="--")
    fig.canvas.draw()
    assert arts


def _wcs_image_axes():
    fig = plt.figure(figsize=(6, 5))
    ax = make_wcs_frame(111, projection="TAN", center=(180.0, 0.0),
                        cdelt=0.01, npix=(200, 200), fig=fig)
    import numpy as np
    im = ax.imshow(np.random.default_rng(0).random((200, 200)), origin="lower")
    fig.canvas.draw()
    return fig, ax, im


@pytest.mark.parametrize("mode", ["divider", "inset", "simple"])
def test_add_colorbar_modes_match_image_height(mode):
    """Default shrink=1.0 → the colorbar matches the image height (was 0.8,
    so it rendered noticeably short). All three placement modes match."""
    from skyplothelper.overlays.annotations import add_colorbar
    fig, ax, im = _wcs_image_axes()
    cbar = add_colorbar(im, ax=ax, mode=mode)
    fig.canvas.draw()
    ratio = (cbar.ax.get_window_extent().height
             / ax.get_window_extent().height)
    assert ratio > 0.9, (mode, ratio)


@pytest.mark.parametrize("mode", ["divider", "inset", "simple"])
@pytest.mark.parametrize("location", ["right", "left", "top", "bottom"])
def test_add_colorbar_location(location, mode):
    """location= places the bar on any side (all modes) and moves the ticks."""
    from skyplothelper.overlays.annotations import add_colorbar
    fig, ax, im = _wcs_image_axes()
    cbar = add_colorbar(im, ax=ax, mode=mode, location=location, label="x")
    if location == "left":
        assert cbar.ax.yaxis.get_ticks_position() == "left"
    elif location == "top":
        assert cbar.ax.xaxis.get_ticks_position() == "top"
    # orientation follows the side
    want = "horizontal" if location in ("top", "bottom") else "vertical"
    assert cbar.orientation == want
    plt.close(fig)


def test_add_colorbar_invalid_location_raises():
    from skyplothelper.overlays.annotations import add_colorbar
    fig, ax, im = _wcs_image_axes()
    with pytest.raises(ValueError, match="location must be"):
        add_colorbar(im, ax=ax, location="middle")
    plt.close(fig)


def test_add_colorbar_divider_falls_back_to_inset_on_imagegrid():
    """On an ImageGrid panel (axes with a locator) the default divider mode
    would break the layout, so add_colorbar warns and falls back to inset."""
    import numpy as np
    from mpl_toolkits.axes_grid1 import ImageGrid

    from skyplothelper.overlays.annotations import add_colorbar
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.1)
    im = grid[0].imshow(np.random.default_rng(0).random((10, 10)))
    with pytest.warns(UserWarning, match="already has a locator"):
        cbar = add_colorbar(im, ax=grid[0])          # default mode='divider'
    # the panel image survives (locator not clobbered)
    assert grid[0].images and cbar is not None
    plt.close(fig)


def test_add_colorbar_divider_handles_tall_axes():
    """The divider mode must NOT overshoot on a tall fixed-aspect axes (the
    plt.colorbar gotcha that can size the bar ~2× the image there)."""
    from skyplothelper.overlays.annotations import add_colorbar
    fig = plt.figure(figsize=(4, 7))
    ax = make_wcs_frame(111, projection="TAN", center=(180, 0),
                        cdelt=0.01, npix=(200, 200), fig=fig)
    import numpy as np
    im = ax.imshow(np.random.default_rng(0).random((360, 120)), origin="lower")
    fig.canvas.draw()
    cbar = add_colorbar(im, ax=ax)   # default divider
    fig.canvas.draw()
    ratio = (cbar.ax.get_window_extent().height
             / ax.get_window_extent().height)
    assert 0.9 < ratio < 1.1, ratio


def test_add_colorbar_divider_falls_back_to_simple_on_polar():
    """On a polar axes (cone/bowtie frame) divider mode would collapse the
    wedge, so add_colorbar warns and falls back to 'simple'."""
    import warnings

    from skyplothelper import make_cone_frame
    from skyplothelper.overlays.annotations import add_colorbar
    plt.figure()
    cf = make_cone_frame(111)
    sc = cf.scatter([0.3, 0.6, 1.0], [1, 2, 3], c=[1, 2, 3], cmap="magma")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        add_colorbar(sc, ax=cf)          # default mode='divider'
    assert any("divider" in str(x.message) for x in w)
    assert cf.get_position().width > 0.2  # wedge not collapsed to a blob


def test_add_colorbar_shrink_is_honored():
    from skyplothelper.overlays.annotations import add_colorbar
    fig, ax, im = _wcs_image_axes()
    cbar = add_colorbar(im, ax=ax, mode="divider", shrink=0.5)
    fig.canvas.draw()
    ratio = (cbar.ax.get_window_extent().height
             / ax.get_window_extent().height)
    assert 0.4 < ratio < 0.6, ratio


@pytest.mark.parametrize("mode", ["divider", "inset", "simple"])
def test_add_colorbar_horizontal_under_allsky_frame(mode):
    """Horizontal colorbar beneath an AITOFF all-sky frame (the healpix case):
    spans the frame width and sits below it, for every mode."""
    import numpy as np

    from skyplothelper.overlays.annotations import add_colorbar
    fig = plt.figure(figsize=(8, 5))
    ax = make_wcs_frame(111, projection="AIT", center=0, fig=fig)
    fig.canvas.draw()
    im = ax.imshow(np.random.default_rng(0).random((90, 180)),
                   origin="lower", transform=ax.get_transform("world"))
    cbar = add_colorbar(im, ax=ax, mode=mode, orientation="horizontal")
    fig.canvas.draw()
    ce = cbar.ax.get_window_extent()
    ae = ax.get_window_extent()
    assert ce.width / ae.width > 0.9               # spans the frame width
    assert ce.y1 <= ae.y0 + 1                       # sits below the frame


@pytest.mark.parametrize("mode", ["divider", "inset", "simple"])
def test_add_colorbar_ticks_draw_above_solids(mode):
    """The tick-bearing axes must draw ABOVE the color solids. The sph base
    styles set inward-pointing ticks (``*tick.direction='in'``), which land
    inside the bar — the colorbar Axis draws as a unit at a low zorder while
    the solids QuadMesh defaults higher, so without the lift the marks are
    painted over and vanish."""
    from skyplothelper.overlays.annotations import add_colorbar
    fig, ax, im = _wcs_image_axes()
    cbar = add_colorbar(im, ax=ax, mode=mode)
    z = cbar.solids.get_zorder()
    assert cbar.ax.yaxis.get_zorder() > z, (cbar.ax.yaxis.get_zorder(), z)
    assert cbar.ax.xaxis.get_zorder() > z, (cbar.ax.xaxis.get_zorder(), z)


def test_add_colorbar_cax_passthrough_uses_given_axes():
    """cax= draws the bar into the caller's axes and bypasses mode placement."""
    from skyplothelper.overlays.annotations import add_colorbar
    fig, ax, im = _wcs_image_axes()
    my_cax = ax.inset_axes([1.02, 0.0, 0.04, 1.0])
    cbar = add_colorbar(im, cax=my_cax, label="v")
    assert cbar.ax is my_cax
    # polish still applies through the cax path
    assert cbar.ax.yaxis.get_zorder() > cbar.solids.get_zorder()


def test_add_colorbar_cax_still_strokes():
    """The stroke option works on the explicit-cax path too."""
    from skyplothelper.overlays.annotations import add_colorbar
    fig, ax, im = _wcs_image_axes()
    my_cax = ax.inset_axes([1.02, 0.0, 0.04, 1.0])
    cbar = add_colorbar(im, cax=my_cax, stroke_color="white")
    assert all(t.get_path_effects()
               for t in cbar.ax.yaxis.get_ticklines(minor=False))
    assert cbar.outline.get_path_effects()


def test_add_colorbar_cax_ignores_bad_mode():
    """With cax= given, mode is bypassed entirely — even an invalid mode is
    accepted (and ignored) rather than raising."""
    from skyplothelper.overlays.annotations import add_colorbar
    fig, ax, im = _wcs_image_axes()
    my_cax = ax.inset_axes([1.02, 0.0, 0.04, 1.0])
    cbar = add_colorbar(im, cax=my_cax, mode="fancy")  # not validated
    assert cbar.ax is my_cax


def test_add_colorbar_two_bars_via_cax():
    """Two independently-cmapped mappables → two non-overlapping bars via cax."""
    import numpy as np

    from skyplothelper.overlays.annotations import add_colorbar
    fig, ax = plt.subplots()
    rng = np.random.default_rng(0)
    sc1 = ax.scatter(rng.random(20), rng.random(20), c=rng.random(20),
                     cmap="viridis")
    sc2 = ax.scatter(rng.random(20), rng.random(20), c=rng.random(20),
                     cmap="plasma")
    cax1 = ax.inset_axes([1.02, 0.55, 0.04, 0.45])
    cax2 = ax.inset_axes([1.02, 0.00, 0.04, 0.45])
    cb1 = add_colorbar(sc1, cax=cax1, label="set 1")
    cb2 = add_colorbar(sc2, cax=cax2, label="set 2")
    fig.canvas.draw()
    a, b = cb1.ax.get_window_extent(), cb2.ax.get_window_extent()
    ox = max(0, min(a.x1, b.x1) - max(a.x0, b.x0))
    oy = max(0, min(a.y1, b.y1) - max(a.y0, b.y0))
    assert ox * oy == 0, "the two cax-placed bars must not overlap"


def test_add_colorbar_rejects_bad_mode():
    from skyplothelper.overlays.annotations import add_colorbar
    fig, ax, im = _wcs_image_axes()
    with pytest.raises(ValueError, match="mode must be"):
        add_colorbar(im, ax=ax, mode="fancy")


def test_add_colorbar_stroke_applies_to_ticks_and_spine():
    """stroke_color draws a stroke behind the tick marks AND the frame
    (so ticks stay legible where they'd blend into the colormap)."""
    from skyplothelper.overlays.annotations import add_colorbar
    fig, ax, im = _wcs_image_axes()
    cbar = add_colorbar(im, ax=ax, stroke_color="white", stroke_lw=2.5)
    ticklines = (list(cbar.ax.yaxis.get_ticklines(minor=False))
                 + list(cbar.ax.yaxis.get_ticklines(minor=True)))
    assert ticklines and all(t.get_path_effects() for t in ticklines)
    assert cbar.outline.get_path_effects()


def test_add_colorbar_stroke_off_by_default():
    """No stroke unless stroke_color is given."""
    from skyplothelper.overlays.annotations import add_colorbar
    fig, ax, im = _wcs_image_axes()
    cbar = add_colorbar(im, ax=ax)
    assert not any(t.get_path_effects()
                   for t in cbar.ax.yaxis.get_ticklines(minor=False))
    assert not cbar.outline.get_path_effects()


def test_add_colorbar_stroke_targets_ticks_only():
    """stroke_targets='ticks' strokes the marks but leaves the frame plain."""
    from skyplothelper.overlays.annotations import add_colorbar
    fig, ax, im = _wcs_image_axes()
    cbar = add_colorbar(im, ax=ax, stroke_color="white",
                        stroke_targets="ticks")
    assert all(t.get_path_effects()
               for t in cbar.ax.yaxis.get_ticklines(minor=False))
    assert not cbar.outline.get_path_effects()


def test_add_colorbar_stroke_targets_spine_only():
    """stroke_targets='spine' strokes the frame but leaves the marks plain."""
    from skyplothelper.overlays.annotations import add_colorbar
    fig, ax, im = _wcs_image_axes()
    cbar = add_colorbar(im, ax=ax, stroke_color="white",
                        stroke_targets="spine")
    assert cbar.outline.get_path_effects()
    assert not any(t.get_path_effects()
                   for t in cbar.ax.yaxis.get_ticklines(minor=False))


def test_add_colorbar_rejects_bad_stroke_targets():
    from skyplothelper.overlays.annotations import add_colorbar
    fig, ax, im = _wcs_image_axes()
    with pytest.raises(ValueError, match="stroke_targets must be"):
        add_colorbar(im, ax=ax, stroke_color="white", stroke_targets="all")


def test_add_bandlabels_no_vertical_drift():
    """Band labels must sit at one consistent height — the old code anchored
    each to the previous label's bbox at a 0.96 y-fraction PLUS a stray
    (0.04, 0.96)-point offset, so each crept ~0.5 pt higher than the last."""
    fig, ax = plt.subplots()
    sph_labels = ["a", "b", "c", "d", "e", "f"]
    add_bandlabels(ax, sph_labels, ["r", "g", "b", "c", "m", "y"])
    fig.canvas.draw()
    y0s = [t.get_window_extent().y0 for t in ax.texts][:len(sph_labels)]
    assert max(y0s) - min(y0s) < 1.0, y0s   # aligned, no cumulative drift


def test_add_axis_inlay_smoke():
    fig, ax = _wcs_axes_with_beam()
    fig.canvas.draw()  # initialize WCS pixel-to-data transform first
    add_axis_inlay(ax, lon_label="RA", lat_label="Dec")
    fig.canvas.draw()


def test_add_sizebar_smoke():
    fig, ax = _wcs_axes_with_beam()
    add_sizebar(ax, length_pixels=20, label="20 px")
    fig.canvas.draw()


def test_add_sizebar_stroke_disabled():
    """Setting stroke_color=None drops the prior hardcoded black stroke."""
    fig, ax = _wcs_axes_with_beam()
    asb = add_sizebar(ax, length_pixels=20, label="20 px", stroke_color=None)
    # The label and bar children should now have no path effects.
    for a in [asb.size_bar._children[0], asb.txt_label._text]:
        assert not a.get_path_effects()


def test_add_sizebar_stroke_default_preserves_black_stroke():
    """Default behavior (no kwargs) still applies the prior black stroke."""
    fig, ax = _wcs_axes_with_beam()
    asb = add_sizebar(ax, length_pixels=20, label="20 px")
    for a in [asb.size_bar._children[0], asb.txt_label._text]:
        assert len(a.get_path_effects()) == 1  # just withStroke (core drawn by the effect)


def test_style_ax_colors_runs():
    """Quick test on a plain matplotlib axes."""
    fig, mpl_ax = plt.subplots()
    style_ax_colors(mpl_ax, color="white")
    fig.canvas.draw()


def test_add_bandlabels_smoke():
    fig, ax = _wcs_axes_with_beam()
    add_bandlabels(ax, labels=["A", "B"], labcolors=["red", "blue"])
    fig.canvas.draw()


import skyplothelper as sph  # noqa: E402


def test_add_axis_inlay_kwargs_reach_the_arrows():
    """``**kwargs`` was declared and silently swallowed.

    ``add_compass`` forwards kwargs into ``arrowprops``; the inlay accepted
    them and used them nowhere, so callers got no error and no effect.
    """
    fig, ax = sph.allsky_figure(projection="AIT")
    artists = sph.add_axis_inlay(ax, lw=1.0, mutation_scale=30)
    arrows = [a for a in artists if getattr(a, "arrow_patch", None) is not None]
    assert arrows, "inlay drew no arrows"
    assert all(a.arrow_patch.get_mutation_scale() == 30 for a in arrows)
    plt.close(fig)


def test_add_axis_inlay_default_mutation_scale_unchanged():
    fig, ax = sph.allsky_figure(projection="AIT")
    artists = sph.add_axis_inlay(ax)
    arrows = [a for a in artists if getattr(a, "arrow_patch", None) is not None]
    assert all(a.arrow_patch.get_mutation_scale() == 15 for a in arrows)
    plt.close(fig)


# ---- theme-aware defaults + the knobs to override them (audit tracks B+C) ---

def _pair_for(theme, fn, **kw):
    import matplotlib
    from matplotlib import rcParams
    matplotlib.rcdefaults()
    sph.set_style(base="standard", **({"theme": theme} if theme else {}))
    fig, ax = sph.allsky_figure(projection="AIT")
    artists = fn(ax, **kw)
    out = (rcParams["text.color"], rcParams["axes.facecolor"], artists)
    return fig, out


@pytest.mark.parametrize("fn", [sph.add_compass, sph.add_axis_inlay])
def test_ink_and_stroke_flip_together_on_a_dark_theme(fn):
    """``color='k'`` + ``stroke_color='w'`` is a contrast PAIR. Flipping one
    member alone makes the artist less legible than the hard-coded original,
    so both must resolve together."""
    from matplotlib.colors import to_rgb

    def lum(c):
        r, g, b = to_rgb(c)
        return 0.299 * r + 0.587 * g + 0.114 * b

    for theme in (None, "dark_sky"):
        fig, (ink, stroke, artists) = _pair_for(theme, fn)
        assert artists
        assert abs(lum(ink) - lum(stroke)) > 0.3, (
            f"{fn.__name__} lost contrast on theme={theme}")
        plt.close(fig)
    import matplotlib
    matplotlib.rcdefaults()


def test_compass_explicit_colors_still_win():
    fig, ax = sph.allsky_figure(projection="AIT")
    artists = sph.add_compass(ax, color="#ff0000", stroke_color="#00ff00")
    arrows = [a for a in artists if getattr(a, "arrow_patch", None) is not None]
    assert arrows
    for a in arrows:
        assert tuple(round(v, 2) for v in a.arrow_patch.get_edgecolor()[:3]) \
            == (1.0, 0.0, 0.0)
    plt.close(fig)


def test_axis_inlay_background_box_is_stylable():
    """`bg_edgecolor` / `bg_lw` had no knob at all."""
    fig, ax = sph.allsky_figure(projection="AIT")
    artists = sph.add_axis_inlay(ax, bg_edgecolor="#ff00ff", bg_lw=2.5)
    boxes = [a for a in artists if type(a).__name__ == "FancyBboxPatch"]
    assert boxes
    for b in boxes:
        assert tuple(round(v, 2) for v in b.get_edgecolor()[:3]) == (1.0, 0.0, 1.0)
        assert b.get_linewidth() == pytest.approx(2.5)
    plt.close(fig)


def test_compass_stroke_default_reaches_the_artist():
    """The theme-aware stroke must actually be applied.

    Regression: the path effects were built from the raw ``stroke_color``
    parameter *before* ``_ink_and_stroke`` resolved it, so the default
    ``None`` reached ``_stroke_path_effects`` unresolved, which returns None —
    and the compass drew unstroked. An explicit ``stroke_color=`` still
    worked, which is exactly why it read as "the default is no stroke".
    """
    import matplotlib.pyplot as plt

    import skyplothelper as sph
    fig = plt.figure()
    ax = sph.make_wcs_frame(111, "TAN", center=(180.0, 30.0), fov_deg=1.0,
                            fig=fig)
    artists = sph.add_compass(ax)
    for a in artists:
        patch = a.arrow_patch if hasattr(a, "arrow_patch") else a
        assert patch.get_path_effects(), "compass artist lost its stroke"
    plt.close(fig)


def test_compass_ink_and_stroke_flip_together():
    """They are a contrast pair — moving one alone makes the mark *less*
    legible than the hard-coded original, which is the whole reason
    _ink_and_stroke resolves them side by side."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    import skyplothelper as sph

    def _pair(theme):
        sph.set_theme(theme)
        fig = plt.figure()
        ax = sph.make_wcs_frame(111, "TAN", center=(180.0, 30.0),
                                fov_deg=1.0, fig=fig)
        patch = sph.add_compass(ax)[0].arrow_patch
        ink = mcolors.to_rgb(patch.get_edgecolor())
        stroke = mcolors.to_rgb(patch.get_path_effects()[0]._gc["foreground"])
        plt.close(fig)
        return ink, stroke

    light_ink, light_stroke = _pair("publication")
    dark_ink, dark_stroke = _pair("dark_sky")
    sph.set_theme("publication")
    # Each theme must contrast internally, and the two must not agree —
    # if they did, one of them is not following the canvas.
    assert sum(light_ink) < sum(light_stroke)
    assert sum(dark_ink) > sum(dark_stroke)


# --- add_colorbar adaptive minor ticks (ported from quicklook) ---------------

def _cbar_minor(cbar):
    import matplotlib.pyplot as plt  # noqa: F401
    ax = (cbar.ax.yaxis if cbar.orientation == "vertical"
          else cbar.ax.xaxis)
    return list(ax.get_minorticklocs())


def test_add_colorbar_default_has_minor_ticks():
    """Deliberate departure from bare matplotlib (which draws none). The knob
    is adaptive by default; this is the visible change ported from quicklook.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    import skyplothelper as sph
    fig, ax = plt.subplots()
    im = ax.imshow(np.random.default_rng(0).random((16, 16)), vmin=0, vmax=1)
    cb = sph.add_colorbar(im, ax=ax)
    assert len(_cbar_minor(cb)) > 0
    plt.close(fig)


def test_add_colorbar_log_bar_gets_decade_multiples():
    """A compressed bar gets 1/2/3/5 x 10^k, not an even subdivision."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

    import skyplothelper as sph
    fig, ax = plt.subplots()
    im = ax.imshow(np.random.default_rng(1).random((16, 16)) * 999 + 1,
                   norm=LogNorm(vmin=1, vmax=1000))
    cb = sph.add_colorbar(im, ax=ax)
    mantissas = {round(v / 10 ** np.floor(np.log10(v)), 3)
                 for v in _cbar_minor(cb) if v > 0}
    assert mantissas <= {1.0, 2.0, 3.0, 5.0}
    assert mantissas, "no decade minor ticks placed"
    plt.close(fig)


def test_add_colorbar_minor_ticks_false_restores_matplotlib_look():
    import matplotlib.pyplot as plt
    import numpy as np

    import skyplothelper as sph
    fig, ax = plt.subplots()
    im = ax.imshow(np.random.default_rng(2).random((16, 16)))
    cb = sph.add_colorbar(im, ax=ax, minor_ticks=False)
    assert _cbar_minor(cb) == []
    plt.close(fig)


def test_add_colorbar_minor_ticks_explicit_sequence():
    import matplotlib.pyplot as plt
    import numpy as np

    import skyplothelper as sph
    fig, ax = plt.subplots()
    im = ax.imshow(np.random.default_rng(3).random((16, 16)), vmin=0, vmax=10)
    cb = sph.add_colorbar(im, ax=ax, minor_ticks=[1, 3, 7])
    assert _cbar_minor(cb) == [1, 3, 7]
    plt.close(fig)


def test_add_colorbar_horizontal_ticks_the_x_axis():
    """The helper must pick the tick axis from the bar's orientation."""
    import matplotlib.pyplot as plt
    import numpy as np

    import skyplothelper as sph
    fig, ax = plt.subplots()
    im = ax.imshow(np.random.default_rng(4).random((16, 16)))
    cb = sph.add_colorbar(im, ax=ax, orientation="horizontal")
    assert cb.orientation == "horizontal"
    assert len(_cbar_minor(cb)) > 0
    plt.close(fig)


# --- add_colorbar tick_format (opt-in precision) -----------------------------

def _cbar_labels(cbar):
    import matplotlib.pyplot as plt  # noqa: F401
    cbar.ax.figure.canvas.draw()
    ax = (cbar.ax.yaxis if cbar.orientation == "vertical" else cbar.ax.xaxis)
    return [t.get_text() for t in ax.get_ticklabels() if t.get_text()]


def _bar_0_3(**kw):
    import matplotlib.pyplot as plt
    import numpy as np

    import skyplothelper as sph
    fig, ax = plt.subplots()
    im = ax.imshow(np.random.default_rng(0).random((16, 16)) * 3,
                   vmin=0, vmax=3)
    cb = sph.add_colorbar(im, ax=ax, **kw)
    labels = _cbar_labels(cb)
    plt.close(fig)
    return labels


def test_tick_format_is_opt_in():
    """Default leaves matplotlib's labels — precision reformatting rewrites
    every label, so unlike minor_ticks it is not on by default."""
    assert _bar_0_3() == _bar_0_3(tick_format=None)


def test_tick_format_auto_follows_the_range():
    labels = _bar_0_3(tick_format="auto")
    assert "0.5" in labels and "1" in labels        # trimmed, range-aware


@pytest.mark.parametrize("spec", ["%.3f", "{x:.3f}"])
def test_tick_format_accepts_either_format_style(spec):
    """Both printf ('%.3f') and str.format ('{x:.3f}') are in wide use, so
    both must be honored — set_major_formatter(str) alone would print an
    old-style string verbatim."""
    labels = _bar_0_3(tick_format=spec)
    assert "0.000" in labels and "1.000" in labels


def test_add_colorbar_decrowds_piled_up_asinh_minor_ticks():
    """decade_minor_ticks enumerates by value decade, blind to the display, so
    on an asinh bar sitting in its linear regime the low decades map to nearly
    one spot and pile into a smudge. The de-crowd filter keeps only the ones
    the norm visibly separates.

    Asserted through the norm (where the pile-up exists), not on raw positions
    — a decade_minor_ticks unit test cannot see this, since the crowding only
    appears once the norm maps the positions onto the bar.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.visualization import AsinhStretch, ImageNormalize

    import skyplothelper as sph
    from skyplothelper._colorbar import _MIN_TICK_FRAC, decade_minor_ticks

    fig, ax = plt.subplots()
    img = np.abs(np.random.default_rng(0).normal(0, 5e-3, (30, 30)))
    img[10:20, 10:20] += 0.04
    im = ax.imshow(img, norm=ImageNormalize(vmin=-1.3e-3, vmax=0.05,
                                            stretch=AsinhStretch()))
    cb = sph.add_colorbar(im, ax=ax)
    positions = list(cb.ax.yaxis.get_minorticklocs())

    # No two kept ticks land closer than min_frac in display space.
    ps = sorted(p for p in (float(cb.norm(v)) for v in positions)
                if np.isfinite(p))
    gaps = [b - a for a, b in zip(ps, ps[1:])]
    assert all(g >= _MIN_TICK_FRAC - 1e-9 for g in gaps), \
        "minor ticks still pile up in display space"

    # And it genuinely dropped some — the raw decade set crowds this bar.
    raw = decade_minor_ticks(-1.3e-3, 0.05)
    assert len(positions) < len(raw)
    plt.close(fig)


def test_add_colorbar_strokes_axis_label():
    """stroke_targets='both' (default) strokes the colorbar's axis label too,
    not only the tick marks / spine — the axis label is set after the polish
    pass, so the stroke must persist onto the later-set text."""
    import numpy as np

    from skyplothelper.overlays.annotations import add_colorbar
    fig, ax = plt.subplots()
    im = ax.imshow(np.random.default_rng(0).random((16, 16)))
    cbar = add_colorbar(im, ax=ax, label="Jy/beam", stroke_color="k",
                        stroke_lw=2.4)
    assert cbar.ax.yaxis.label.get_text() == "Jy/beam"
    assert cbar.ax.yaxis.label.get_path_effects(), "colorbar label not stroked"
    plt.close(fig)


def test_add_colorbar_no_stroke_leaves_label_plain():
    import numpy as np

    from skyplothelper.overlays.annotations import add_colorbar
    fig, ax = plt.subplots()
    im = ax.imshow(np.random.default_rng(1).random((16, 16)))
    cbar = add_colorbar(im, ax=ax, label="Jy/beam")
    assert not cbar.ax.yaxis.label.get_path_effects()
    plt.close(fig)


def test_add_colorbar_stroke_targets_spine_skips_label():
    """stroke_targets='spine' strokes only the outline, not the axis label."""
    import numpy as np

    from skyplothelper.overlays.annotations import add_colorbar
    fig, ax = plt.subplots()
    im = ax.imshow(np.random.default_rng(2).random((16, 16)))
    cbar = add_colorbar(im, ax=ax, label="Jy/beam", stroke_color="k",
                        stroke_targets="spine")
    assert not cbar.ax.yaxis.label.get_path_effects()
    plt.close(fig)


def test_add_axis_inlay_wireframe_stroke_opt_in():
    """The wireframe reference frame is subtle by default (unstroked); the
    opt-in wireframe_stroke outlines it for a busy image."""
    import skyplothelper as sph
    fig = plt.figure(figsize=(4, 4))
    ax = sph.make_wcs_frame(projection="AIT", center=0, fig=fig, grid=False)
    sph.add_axis_inlay(ax, wireframe=True)
    assert not any(ln.get_path_effects() for ln in ax.lines)
    fig2 = plt.figure(figsize=(4, 4))
    ax2 = sph.make_wcs_frame(projection="AIT", center=0, fig=fig2, grid=False)
    sph.add_axis_inlay(ax2, wireframe=True, wireframe_stroke="k")
    assert sum(1 for ln in ax2.lines if ln.get_path_effects()) >= 3
    plt.close("all")
