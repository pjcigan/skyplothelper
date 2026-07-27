"""Smoke tests for skyplothelper.globe.decorations."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from skyplothelper.coord_overlay import add_overlay_ticks
from skyplothelper.globe.decorations import (
    add_checkered_border,
    add_compass_rose,
    add_pole_rod,
    add_scale_bar,
    add_scale_bar_curved_parallel,
    add_scale_bar_cylindrical,
    add_surface_compass,
    highlight_great_circle,
    highlight_meridian_tracer,
    plot_ortho_grid,
)
from skyplothelper.globe.frame import make_globe_frame, make_planet_frame
from skyplothelper.wcs_frame import make_wcs_frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _overlay_label_kind_counts(ax):
    """(#lon, #lat, #auto-tagged) sph overlay tick labels on the axes."""
    ax.figure.canvas.draw()
    lon = lat = auto = 0
    for t in ax.texts:
        if not getattr(t, "_sph_overlay_ticklabel", False):
            continue
        k = getattr(t, "_sph_overlay_kind", None)
        lon += k == "lon"
        lat += k == "lat"
        auto += bool(getattr(t, "_sph_auto_overlay", False))
    return lon, lat, auto


def test_globe_auto_inframe_labels_are_auto_tagged():
    """make_globe_frame's auto in-frame labels carry _sph_auto_overlay so a
    later user add_overlay_ticks can replace them."""
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=20)
    lon, lat, auto = _overlay_label_kind_counts(ax)
    assert lon > 0 and lat > 0
    assert auto == lon + lat   # every auto label tagged


def test_globe_overlay_replace_is_kind_aware_no_double():
    """Placing custom Dec labels (lat_at, lon_at=None) clears the auto LAT
    overlay but KEEPS the auto RA (lon) labels — no doubling, RA preserved."""
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=20)
    lon0, lat0, _ = _overlay_label_kind_counts(ax)
    assert lon0 > 0 and lat0 > 0
    add_overlay_ticks(ax, lat_at="lon=105", lon_at=None,
                      suppress_default="none")
    lon1, lat1, auto1 = _overlay_label_kind_counts(ax)
    # RA labels (auto lon) survive; Dec replaced (auto lat gone), not doubled
    assert lon1 == lon0
    assert auto1 == lon0          # only the lon auto set remains tagged
    assert 0 < lat1 < lat0 + lat0  # a single (new) Dec set, not auto+new


def test_globe_inframe_label_spacing_follows_grid():
    """lat_deg_spacing coarsens the in-frame lat labels (they sit on the grid
    lines) instead of the fixed all-sky 15° default."""
    def lat_vals(ax):
        ax.figure.canvas.draw()
        out = set()
        for t in ax.texts:
            if (getattr(t, "_sph_overlay_ticklabel", False)
                    and getattr(t, "_sph_overlay_kind", None) == "lat"):
                s = t.get_text().replace("°", "").replace("+", "").replace(
                    "−", "-")
                try:
                    out.add(int(round(float(s))))
                except ValueError:
                    pass
        return out
    ax30 = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0,
                            lat_deg_spacing=30)
    v30 = lat_vals(ax30)
    assert v30 and all(v % 30 == 0 for v in v30)   # multiples of 30 only
    ax45 = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0,
                            lat_deg_spacing=45)
    assert lat_vals(ax45) <= {-45, 0, 45}


def test_plot_ortho_grid_smoke():
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    plot_ortho_grid(ax)


def _meridian_segs(ax, color="crimson"):
    import matplotlib.colors as mcolors
    ax.figure.canvas.draw()
    tr = [ln for ln in ax.lines
          if mcolors.same_color(ln.get_color(), color)]
    solid = [ln for ln in tr if ln.get_linestyle() == "-"
             and np.isfinite(np.asarray(ln.get_data()[0], float)).any()]
    dash = [ln for ln in tr if ln.get_linestyle() == "--"
            and np.isfinite(np.asarray(ln.get_data()[0], float)).any()]
    return tr, solid, dash


@pytest.mark.parametrize("spec", [
    dict(pole=(60, 30)),
    dict(points=((-30, -20), (60, 40))),
    dict(inclination=55, node=10),
])
def test_highlight_great_circle_all_specs_render_front_and_back(spec):
    """Each great-circle specification (pole / two points / inclination+node)
    traces the full ring on a globe: front solid AND far dashed both render,
    matched color + lw."""
    ax = make_globe_frame(111, center_LONdeg=20, center_LATdeg=20)
    highlight_great_circle(ax, color="crimson", lw=2.0, **spec)
    tr, solid, dash = _meridian_segs(ax)
    assert solid, "front (solid) half did not render"
    assert dash, "far (dashed) half did not render (affine path)"
    assert {round(ln.get_linewidth(), 2) for ln in tr} == {2.0}


def test_highlight_great_circle_requires_exactly_one_spec():
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    with pytest.raises(ValueError, match="exactly one"):
        highlight_great_circle(ax)                       # none
    with pytest.raises(ValueError, match="exactly one"):
        highlight_great_circle(ax, pole=(0, 90), node=0)  # two


def test_highlight_great_circle_pole_math():
    """pole=(0, 90) is the equator; a pole on the equator gives a pole-to-pole
    ring (a meridian)."""
    from skyplothelper.globe.decorations import _great_circle_ring
    _, lat_eq = _great_circle_ring(0, 90, 200)
    assert np.max(np.abs(lat_eq)) < 0.5             # equator
    _, lat_mer = _great_circle_ring(90, 0, 200)
    assert np.max(lat_mer) > 88 and np.min(lat_mer) < -88   # reaches the poles


def test_meridian_tracer_matches_great_circle_through_poles():
    """highlight_meridian_tracer is now a thin wrapper: meridian_lon=M is the
    great circle with pole on the equator at lon M+90."""
    ax1 = make_globe_frame(111, center_LONdeg=30, center_LATdeg=15)
    highlight_meridian_tracer(ax1, meridian_lon=0, color="crimson", lw=1.5)
    segs1 = _meridian_segs(ax1)[0]
    plt.close("all")
    ax2 = make_globe_frame(111, center_LONdeg=30, center_LATdeg=15)
    highlight_great_circle(ax2, pole=(90, 0), color="crimson", lw=1.5)
    segs2 = _meridian_segs(ax2)[0]
    # same number of front/back line artists from the same ring machinery
    assert len(segs1) == len(segs2) and len(segs1) > 0


def test_meridian_tracer_full_ring_matched_on_wcs_globe():
    """highlight_meridian_tracer traces the full great circle (meridian +
    antimeridian) on a make_globe_frame WCSAxes — front solid AND far dashed
    BOTH render (far side via the shared affine), at matched color + lw."""
    ax = make_globe_frame(111, center_LONdeg=40, center_LATdeg=25)
    highlight_meridian_tracer(ax, meridian_lon=0, color="crimson", lw=2.0)
    tr, solid, dash = _meridian_segs(ax)
    assert solid, "front (solid) half did not render"
    assert dash, "far (dashed) half did not render (affine path)"
    # matched weight front/back (only the linestyle differs)
    assert {round(ln.get_linewidth(), 2) for ln in tr} == {2.0}


def test_meridian_tracer_on_plain_axes():
    """On a plain mpl axes the tracer draws via orthographic_forward data
    coords (no WCS / affine needed) — full ring, front + back."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    highlight_meridian_tracer(ax, meridian_lon=0, lon_0=40, lat_0=25,
                              color="crimson", lw=2.0)
    _, solid, dash = _meridian_segs(ax)
    assert solid and dash


def test_meridian_tracer_autodetects_wcs_center():
    """lon_0/lat_0 default to the globe's CRVAL, so the tracer matches the
    frame without the caller repeating the center."""
    ax = make_globe_frame(111, center_LONdeg=120, center_LATdeg=-30)
    # no lon_0/lat_0 passed → must still render a full ring
    highlight_meridian_tracer(ax, meridian_lon=120, color="crimson")
    _, solid, dash = _meridian_segs(ax)
    assert solid and dash


def test_plot_ortho_grid_far_side_renders_on_wcs_globe():
    """show_back must draw the far hemisphere on a make_globe_frame WCSAxes too
    (a SIN WCS NaNs far-side world->pixel, so the lines used to vanish). The
    far lines are blitted via an affine calibrated against the front hemisphere
    — independently styled (color/lw/ls/alpha) like on a plain axes."""
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=40, center_LATdeg=25)
    plot_ortho_grid(ax, show_back=True, front_color="navy", front_ls="-",
                    back_color="crimson", back_lw=1.2, back_ls=":")
    ax.figure.canvas.draw()
    import matplotlib.colors as mcolors
    back = [ln for ln in ax.lines
            if ln.get_linestyle() == ":"
            and mcolors.same_color(ln.get_color(), "crimson")
            and np.isfinite(np.asarray(ln.get_data()[0], float)).any()]
    assert back, "far-side lines did not render on the WCS globe"
    assert back[0].get_linewidth() == 1.2     # back styling honored


def test_plot_ortho_grid_show_back_false_omits_far_side():
    """show_back=False draws no far-side lines on a WCS globe."""
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=40, center_LATdeg=25)
    plot_ortho_grid(ax, show_back=False, back_ls=":")
    ax.figure.canvas.draw()
    dotted = [ln for ln in ax.lines if ln.get_linestyle() == ":"]
    assert not dotted


def test_add_checkered_border_smoke():
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    add_checkered_border(ax)


def test_add_compass_rose_smoke():
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    add_compass_rose(ax)


def test_add_scale_bar_cylindrical_smoke():
    """Cylindrical scale bar exercised on a CAR axes."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="CAR", fig=fig)
    add_scale_bar_cylindrical(ax, lat=0.0, length_km=1000)


def test_add_scale_bar_curved_parallel_smoke():
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    add_scale_bar_curved_parallel(ax, length_km=1000)


# ===== add_pole_rod =====

def test_add_pole_rod_equator_on_single_segment():
    """At lat_0=0 the poles are on the limb — single line, no occlusion split."""
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    artists = add_pole_rod(ax, length=1.5)
    # One Line2D for the rod, no markers.
    assert len(artists) == 1
    line = artists[0]
    xs, ys = line.get_xdata(), line.get_ydata()
    # Rod runs along Y at lat_0=0 (no in-plane rotation from default lonpole=0).
    assert np.isclose(xs[0], xs[1], atol=1e-6)
    # SIN disk radius in pixels = (180/π) / |cdelt|.
    cdelt = abs(ax.wcs.wcs.cdelt[0])
    R_disk = (180.0 / np.pi) / cdelt
    xc = ax.wcs.wcs.crpix[0] - 1.0
    yc = ax.wcs.wcs.crpix[1] - 1.0
    # length=1.5 → endpoints at 1.5 × R_disk from center → outside the disk.
    for x, y in zip(xs, ys):
        assert np.isclose(np.hypot(x - xc, y - yc), 1.5 * R_disk, rtol=1e-3)


def test_add_pole_rod_tilted_two_segments():
    """At lat_0!=0 with occlude_back=True the rod splits into front + back."""
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=30)
    artists = add_pole_rod(ax, occlude_back=True)
    # Two Line2D: back and front halves.
    assert len(artists) == 2
    back, front = artists
    # zorders distinguish them.
    assert back.get_zorder() < front.get_zorder()
    assert front.get_zorder() == 10
    assert back.get_zorder() == -5


def test_add_pole_rod_back_segment_trimmed_to_limb():
    """The back (far-pole) segment must start AT the limb, not inside the disk —
    otherwise its near-limb sliver pokes through the body's texture/raster gap
    on a tilted globe. The front segment still runs pole→tip across the disk."""
    plt.figure()
    ax = make_globe_frame(111, radesys="ITRS", center_LONdeg=0,
                          center_LATdeg=25, lonpole=-23.44)
    ax.figure.canvas.draw()
    back, front = add_pole_rod(ax, occlude_back=True)

    wcs = ax.wcs
    xc = float(wcs.wcs.crpix[0]) - 1.0
    yc = float(wcs.wcs.crpix[1]) - 1.0
    limb = wcs.wcs_world2pix([[float(wcs.wcs.crval[0]), 25 - 89.999]], 0)[0]
    R = float(np.hypot(limb[0] - xc, limb[1] - yc))

    (bx0, _bx1), (by0, _by1) = back.get_data()
    # back segment starts on/outside the limb (within 1 px), not inside the disk
    assert np.hypot(bx0 - xc, by0 - yc) >= R - 1.0
    # front segment still crosses INTO the disk (pole is inside the limb)
    (fx0, _fx1), (fy0, _fy1) = front.get_data()
    assert np.hypot(fx0 - xc, fy0 - yc) < R


def test_add_pole_rod_occlude_back_false_single_line():
    """occlude_back=False forces a single end-to-end line even when tilted."""
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=30)
    artists = add_pole_rod(ax, occlude_back=False)
    assert len(artists) == 1


def test_add_pole_rod_front_pole_selection():
    """For lat_0>0, the N pole gets zorder_front; for lat_0<0, the S pole does."""
    def _front_end(ax, artists):
        # The high-zorder line is the front rod; its second endpoint is the
        # front-pole extension tip. Project N and S poles and check which
        # one matches the line's terminating direction from disk center.
        back, front = artists
        xc = ax.wcs.wcs.crpix[0] - 1.0
        yc = ax.wcs.wcs.crpix[1] - 1.0
        # The front rod runs from front-pole's pixel → its extension.
        fx0, fy0 = front.get_xdata()[0], front.get_ydata()[0]
        # Direction from disk center to the front rod's start (front pole position):
        return fx0 - xc, fy0 - yc

    plt.figure()
    ax_north = make_globe_frame(111, center_LONdeg=0, center_LATdeg=45)
    artists_n = add_pole_rod(ax_north)
    n_pole_pix = ax_north.wcs.wcs_world2pix(np.array([[0.0, 90.0]]), 0)[0]
    xc, yc = ax_north.wcs.wcs.crpix[0] - 1.0, ax_north.wcs.wcs.crpix[1] - 1.0
    expected_dir = (n_pole_pix[0] - xc, n_pole_pix[1] - yc)
    actual_dir = _front_end(ax_north, artists_n)
    assert np.isclose(actual_dir[0], expected_dir[0])
    assert np.isclose(actual_dir[1], expected_dir[1])

    plt.close("all")
    plt.figure()
    ax_south = make_globe_frame(111, center_LONdeg=0, center_LATdeg=-45)
    artists_s = add_pole_rod(ax_south)
    s_pole_pix = ax_south.wcs.wcs_world2pix(np.array([[0.0, -90.0]]), 0)[0]
    xc, yc = ax_south.wcs.wcs.crpix[0] - 1.0, ax_south.wcs.wcs.crpix[1] - 1.0
    expected_dir = (s_pole_pix[0] - xc, s_pole_pix[1] - yc)
    actual_dir = _front_end(ax_south, artists_s)
    assert np.isclose(actual_dir[0], expected_dir[0])
    assert np.isclose(actual_dir[1], expected_dir[1])


def test_add_pole_rod_length_scales_endpoints():
    """The rod extension length scales the (pole - center) vector."""
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    a1 = add_pole_rod(ax, length=1.0)
    a2 = add_pole_rod(ax, length=2.0)
    # length=1.0 puts endpoints at the projected poles; length=2.0 puts them twice as far from center.
    xc = ax.wcs.wcs.crpix[0] - 1.0
    yc = ax.wcs.wcs.crpix[1] - 1.0
    d1 = np.hypot(a1[0].get_xdata()[1] - xc, a1[0].get_ydata()[1] - yc)
    d2 = np.hypot(a2[0].get_xdata()[1] - xc, a2[0].get_ydata()[1] - yc)
    assert np.isclose(d2 / d1, 2.0, rtol=1e-3)


def test_add_pole_rod_end_markers():
    """Specifying end_marker adds two marker Line2Ds in addition to the rod lines."""
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=30)
    artists = add_pole_rod(ax, end_marker='o', end_marker_size=10)
    # 2 rod halves + 2 marker artists = 4.
    assert len(artists) == 4


def test_add_pole_rod_clip_off():
    """All rod artists have clip_on=False so they extend past the frame."""
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=30)
    artists = add_pole_rod(ax, end_marker='o')
    for a in artists:
        assert a.get_clip_on() is False


def test_add_pole_rod_rejects_non_wcs():
    """Plain matplotlib axes are rejected with TypeError."""
    fig, ax = plt.subplots()
    with pytest.raises(TypeError, match="WCSAxes"):
        add_pole_rod(ax)


def test_add_pole_rod_rejects_non_sin_projection():
    """Non-SIN zenithal projections are rejected with a clear message."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="CAR", fig=fig)
    with pytest.raises(ValueError, match="SIN"):
        add_pole_rod(ax)


def test_add_pole_rod_pole_singularity_returns_empty():
    """Looking straight down the rotation axis: poles collapse, no rod drawn."""
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=90)
    artists = add_pole_rod(ax)
    assert artists == []


def test_add_pole_rod_exported_at_package_root():
    """sph.add_pole_rod is wired in the public API."""
    import skyplothelper as sph
    assert hasattr(sph, "add_pole_rod")
    assert sph.add_pole_rod is add_pole_rod


def test_add_pole_rod_default_stroke_applied():
    """Default look: bone-white core + dark stroke via PathEffects."""
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=30)
    artists = add_pole_rod(ax, end_marker='o')
    for a in artists:
        effects = a.get_path_effects()
        # One effect: withStroke draws the stroke + the core on top.
        assert len(effects) == 1


def test_add_pole_rod_stroke_disabled():
    """stroke_color=None disables the stroke on all artists."""
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=30)
    artists = add_pole_rod(ax, stroke_color=None, end_marker='o')
    for a in artists:
        assert a.get_path_effects() == []


def test_add_pole_rod_stroke_skipped_when_lw_ge_stroke_lw():
    """If stroke_lw <= linewidth there's no visible stroke — skip the effect."""
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=30)
    artists = add_pole_rod(ax, linewidth=4.0, stroke_lw=4.0)
    for a in artists:
        assert a.get_path_effects() == []


# ===== add_compass_rose stroke =====

def _compass_label_texts(anchor):
    """Pull the matplotlib Text artists out of the compass rose anchor."""
    from matplotlib.text import Text
    da = anchor.get_child()
    return [a for a in da.get_children() if isinstance(a, Text)]


def test_add_compass_rose_default_stroke_on_labels():
    """Default polish: N/S/E/W labels get a dark stroke."""
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    anchor = add_compass_rose(ax)
    texts = _compass_label_texts(anchor)
    assert len(texts) == 4  # N, S, E, W
    for t in texts:
        assert len(t.get_path_effects()) == 1


def test_add_compass_rose_stroke_disabled():
    """stroke_color=None disables the stroke on labels."""
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    anchor = add_compass_rose(ax, stroke_color=None)
    for t in _compass_label_texts(anchor):
        assert t.get_path_effects() == []


def test_add_compass_rose_arrow_style_stroke():
    """The arrow-style single 'N' label also gets the stroke."""
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    anchor = add_compass_rose(ax, style='arrow')
    texts = _compass_label_texts(anchor)
    assert len(texts) == 1  # only the N label
    assert len(texts[0].get_path_effects()) == 1


def _compass_star_polys(anchor):
    """Pull the four star-point Polygon artists out of the rose anchor."""
    import matplotlib.patches as mpatches
    da = anchor.get_child()
    return [a for a in da.get_children() if isinstance(a, mpatches.Polygon)]


def test_add_compass_rose_default_hollow_points_white():
    """Default: filled points use `color`, hollow points are white."""
    import matplotlib.colors as mcolors
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    anchor = add_compass_rose(ax, color='k')
    facecolors = {mcolors.to_hex(p.get_facecolor())
                  for p in _compass_star_polys(anchor)}
    assert mcolors.to_hex('white') in facecolors
    assert mcolors.to_hex('k') in facecolors


def test_add_compass_rose_color_alt_overrides_hollow_fill():
    """color_alt recolors the hollow points (parity with surface compass)."""
    import matplotlib.colors as mcolors
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    anchor = add_compass_rose(ax, color='navy', color_alt='gold')
    facecolors = {mcolors.to_hex(p.get_facecolor())
                  for p in _compass_star_polys(anchor)}
    assert mcolors.to_hex('gold') in facecolors
    assert mcolors.to_hex('navy') in facecolors
    assert mcolors.to_hex('white') not in facecolors


# ===== add_scale_bar_curved_parallel stroke =====

def _scale_bar_annotations(ax):
    """Pull the matplotlib Annotation artists added to the axes."""
    from matplotlib.text import Annotation
    return [c for c in ax.get_children() if isinstance(c, Annotation)]


def test_add_scale_bar_curved_parallel_default_preserves_white_stroke():
    """The white label stroke is the default."""
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    add_scale_bar_curved_parallel(ax, length_km=1000)
    annos = _scale_bar_annotations(ax)
    assert len(annos) >= 1
    for a in annos:
        effects = a.get_path_effects()
        assert len(effects) == 1  # just withStroke (core drawn by the effect)


def test_add_scale_bar_curved_parallel_stroke_disabled():
    """Setting stroke_color=None drops the stroke."""
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    add_scale_bar_curved_parallel(ax, length_km=1000, stroke_color=None)
    for a in _scale_bar_annotations(ax):
        # Annotation reports None when never set; Line2D reports [].
        assert not a.get_path_effects()


# ===== add_scale_bar_cylindrical stroke =====

def test_add_scale_bar_cylindrical_default_stroke():
    """Text labels get a thin white stroke by default."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="CAR", fig=fig)
    add_scale_bar_cylindrical(ax, lat=0.0, length_km=1000)
    annos = _scale_bar_annotations(ax)
    assert len(annos) >= 1
    for a in annos:
        assert len(a.get_path_effects()) == 1


def test_add_scale_bar_cylindrical_stroke_disabled():
    """Setting stroke_color=None drops the stroke across all labels."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="CAR", fig=fig)
    add_scale_bar_cylindrical(ax, lat=0.0, length_km=1000, stroke_color=None)
    for a in _scale_bar_annotations(ax):
        # Annotation reports None when never set; Line2D reports [].
        assert not a.get_path_effects()


# ===== add_scale_bar dispatcher =====

def test_add_scale_bar_routes_sin_to_curved_parallel():
    """A SIN globe should dispatch to add_scale_bar_curved_parallel."""
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    n_lines_before = len(ax.lines)
    add_scale_bar(ax, length_km=1000)
    # Style='plain' adds a thick polyline along the parallel.
    assert len(ax.lines) > n_lines_before


def test_add_scale_bar_routes_ait_pseudocylindrical():
    """AIT (Aitoff pseudocylindrical FITS) dispatches to curved_parallel."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", center=0, fig=fig)
    n_lines_before = len(ax.lines)
    add_scale_bar(ax, length_km=2000)
    assert len(ax.lines) > n_lines_before


def test_add_scale_bar_routes_mol_pseudocylindrical():
    """MOL (Mollweide pseudocylindrical FITS) dispatches to curved_parallel."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="MOL", center=0, fig=fig)
    n_lines_before = len(ax.lines)
    add_scale_bar(ax, length_km=2000)
    assert len(ax.lines) > n_lines_before


def test_add_scale_bar_routes_custom_transform_pseudocylindrical():
    """Robinson (custom mpl-transform projection) dispatches to curved_parallel."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="robinson", center=0, fig=fig)
    # On custom-transform WCSAxes, ax.wcs is None — the dispatcher must still
    # recognize it as a WCSAxes and route to the curved-parallel helper.
    assert ax.wcs is None
    n_patches_before = len(ax.patches)
    add_scale_bar(ax, length_km=2000, style='checkered', segment_km=500)
    assert len(ax.patches) > n_patches_before


def test_add_scale_bar_routes_car_to_cylindrical():
    """A CAR axes should dispatch to add_scale_bar_cylindrical."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="CAR", fig=fig)
    n_patches_before = len(ax.patches)
    add_scale_bar(ax, lat=0.0, length_km=1000)
    # add_scale_bar_cylindrical adds a Rectangle patch (style='plain').
    assert len(ax.patches) > n_patches_before


def test_add_scale_bar_routes_mer_to_cylindrical():
    """A MER (Mercator) WCSAxes should also dispatch to cylindrical."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="MER", fig=fig)
    n_patches_before = len(ax.patches)
    add_scale_bar(ax, lat=0.0, length_km=1000)
    assert len(ax.patches) > n_patches_before


def test_add_scale_bar_unsupported_projection_raises():
    """Cartopy Mercator (data units = meters, not degrees) is unsupported.

    The dispatcher should refuse with a clear error pointing at
    matplotlib-scalebar.
    """
    try:
        import cartopy.crs as ccrs
    except ImportError:
        pytest.skip("cartopy not installed")
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Mercator()})
    with pytest.raises(ValueError, match="matplotlib-scalebar"):
        add_scale_bar(ax, length_km=1000)


def test_add_scale_bar_plain_mpl_axes_routes_cylindrical():
    """Plain mpl axes (no WCS) fall back to the cylindrical path."""
    fig, ax = plt.subplots()
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    # Should not raise; the cylindrical helper's units_per_deg_x=1.0
    # fallback handles plain axes with degree-valued data coords.
    add_scale_bar(ax, lat=0.0, length_km=5000)


def test_add_scale_bar_exported_at_package_root():
    """sph.add_scale_bar is wired in the public API."""
    import skyplothelper as sph
    assert hasattr(sph, "add_scale_bar")
    assert sph.add_scale_bar is add_scale_bar
    assert hasattr(sph, "add_scale_bar_cylindrical")
    assert sph.add_scale_bar_cylindrical is add_scale_bar_cylindrical
    assert hasattr(sph, "add_scale_bar_curved_parallel")
    assert sph.add_scale_bar_curved_parallel is add_scale_bar_curved_parallel


# ===== add_surface_compass =====

def test_surface_compass_star_two_tone_blades():
    """style='star' renders a 4-point rose whose blades are split into a dark
    (color) half and a light (color_alt) half — both colors appear."""
    import matplotlib.colors as mcolors
    ax = make_planet_frame(111, center_LONdeg=-20, center_LATdeg=20)
    ax.figure.canvas.draw()
    r = add_surface_compass(ax, -20, 20, size_deg=18, color="navy",
                            color_alt="white", style="star")
    assert len(r["labels"]) == 4
    fcs = [s.get_facecolor() for s in r["shapes"]]
    n_dark = sum(mcolors.same_color(f, "navy") for f in fcs)
    n_light = sum(mcolors.same_color(f, "white") for f in fcs)
    assert n_dark >= 4 and n_light >= 4          # 4 blades × dark+light halves


def test_surface_compass_lines_full_toggle():
    """style='lines' draws N/E by default, N/E/S/W with full=True."""
    ax = make_planet_frame(111, center_LONdeg=0, center_LATdeg=10)
    ax.figure.canvas.draw()
    ne = add_surface_compass(ax, 0, 10, style="lines")
    assert len(ne["shapes"]) == 2
    plt.close("all")
    ax2 = make_planet_frame(111, center_LONdeg=0, center_LATdeg=10)
    ax2.figure.canvas.draw()
    full = add_surface_compass(ax2, 0, 10, style="lines", full=True)
    assert len(full["shapes"]) == 4


def test_surface_compass_arrow_north_and_east():
    """style='arrow' draws one connected N+E frame (right-angle tails) and
    labels both N and E."""
    ax = make_planet_frame(111, center_LONdeg=0, center_LATdeg=0)
    ax.figure.canvas.draw()
    r = add_surface_compass(ax, 0, 0, style="arrow")
    assert len(r["shapes"]) >= 1
    assert sorted(t.get_text() for t in r["labels"]) == ["E", "N"]


def test_surface_compass_works_on_allsky_aitoff():
    """Projection-agnostic: renders on an AIT all-sky frame, not just globes."""
    ax = make_wcs_frame(111, projection="AIT", center=0)
    ax.figure.canvas.draw()
    r = add_surface_compass(ax, 60, 30, size_deg=15, style="star")
    assert len(r["shapes"]) >= 8 and len(r["labels"]) == 4   # 4 blades × 2 tones


def test_surface_compass_culls_back_hemisphere_labels():
    """A label on the back hemisphere of a globe is skipped (no NaN-position
    matplotlib warning)."""
    import warnings
    ax = make_planet_frame(111, center_LONdeg=0, center_LATdeg=0)
    ax.figure.canvas.draw()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = add_surface_compass(ax, 88, 0, size_deg=20, style="lines",
                                full=True)
        ax.figure.canvas.draw()
    assert not any("posx and posy should be finite" in str(rec.message)
                   for rec in w)
    assert len(r["labels"]) < 4    # at least one cardinal label culled


def test_surface_compass_rejects_bad_style_and_plain_axes():
    ax = make_planet_frame(111, center_LONdeg=0, center_LATdeg=0)
    with pytest.raises(ValueError, match="style must be"):
        add_surface_compass(ax, 0, 0, style="fancy")
    fig, plain = plt.subplots()
    with pytest.raises(TypeError, match="WCSAxes"):
        add_surface_compass(plain, 0, 0)


def test_scale_bar_sublabels_are_muted_and_overridable():
    """Sub-tick labels are deliberately smaller AND lighter than the bar's
    own label. Resolving them to the primary ink would flatten that; only
    the dark case was broken."""
    import skyplothelper as sph
    from skyplothelper.globe.frame import make_globe_frame as _mgf

    def sublabels(theme=None, **kw):
        matplotlib.rcdefaults()
        sph.set_style(base="standard", **({"theme": theme} if theme else {}))
        ax = _mgf(projection="SIN")
        sph.add_scale_bar(ax, length_km=4000, segment_km=1000,
                          style="checkered", **kw)
        cols = sorted({t.get_color() for t in ax.texts
                       if t.get_text().replace(",", "").isdigit()})
        plt.close(ax.figure)
        return cols

    assert sublabels() == ["0.35"]                      # light unchanged
    assert sublabels("dark_sky") == ["0.65"]            # mirrored, legible
    assert sublabels(sublabel_color="#ff00ff") == ["#ff00ff"]
    matplotlib.rcdefaults()
    sph.set_style(base="standard")


@pytest.mark.parametrize("style", ["star", "arrow", "lines"])
def test_surface_compass_stroke_reaches_rose_and_arms(style):
    """The legibility stroke backs the rose/arms geometry too (all styles),
    not only the labels — an exterior outline independent of the two-tone
    fill, e.g. to lift the compass off ocean/ice."""
    import skyplothelper as sph
    fig = plt.figure(figsize=(4, 4))
    ax = sph.make_globe_frame(111, center_LONdeg=0, center_LATdeg=20,
                              lonpole=0)
    out = sph.add_surface_compass(ax, 0, 20, style=style, stroke_color="k",
                                  stroke_lw=3)
    shapes = out["shapes"]
    assert shapes and all(s.get_path_effects() for s in shapes)
    plt.close(fig)


def test_surface_compass_no_stroke_leaves_rose_plain():
    import skyplothelper as sph
    fig = plt.figure(figsize=(4, 4))
    ax = sph.make_globe_frame(111, center_LONdeg=0, center_LATdeg=20,
                              lonpole=0)
    out = sph.add_surface_compass(ax, 0, 20, style="star", stroke_color=None)
    assert not any(s.get_path_effects() for s in out["shapes"])
    plt.close(fig)
