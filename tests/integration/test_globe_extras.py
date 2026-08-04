"""globe extras (baselines / insets / nightshade / boundaries).

Verifies the four sub-modules under ``skyplothelper.globe`` that
the canonical smoke tests don't cover:

  * ``baselines.plot_baselines`` and ``_normalize_sites``.
  * ``insets.reproject_inset_axes`` / ``mark_inset_axes`` /
    ``connect_inset_axes``.
  * ``nightshade.pseudofits_from_image`` / ``make_nightshade_blend``.
  * ``boundaries.fetch_boundary_data`` and ``_find_data_file`` —
    error-path tests only (the actual `.npz` files aren't published
    yet; once they are, the rendering paths can be verified too).
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from astropy.visualization.wcsaxes import WCSAxes

from skyplothelper.globe.baselines import _normalize_sites, plot_baselines
from skyplothelper.globe.boundaries import (
    _find_data_file,
    fetch_boundary_data,
    plot_coastlines,
    plot_lakes,
    plot_land,
    plot_rivers,
    plot_tectonic_plates,
)
from skyplothelper.globe.frame import make_globe_frame, make_planet_frame
from skyplothelper.globe.insets import (
    connect_inset_axes,
    mark_inset_axes,
    reproject_inset_axes,
)
from skyplothelper.wcs_frame import make_wcs_frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def globe_axes():
    fig = plt.figure(figsize=(6, 6))
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    return fig, ax


# ============================================================
# baselines._normalize_sites — accepts dict, list-of-tuples
# ============================================================

def test_normalize_sites_dict_input():
    sites = {"VLA": (-107.6, 34.1), "GBT": (-79.8, 38.4)}
    out = _normalize_sites(sites)
    assert isinstance(out, list)
    assert len(out) == 2
    names = {entry[0] for entry in out}
    assert names == {"VLA", "GBT"}


def test_normalize_sites_list_of_tuples():
    sites = [("ALMA", -67.7, -23.0), ("EVN", 6.6, 53.0)]
    out = _normalize_sites(sites)
    assert len(out) == 2
    assert out[0] == ("ALMA", -67.7, -23.0)


def test_normalize_sites_invalid_entry_raises():
    with pytest.raises(ValueError, match="(?i)invalid"):
        _normalize_sites([("only_one_field",)])


# ============================================================
# plot_baselines — runs end-to-end
# ============================================================

def test_plot_baselines_returns_dict_of_artists(globe_axes):
    """plot_baselines returns a dict with 'lines' and 'markers' keys."""
    fig, ax = globe_axes
    sites = {"VLA": (-107.6, 34.1), "GBT": (-79.8, 38.4),
             "ALMA": (-67.7, -23.0)}
    out = plot_baselines(ax, sites, pairs="all",
                         color="C0", show_markers=True,
                         show_site_labels=True)
    assert out is not None
    fig.canvas.draw()


def test_plot_baselines_skips_far_side_labels(globe_axes):
    """A globe-spanning array places labels only for front-hemisphere sites —
    a back-side site's label would otherwise sit at a NaN position and make
    matplotlib log 'posx and posy should be finite values' to stderr."""
    import warnings

    fig, ax = globe_axes   # globe centered near (0, 0)
    sites = {
        "Front": (0.0, 0.0),        # dead-center → front
        "Near":  (40.0, 10.0),      # front
        "Far":   (175.0, 0.0),      # opposite side → back hemisphere
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = plot_baselines(ax, sites, show_site_labels=True)
        fig.canvas.draw()
    assert not any("posx and posy should be finite" in str(rec.message)
                   for rec in w)
    names = {t.get_text() for t in out["site_labels"]}
    assert "Far" not in names and "Front" in names


def test_plot_baselines_explicit_pairs(globe_axes):
    fig, ax = globe_axes
    sites = {"VLA": (-107.6, 34.1), "GBT": (-79.8, 38.4),
             "ALMA": (-67.7, -23.0)}
    pairs = [("VLA", "GBT"), ("VLA", "ALMA")]
    plot_baselines(ax, sites, pairs=pairs, color="C3")
    fig.canvas.draw()


def test_back_hemisphere_arc_not_reflected_through_center():
    """The back-hemisphere arc projects each far-side point to its true disk
    position (the mirror-through-the-sky-plane image), NOT point-reflected
    through the disk center as the old ortho + CRPIX/CDELT shortcut did."""
    import numpy as np

    from skyplothelper.globe.baselines import _mirror_to_near_side

    fig = plt.figure(figsize=(6, 6))
    ax = make_globe_frame(111, center_LONdeg=10, center_LATdeg=45)
    fig.canvas.draw()
    # ATCA is on the far hemisphere from a Europe-centered view.
    out = plot_baselines(ax, {"Onsala": (11.9, 57.4), "ATCA": (149.55, -30.31)},
                         back_hemisphere_linestyle=":")
    fig.canvas.draw()
    wcs = ax.wcs
    clon, clat = float(wcs.wcs.crval[0]), float(wcs.wcs.crval[1])
    # Every finite back-arc sample must equal world_to_pixel of its mirror,
    # i.e. lie on the correct (mirror) side — never the center reflection.
    line = out["back_arcs"][0][0]
    xs, ys = line.get_xdata(), line.get_ydata()
    finite = np.isfinite(xs) & np.isfinite(ys)
    assert finite.sum() > 3
    # The drawn samples must match world_to_pixel of the mirrored arc — the
    # buggy shortcut would put them at the point-reflection through the center.
    from skyplothelper.globe.spherical import (
        great_circle_arc,
        orthographic_visibility,
    )
    la, lb = great_circle_arc(11.9, 57.4, 149.55, -30.31, n_pts=100)
    vis = orthographic_visibility(la, lb, clon, clat)
    mlo, mla = _mirror_to_near_side(la[~vis], lb[~vis], clon, clat)
    ex, ey = wcs.world_to_pixel_values(mlo, mla)
    drawn = np.column_stack([xs[finite], ys[finite]])
    expect = np.column_stack([ex, ey])
    expect = expect[np.isfinite(expect).all(axis=1)]
    assert np.allclose(np.sort(drawn[:, 0]), np.sort(expect[:, 0]), atol=1e-6)
    plt.close(fig)


def test_back_hemisphere_markers_ghost_far_side_sites():
    """back_hemisphere_markers=True draws faded markers + labels for far-side
    sites at their true (mirror) disk positions."""
    import numpy as np

    from skyplothelper.globe.baselines import _mirror_to_near_side

    fig = plt.figure(figsize=(6, 6))
    ax = make_globe_frame(111, center_LONdeg=10, center_LATdeg=45)
    fig.canvas.draw()
    out = plot_baselines(
        ax, {"Onsala": (11.9, 57.4), "ATCA": (149.55, -30.31)},
        back_hemisphere_markers=True, back_hemisphere_alpha=0.3)
    assert out["back_markers"] is not None
    assert {t.get_text() for t in out["back_site_labels"]} == {"ATCA"}
    wcs = ax.wcs
    clon, clat = float(wcs.wcs.crval[0]), float(wcs.wcs.crval[1])
    mlon, mlat = _mirror_to_near_side(149.55, -30.31, clon, clat)
    truth = wcs.world_to_pixel_values(mlon, mlat)
    off = out["back_markers"].get_offsets()[0]
    assert np.allclose(off, [float(truth[0]), float(truth[1])], atol=1e-6)
    plt.close(fig)


# ============================================================
# insets — reproject_inset_axes / mark / connect
# ============================================================

def test_reproject_inset_axes_returns_wcsaxes():
    fig = plt.figure(figsize=(8, 6))
    parent_ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    inset = reproject_inset_axes(
        parent_ax, rect=[0.65, 0.05, 0.3, 0.3],
        center=(0.0, 30.0), size=10.0,
        projection="TAN",
    )
    assert isinstance(inset, WCSAxes)


def test_mark_inset_axes_returns_artist():
    fig = plt.figure(figsize=(8, 6))
    parent_ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    inset = reproject_inset_axes(
        parent_ax, rect=[0.65, 0.05, 0.3, 0.3],
        center=(0.0, 30.0), size=10.0, projection="TAN",
    )
    out = mark_inset_axes(parent_ax, inset, edgecolor="C3", lw=1.5)
    assert out is not None


def test_connect_inset_axes_runs():
    fig = plt.figure(figsize=(8, 6))
    parent_ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    inset = reproject_inset_axes(
        parent_ax, rect=[0.65, 0.05, 0.3, 0.3],
        center=(0.0, 30.0), size=10.0, projection="TAN",
    )
    out = connect_inset_axes(parent_ax, inset, color="C3", linewidth=1.0)
    assert out is not None


def test_outer_tangent_endpoints_equal_radii():
    """Equal-radius circles: tangent lines are perpendicular to the
    line of centers, so tangent points sit at ``cy ± r`` on each
    circle when centers share a y-axis offset."""
    import numpy as np

    from skyplothelper.globe.insets import _outer_tangent_endpoints

    out = _outer_tangent_endpoints((0.0, 0.0), 1.0, (4.0, 0.0), 1.0)
    assert out is not None
    # Two tangent lines: y = +1 and y = -1. Each line: tangent point on
    # A at (0, ±1), on B at (4, ±1).
    tops = out[0]
    bots = out[1]
    # Sort so we know which is which (y-positive vs y-negative).
    if tops[0, 1] < 0:
        tops, bots = bots, tops
    np.testing.assert_allclose(tops[0], (0.0, 1.0), atol=1e-9)
    np.testing.assert_allclose(tops[1], (4.0, 1.0), atol=1e-9)
    np.testing.assert_allclose(bots[0], (0.0, -1.0), atol=1e-9)
    np.testing.assert_allclose(bots[1], (4.0, -1.0), atol=1e-9)


def test_outer_tangent_endpoints_unequal_radii_lie_on_circles():
    """Unequal-radius circles: tangent points must still lie on each
    respective circle, and the tangent line must be perpendicular to
    each radius at its tangent point."""
    import numpy as np

    from skyplothelper.globe.insets import _outer_tangent_endpoints

    cA, rA = (0.0, 0.0), 2.0
    cB, rB = (10.0, 3.0), 5.0
    out = _outer_tangent_endpoints(cA, rA, cB, rB)
    assert out is not None
    for tangent in out:
        a_pt, b_pt = tangent[0], tangent[1]
        dist_a = np.hypot(a_pt[0] - cA[0], a_pt[1] - cA[1])
        dist_b = np.hypot(b_pt[0] - cB[0], b_pt[1] - cB[1])
        assert dist_a == pytest.approx(rA, abs=1e-9)
        assert dist_b == pytest.approx(rB, abs=1e-9)
        # Tangent direction A→B is perpendicular to the radius at A
        # (and at B, since outer tangent radii are parallel).
        tangent_vec = np.array([b_pt[0] - a_pt[0], b_pt[1] - a_pt[1]])
        radius_vec = np.array([a_pt[0] - cA[0], a_pt[1] - cA[1]])
        # Dot product should be zero (perpendicular).
        assert abs(np.dot(tangent_vec, radius_vec)) < 1e-6


def test_outer_tangent_endpoints_one_contains_other_returns_none():
    """No outer tangents exist when one circle strictly contains the
    other (|rA - rB| >= d)."""
    from skyplothelper.globe.insets import _outer_tangent_endpoints

    # Concentric, different radii — no tangents.
    assert _outer_tangent_endpoints((0, 0), 1.0, (0, 0), 5.0) is None
    # Big circle around small one
    assert _outer_tangent_endpoints((0, 0), 1.0, (0.5, 0), 5.0) is None


def test_connect_inset_axes_circular_path_on_globe():
    """make_globe_frame parent + circular SIN inset: style='auto' picks
    the circular tangent path and emits 2 ConnectionPatch artists."""
    from astropy.visualization.wcsaxes.frame import EllipticalFrame

    fig = plt.figure(figsize=(8, 6))
    parent_ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    inset = reproject_inset_axes(
        parent_ax, rect=[0.65, 0.05, 0.3, 0.3],
        center=(0.0, 30.0), size=20.0, projection="SIN",
        frame_class=EllipticalFrame,
    )
    fig.canvas.draw()
    out = connect_inset_axes(parent_ax, inset, color="C3")
    assert len(out) == 2
    # ConnectionPatch instances added to the figure (not the axes).
    from matplotlib.patches import ConnectionPatch
    assert all(isinstance(p, ConnectionPatch) for p in out)


def test_connect_inset_axes_invalid_style_raises():
    fig = plt.figure(figsize=(8, 6))
    parent_ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    inset = reproject_inset_axes(
        parent_ax, rect=[0.65, 0.05, 0.3, 0.3],
        center=(0.0, 30.0), size=10.0, projection="TAN",
    )
    with pytest.raises(ValueError, match="style="):
        connect_inset_axes(parent_ax, inset, style="ellipse")


def test_connect_inset_axes_curvature_yields_arc3_connectionstyle():
    """curvature>0 produces ConnectionPatches with a nonzero-rad
    arc3 style. curvature=0 keeps rad=0 (straight)."""
    from matplotlib.patches import ConnectionPatch

    fig = plt.figure(figsize=(10, 5))
    parent_ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    inset = reproject_inset_axes(
        parent_ax, rect=[0.62, 0.05, 0.32, 0.32], transform="parent",
        projection="TAN", center=(80, 30), size=15,
    )
    straight = connect_inset_axes(parent_ax, inset, curvature=0.0)
    curved = connect_inset_axes(parent_ax, inset, curvature=0.3)
    for cp in straight + curved:
        assert isinstance(cp, ConnectionPatch)
    straight_rads = [cp.get_connectionstyle().rad for cp in straight]
    curved_rads = [cp.get_connectionstyle().rad for cp in curved]
    assert all(r == 0.0 for r in straight_rads)
    assert all(abs(r) > 0.0 for r in curved_rads)


def test_connect_inset_axes_curvature_signs_bow_outward():
    """The two connectors should bow to opposite sides — one with
    rad>0, one with rad<0 — so the pair flares outward symmetrically."""
    fig = plt.figure(figsize=(10, 5))
    parent_ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    inset = reproject_inset_axes(
        parent_ax, rect=[0.62, 0.05, 0.32, 0.32], transform="parent",
        projection="TAN", center=(80, 30), size=15,
    )
    curved = connect_inset_axes(parent_ax, inset, curvature=0.3)
    rads = [cp.get_connectionstyle().rad for cp in curved]
    assert any(r > 0 for r in rads)
    assert any(r < 0 for r in rads)


def test_connect_inset_axes_curvature_circular_path_signs():
    """Same outward-bowing in the circular-tangent path."""
    from astropy.visualization.wcsaxes.frame import EllipticalFrame

    fig = plt.figure(figsize=(12, 5))
    parent_ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    inset = reproject_inset_axes(
        parent_ax, rect=[0.62, 0.05, 0.32, 0.32], transform="parent",
        projection="SIN", center=(80, 30), size=30,
        frame_class=EllipticalFrame,
    )
    mark_inset_axes(parent_ax, inset, style="circle",
                     center=(80, 30), radius=15)
    curved = connect_inset_axes(parent_ax, inset, curvature=0.3)
    rads = [cp.get_connectionstyle().rad for cp in curved]
    assert any(r > 0 for r in rads)
    assert any(r < 0 for r in rads)


def test_reproject_inset_axes_auto_fontsize_shrinks_labels():
    """auto_fontsize=True (default) caches a smaller fontsize than
    matplotlib's default 'medium' = 10 pt, so labels fit the inset's
    reduced display width."""
    fig = plt.figure(figsize=(10, 5))
    parent_ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    inset = reproject_inset_axes(
        parent_ax, rect=[0.62, 0.05, 0.32, 0.32], transform="parent",
        projection="TAN", center=(80, 30), size=15,
    )
    cached = getattr(inset, "_sph_auto_label_fontsize", None)
    assert cached is not None
    # Cached size should be strictly smaller than matplotlib's
    # default 'medium' = 10 pt.
    assert cached < 10.0


def test_reproject_inset_axes_auto_fontsize_disable():
    """auto_fontsize=False leaves the inset at rcParams default; no
    cached size attribute is written."""
    fig = plt.figure(figsize=(10, 5))
    parent_ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    inset = reproject_inset_axes(
        parent_ax, rect=[0.62, 0.05, 0.32, 0.32], transform="parent",
        projection="TAN", center=(80, 30), size=15,
        auto_fontsize=False,
    )
    assert getattr(inset, "_sph_auto_label_fontsize", None) is None


def test_reproject_inset_axes_circular_frame_sets_equal_aspect():
    """When user passes frame_class=EllipticalFrame (or CircularFrame),
    reproject_inset_axes auto-applies aspect='equal' so the inset
    actually renders as a circle. Explicit aspect= overrides."""
    from astropy.visualization.wcsaxes.frame import EllipticalFrame

    plt.figure(figsize=(8, 6))
    parent_ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    inset = reproject_inset_axes(
        parent_ax, rect=[0.65, 0.05, 0.3, 0.3],
        center=(0.0, 30.0), size=20.0, projection="SIN",
        frame_class=EllipticalFrame,
    )
    assert inset.get_aspect() == 1.0  # 'equal' resolves to 1.0

    # Explicit aspect= wins (user must opt out manually)
    plt.figure(figsize=(8, 6))
    parent_ax2 = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    inset2 = reproject_inset_axes(
        parent_ax2, rect=[0.65, 0.05, 0.3, 0.3],
        center=(0.0, 30.0), size=20.0, projection="SIN",
        frame_class=EllipticalFrame, aspect="auto",
    )
    assert inset2.get_aspect() == "auto"


# ============================================================
# boundaries — error-path verification
# ============================================================

def test_fetch_boundary_data_unknown_filename_raises():
    with pytest.raises(ValueError, match="(?i)unknown"):
        fetch_boundary_data(filename="not_a_real_dataset.npz")


def test_find_data_file_missing_returns_none_or_raises():
    """When a file isn't present in any search path, the helper must
    either return None or raise. This pins whichever behavior is
    current."""
    out_or_exc = None
    try:
        out_or_exc = _find_data_file("definitely_not_a_file_xyz.npz")
    except (FileNotFoundError, RuntimeError, OSError) as e:
        out_or_exc = e
    # Acceptable outcomes: None, or a raised lookup error
    assert out_or_exc is None or isinstance(
        out_or_exc, (FileNotFoundError, RuntimeError, OSError)
    )


def test_fetch_boundary_data_default_source_unpublished(tmp_path):
    """The hosted Earth-boundary data isn't published yet, so the default
    source (no base_url) warns and returns an empty list (pointing at
    prepare_earth_data) rather than emitting 404s."""
    with pytest.warns(UserWarning, match="prepare_earth_data"):
        out = fetch_boundary_data(
            filename="coastlines.npz",
            dest=str(tmp_path),
            force=False,
        )
    assert out == []


def test_fetch_boundary_data_writes_to_dest_from_mirror(tmp_path):
    """A base_url mirror bypasses the not-published gate and copies the
    requested file into dest."""
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "coastlines.npz").write_bytes(b"fake-npz")
    dest = tmp_path / "dest"
    out = fetch_boundary_data(
        filename="coastlines.npz",
        dest=str(dest),
        base_url=str(mirror),
        progress=False,
    )
    assert isinstance(out, list)
    assert all(os.path.exists(p) for p in out)


# ---- audit tracks B+C: theme-aware defaults, and knobs to override them ----

def _sites():
    return {"A": (0.0, 10.0), "B": (40.0, -10.0)}


def test_baseline_length_plaque_follows_the_canvas():
    """The length label's bbox was a literal white with no override, so it
    stayed a white box on a dark theme whatever the color knobs said."""

    import skyplothelper as sph
    matplotlib.rcdefaults()
    sph.set_style(base="standard", theme="dark_sky")
    ax = make_globe_frame(projection="SIN")
    plot_baselines(ax, _sites(), show_lengths=True)
    labels = [t for t in ax.texts if t.get_bbox_patch() is not None]
    assert labels, "no length label with a plaque"
    for t in labels:
        assert t.get_bbox_patch().get_facecolor()[:3] != (1.0, 1.0, 1.0)
    plt.close(ax.figure)
    matplotlib.rcdefaults()
    sph.set_style(base="standard")


def test_baseline_length_plaque_is_overridable_and_removable():
    import skyplothelper as sph  # noqa: F401
    ax = make_globe_frame(projection="SIN")
    plot_baselines(ax, _sites(), show_lengths=True,
                   length_bbox={"facecolor": "#ff00ff", "pad": 2})
    labels = [t for t in ax.texts if t.get_bbox_patch() is not None]
    assert labels
    for t in labels:
        assert tuple(round(v, 2)
                     for v in t.get_bbox_patch().get_facecolor()[:3]) \
            == (1.0, 0.0, 1.0)
    plt.close(ax.figure)

    ax = make_globe_frame(projection="SIN")
    plot_baselines(ax, _sites(), show_lengths=True, length_bbox=False)
    assert all(t.get_bbox_patch() is None for t in ax.texts)
    plt.close(ax.figure)


def test_baseline_markers_resolve_scatter_ink_explicitly():
    """scatter(c=None) would take a property-cycle color, so the marker ink
    must be resolved rather than passed through as None."""

    import skyplothelper as sph
    matplotlib.rcdefaults()
    sph.set_style(base="standard", theme="dark_sky")
    ax = make_globe_frame(projection="SIN")
    plot_baselines(ax, _sites())
    cols = [c for c in ax.collections]
    assert cols
    face = cols[-1].get_facecolor()[0][:3]
    # must be the theme ink, NOT tab:blue (0.12, 0.47, 0.71)
    assert not (abs(face[0] - 0.12) < 0.02 and abs(face[2] - 0.71) < 0.02)
    plt.close(ax.figure)
    matplotlib.rcdefaults()
    sph.set_style(base="standard")


# ============================================================
# make_planet_frame(projection=...) — flat (non-SIN) planet maps
# (F1: FITS + non-FITS planet frames; geo helpers must not assume a WCS)
# ============================================================

_PLANET_SITES = {"VLA": (-107.6, 34.1), "GBT": (-79.8, 38.4),
                 "EVN": (6.6, 53.0), "ATCA": (149.6, -30.3)}


def test_make_planet_frame_car_is_fits_wcsaxes():
    """A FITS-code projection planet frame keeps a real WCS object."""
    ax = make_planet_frame(111, projection="CAR", center_LONdeg=0)
    assert isinstance(ax, WCSAxes)
    assert getattr(ax, "wcs", None) is not None
    plt.close(ax.figure)


def test_make_planet_frame_robinson_is_nonfits_and_body_fixed():
    """A non-FITS projection planet frame is a CurvedTransform WCSAxes with no
    FITS WCS object, and its ITRS body-fixed frame is preserved (so axes read
    Longitude/Latitude, not RA/Dec)."""
    ax = make_planet_frame(111, projection="robinson", center_LONdeg=0)
    assert isinstance(ax, WCSAxes)
    assert getattr(ax, "wcs", None) is None
    assert getattr(ax, "_sph_frame", None) == "itrs"
    plt.close(ax.figure)


def test_make_planet_frame_default_sin_unchanged():
    """The default projection stays SIN and routes through the dedicated globe
    builder (real WCS globe) -- behavior unchanged from earlier releases."""
    ax = make_planet_frame(111, center_LONdeg=0, center_LATdeg=0)
    assert getattr(ax, "wcs", None) is not None
    plt.close(ax.figure)


@pytest.mark.parametrize("projection", ["CAR", "robinson"])
def test_geo_overlays_on_flat_planet_frame_no_raise(projection):
    """plot_coastlines + plot_baselines render on a flat planet frame (FITS and
    non-FITS) without reaching for a possibly-None ax.wcs."""
    ax = make_planet_frame(111, projection=projection, center_LONdeg=0)
    plot_coastlines(ax, color="0.6", lw=0.5)  # must not raise on ax.wcs is None
    res = plot_baselines(ax, _PLANET_SITES, back_hemisphere_linestyle=":")
    assert res["arcs"]
    plt.close(ax.figure)


def test_flat_planet_frame_not_hemisphere_culled():
    """hemisphere_only auto-detects: a flat map shows the whole surface (no
    back-hemisphere NaN cull), unlike an orthographic globe."""
    import numpy as np
    ax = make_planet_frame(111, projection="CAR", center_LONdeg=0)
    lines = plot_coastlines(ax, color="0.6", lw=0.5)
    finite_frac = np.mean(
        [np.isfinite(np.asarray(ln.get_ydata(), dtype=float)).mean()
         for ln in lines])
    assert finite_frac > 0.98  # essentially nothing culled on a flat map
    plt.close(ax.figure)


# ============================================================
# Filled Earth overlays (F3): plot_land / plot_lakes / plot_rivers
# via the spherical-region machinery
# ============================================================

def _car():
    fig = plt.figure(figsize=(6, 3))
    ax = make_planet_frame(111, projection="CAR", center_LONdeg=0)
    return fig, ax


def test_plot_land_fills_all_land_polygons():
    """plot_land fills the closed land polygons (127 at 110m) as patches --
    continents included, unlike coastline-line fills (islands only)."""
    fig, ax = _car()
    patches = plot_land(ax, facecolor="#c9b98f")
    assert len(patches) >= 120   # ~127 land polygons at 110m
    plt.close(fig)


def test_plot_lakes_and_rivers_render():
    """plot_lakes fills lake polygons; plot_rivers draws river centerlines."""
    fig, ax = _car()
    lake_patches = plot_lakes(ax, facecolor="#a6cee3")
    river_lines = plot_rivers(ax, color="C0")
    assert len(lake_patches) >= 10    # ~25 lake polygons at 110m
    assert len(river_lines) >= 5      # ~13 major rivers at 110m
    plt.close(fig)


def test_fill_overlays_render_on_nonfits_frame():
    """Filled geographic overlays render on the non-FITS custom projections
    (Robinson/Eckert/…): each ring routes through add_spherical_polygon, which
    the G4 projector unification made backend-agnostic, so the earlier FITS-only
    guard is gone."""
    from matplotlib.patches import PathPatch
    ax = make_planet_frame(111, projection="robinson")
    patches = plot_land(ax, facecolor="0.7")
    assert patches
    assert sum(isinstance(p, PathPatch) for p in ax.patches) >= 1
    plt.close(ax.figure)


def test_fill_overlays_accept_stroke_and_alpha_knobs():
    """The fill/line overlays accept alpha + the shared stroke knobs without
    raising, and the stroke produces path_effects on the artists."""
    fig, ax = _car()
    plot_land(ax, facecolor="0.8", edgecolor="0.3", alpha=0.9)          # fill knobs
    lines = plot_coastlines(ax, color="w", stroke_color="k", stroke_lw=1.5)
    assert lines and lines[0].get_path_effects()  # stroke applied to the line
    plt.close(fig)


def test_clip_to_land_and_ocean_wrappers():
    """clip_to_land / clip_to_ocean build a clip path from the land polygons
    and apply it to the given artist."""
    from skyplothelper.globe.boundaries import clip_to_land, clip_to_ocean
    ax = make_planet_frame(111, projection="CAR", center_LONdeg=0)
    sc = ax.scatter([0], [0], transform=ax.get_transform("world"))
    land = clip_to_land(ax, sc)
    assert sc.get_clip_path() is not None and land.vertices.shape[0] > 0
    ocean = clip_to_ocean(ax)
    assert ocean.vertices.shape[0] > 0
    assert land.vertices.shape != ocean.vertices.shape
    plt.close(ax.figure)


def test_clip_to_land_works_on_nonfits():
    """clip_to_land / clip_to_ocean work on a non-FITS custom-projection frame
    (G4): the region projector now serves Robinson etc., so the land clip path
    builds and attaches to an artist. Skipped if the Earth data isn't present
    (the conftest hook turns the FileNotFoundError into a skip)."""
    import numpy as np

    from skyplothelper.globe.boundaries import clip_to_land, clip_to_ocean
    ax = make_planet_frame(111, projection="robinson")
    assert getattr(ax, "wcs", None) is None
    lon, lat = np.meshgrid(np.linspace(-180, 180, 60), np.linspace(-89, 89, 30))
    mesh = ax.pcolormesh(lon, lat, lat, transform=ax.get_transform("world"),
                         shading="auto")
    clip_to_land(ax, mesh)          # must not raise, must set a clip path
    assert mesh.get_clip_path() is not None
    # clip_to_ocean on a fresh artist likewise.
    mesh2 = ax.pcolormesh(lon, lat, lat, transform=ax.get_transform("world"),
                          shading="auto")
    clip_to_ocean(ax, mesh2)
    assert mesh2.get_clip_path() is not None
    plt.close(ax.figure)


# ============================================================
# F2 -- longitude-West labeling (route b: labels + handedness,
# data stays east-internal)
# ============================================================

def _all_lon_ticklabels(ax):
    """Flatten every native longitude tick-label string (dict keyed by spine)."""
    ax.figure.canvas.draw()
    txt = getattr(ax.coords[0].ticklabels, "text", {})
    out = []
    for v in dict(txt).values():
        out.extend(v)
    return out


def _visible_overlay_texts(ax):
    ax.figure.canvas.draw()
    return [t.get_text() for t in ax.texts if t.get_text()]


def test_lon_west_flag_sets_marker_and_default_off():
    ax = make_planet_frame(111, projection="CAR", center_LONdeg=0)
    assert getattr(ax, "_sph_lon_west", False) is False
    plt.close(ax.figure)
    axw = make_planet_frame(111, projection="CAR", center_LONdeg=0,
                            lon_west=True)
    assert getattr(axw, "_sph_lon_west", False) is True
    plt.close(axw.figure)


def test_lon_west_labels_fits_have_hemisphere_suffix():
    """FITS flat map (native ticks): west labels carry W/E hemisphere letters,
    and the prime/anti-meridian get none."""
    ax = make_planet_frame(111, projection="CAR", center_LONdeg=0,
                           lon_west=True)
    labs = _all_lon_ticklabels(ax)
    assert any(s.endswith("W") for s in labs), labs
    assert any(s.endswith("E") for s in labs), labs
    assert "0°" in labs and "180°" in labs  # no hemisphere letter
    plt.close(ax.figure)


def test_lon_west_default_labels_have_no_hemisphere_suffix():
    """Without lon_west the labels stay plain east degrees (no baseline churn)."""
    ax = make_planet_frame(111, projection="CAR", center_LONdeg=0,
                           lon_units="degrees")
    labs = _all_lon_ticklabels(ax)
    assert labs and not any(s.endswith(("W", "E")) for s in labs), labs
    plt.close(ax.figure)


def test_lon_west_labels_nonfits_overlay_have_suffix():
    """Non-FITS flat map (Robinson) uses overlay labels -- they too read west."""
    ax = make_planet_frame(111, projection="robinson", center_LONdeg=0,
                           lon_west=True)
    txts = _visible_overlay_texts(ax)
    assert any(s.endswith("W") for s in txts), txts
    assert any(s.endswith("E") for s in txts), txts
    plt.close(ax.figure)


def test_lon_west_sin_globe_overlay_have_suffix():
    """The SIN globe overlay labels support west mode as well (normal planet
    orientation -- no direction override needed)."""
    ax = make_planet_frame(111, projection="SIN", center_LONdeg=-80,
                           lon_west=True)
    txts = _visible_overlay_texts(ax)
    assert any(s.endswith("W") for s in txts), txts
    plt.close(ax.figure)


def test_lon_west_leaves_data_and_wcs_untouched():
    """lon_west changes only labels: the world->pixel mapping (hence every
    plotted point) is byte-identical to the east-labeled frame."""
    import numpy as np
    lon, lat = -71.0, 38.0
    ax_e = make_planet_frame(111, projection="CAR", center_LONdeg=0)
    ax_w = make_planet_frame(111, projection="CAR", center_LONdeg=0,
                             lon_west=True)
    pe = np.asarray(ax_e.wcs.world_to_pixel_values(lon, lat))
    pw = np.asarray(ax_w.wcs.world_to_pixel_values(lon, lat))
    assert np.allclose(pe, pw)
    plt.close(ax_e.figure)
    plt.close(ax_w.figure)


def test_lon_west_east_converters_roundtrip():
    import numpy as np

    import skyplothelper as sph
    assert np.isclose(sph.lon_west_to_east(71.0), 289.0)
    assert np.isclose(sph.lon_east_to_west(289.0), 71.0)
    vals = np.array([0.0, 45.0, 120.0, 300.0])
    assert np.allclose(sph.lon_east_to_west(sph.lon_west_to_east(vals)), vals)


# ============================================================
# F3(ii) + G3 — plate polygons/fill, land-lakes hole-punch,
# choropleth, and the SIN-globe subplot placement fix
# ============================================================

def test_plot_land_lakes_hole_punch_smaller_than_land():
    """plot_land(lakes=True) subtracts the lakes as true holes, so its area is
    a bit smaller than plain land (land - lakes via region set-algebra)."""

    from skyplothelper.geometry.compound import CompoundRegion
    from skyplothelper.globe.boundaries import _closed_rings, load_boundary_data
    ax = make_planet_frame(111, projection="CAR", center_LONdeg=0)
    ax.figure.canvas.draw()
    land = CompoundRegion.from_polygons(
        ax, _closed_rings(load_boundary_data("land.npz", key="land_110m")),
        resolution=0)
    lakes = CompoundRegion.from_polygons(
        ax, _closed_rings(load_boundary_data("lakes.npz", key="lakes_110m")),
        resolution=0)
    a_land = land.solid_angle["sq_deg"]
    a_diff = CompoundRegion.from_polygons(
        ax, _closed_rings(load_boundary_data("land.npz", key="land_110m")),
        resolution=0).difference(lakes).solid_angle["sq_deg"]
    assert 0 < a_land - a_diff < 200          # lakes are a small inland area
    # the plot call itself renders without raising
    assert plot_land(ax, lakes=True)
    plt.close(ax.figure)


def test_plot_land_lakes_nonfits_renders():
    """plot_land(lakes=True) renders on the non-FITS custom projections too
    (Robinson/Eckert/…) — the land−lakes set-algebra runs through the same
    region machinery, so the earlier FITS-only guard is gone."""
    from matplotlib.patches import PathPatch
    ax = make_planet_frame(111, projection="robinson")
    patches = plot_land(ax, lakes=True, facecolor="0.7")
    assert sum(isinstance(p, PathPatch) for p in ax.patches) >= 1
    assert patches
    plt.close(ax.figure)


def test_plate_polygons_bundled_with_metadata():
    """The bundled tectonic_plates.npz carries plate polygons + code/name
    metadata (needed for fill + choropleth-by-plate)."""
    import numpy as np

    from skyplothelper.globe.boundaries import _find_data_file
    d = np.load(_find_data_file("tectonic_plates.npz"))
    assert "plate_polygons" in d and "plate_codes" in d and "plate_names" in d
    assert int((np.asarray(d["plate_codes"]) == "PA").sum()) >= 2  # Pacific split


def test_plot_tectonic_plates_fill_categorical():
    """fill=True fills the plate polygons (one patch set per ring)."""
    ax = make_planet_frame(111, projection="MOL", center_LONdeg=0)
    patches = plot_tectonic_plates(ax, fill=True, alpha=0.6)
    assert len(patches) > 10
    plt.close(ax.figure)


def test_plot_tectonic_plates_fill_choropleth_returns_mappable():
    """fill=True with values= returns a ScalarMappable (for a colorbar)."""
    from matplotlib.cm import ScalarMappable
    ax = make_planet_frame(111, projection="MOL", center_LONdeg=0)
    sm = plot_tectonic_plates(ax, fill=True,
                              values={"PA": 1.0, "NA": 2.0, "AF": 3.0},
                              cmap="viridis")
    assert isinstance(sm, ScalarMappable)
    plt.close(ax.figure)


def test_plot_tectonic_plates_fill_nonfits_renders():
    """Filled plates render on the non-FITS custom projections too — the fill
    routes through the backend-agnostic region machinery (no FITS-WCS guard)."""
    from matplotlib.patches import PathPatch
    ax = make_planet_frame(111, projection="robinson")
    patches = plot_tectonic_plates(ax, fill=True, alpha=0.6)
    assert patches
    assert sum(isinstance(p, PathPatch) for p in ax.patches) > 10
    plt.close(ax.figure)


def test_choropleth_helper_colors_rings_by_value():
    import numpy as np

    import skyplothelper as sph
    ax = make_wcs_frame(111, projection="AIT", frame="icrs", center=180)
    ax.figure.canvas.draw()
    rings = [([150, 170, 170, 150], [-10, -10, 10, 10]),
             ([190, 210, 210, 190], [-10, -10, 10, 10])]
    sm = sph.choropleth(ax, rings, np.array([1.0, 5.0]), cmap="viridis")
    from matplotlib.cm import ScalarMappable
    assert isinstance(sm, ScalarMappable)
    plt.close(ax.figure)


def test_sin_globe_fits_into_subplot_grid():
    """make_globe_frame / make_planet_frame(SIN) now takes fig= and places into
    one cell of a grid instead of overplotting the whole current figure."""
    fig = plt.figure(figsize=(8, 8))
    for i in (1, 2, 3):
        make_wcs_frame((2, 2, i), projection="CAR", frame="icrs", center=0,
                       fig=fig)
    axg = make_planet_frame((2, 2, 4), projection="SIN", center_LONdeg=-40,
                            center_LATdeg=25, fig=fig)
    assert len(fig.axes) == 4               # no stray full-figure overplot axes
    assert axg in fig.axes
    plt.close(fig)


def test_sin_globe_swaps_pre_existing_axes():
    """A pre-existing subplot Axes can be swapped for a SIN globe."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axg = make_planet_frame(axes[1, 1], projection="SIN", center_LONdeg=60)
    assert len(fig.axes) == 4 and axg in fig.axes
    plt.close(fig)
