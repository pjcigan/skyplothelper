"""Tests for the sph.scatter / plot / text ... convenience wrappers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402

import skyplothelper as sph  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _allsky(center=180, frame="ICRS"):
    fig, ax = sph.allsky_figure(projection="AIT", center=center, frame=frame)
    fig.canvas.draw()
    return fig, ax


def _drawn_gap(line):
    """Largest display-space gap between points matplotlib actually CONNECTS.

    Pairs separated by a NaN are not drawn, so they are skipped rather than
    filtered out — filtering would measure straight across the break and hide
    exactly the thing under test.
    """
    xy = np.asarray(line.get_xydata(), dtype=float)
    disp = line.get_transform().transform(xy)
    worst = 0.0
    for i in range(len(disp) - 1):
        p, q = disp[i], disp[i + 1]
        if np.isfinite(p).all() and np.isfinite(q).all():
            worst = max(worst, abs(q[0] - p[0]))
    return worst


# ---- the world transform is applied for you ----

def test_wrappers_apply_world_transform():
    fig, ax = _allsky()
    art = sph.scatter(ax, [100.0, 200.0], [10.0, -10.0], s=8)
    assert art.get_transform() is not ax.transData


def test_world_transform_helper_rejects_plain_axes():
    fig = plt.figure()
    with pytest.raises(TypeError):
        sph.world_transform(fig.add_subplot(111))


# ---- SkyCoord input, converted into the axes frame ----

def test_scatter_accepts_skycoord_and_converts_frame():
    fig, ax = _allsky()
    gal = SkyCoord([120.0, 130.0], [30.0, 35.0], unit="deg", frame="galactic")
    off = np.asarray(sph.scatter(ax, gal, s=8).get_offsets())
    assert np.allclose(off[:, 0], gal.icrs.ra.deg)


def test_scatter_arrays_and_skycoord_agree():
    fig, ax = _allsky()
    sc = SkyCoord([100.0, 120.0], [10.0, 20.0], unit="deg")
    a = np.asarray(sph.scatter(ax, sc).get_offsets())
    b = np.asarray(sph.scatter(ax, [100.0, 120.0], [10.0, 20.0]).get_offsets())
    assert np.allclose(a, b)


# ---- the antimeridian seam ----

@pytest.mark.parametrize("center,lons", [
    (0, np.linspace(170, 190, 9)),        # crosses the seam at 180
    (180, np.linspace(350, 370, 9) % 360),  # crosses the seam at 0
])
def test_plot_splits_at_seam(center, lons):
    """A path across the wrap edge must not streak across the whole map.

    Regression for a subtle failure: deriving the crossing analytically with
    ``((lon - center + 180) % 360) - 180`` puts lon=180 at MINUS 180 while the
    projection draws it at PLUS 180, so a center=0 path stepping across exactly
    180 got its break one position off and still streaked. Detection happens in
    display space instead, which is projection- and convention-independent.
    """
    fig, ax = _allsky(center=center)
    lats = np.zeros_like(lons)
    raw = ax.plot(lons, lats, transform=ax.get_transform("world"))[0]
    wrapped = sph.plot(ax, lons, lats)[0]
    fig.canvas.draw()
    width = float(ax.get_window_extent().width)
    assert _drawn_gap(raw) > 0.5 * width          # raw really does streak
    assert _drawn_gap(wrapped) < 0.1 * width      # wrapper does not


def test_plot_wrap_false_is_the_escape_hatch():
    fig, ax = _allsky(center=0)
    lons = np.linspace(170, 190, 9)
    lats = np.zeros_like(lons)
    fig.canvas.draw()
    width = float(ax.get_window_extent().width)
    assert _drawn_gap(sph.plot(ax, lons, lats, wrap=False)[0]) > 0.5 * width


def test_plot_leaves_non_crossing_paths_untouched():
    fig, ax = _allsky(center=0)
    lons = np.linspace(100.0, 140.0, 9)
    lats = np.zeros_like(lons)
    a = sph.plot(ax, lons, lats)[0]
    b = ax.plot(lons, lats, transform=ax.get_transform("world"))[0]
    assert np.array_equal(np.asarray(a.get_xydata()),
                          np.asarray(b.get_xydata()))


# ---- text / annotate position spellings ----

def test_text_and_annotate_position_spellings():
    fig, ax = _allsky()
    coord = SkyCoord(110.0, 20.0, unit="deg")
    assert sph.text(ax, 110.0, 20.0, "a") is not None
    assert sph.text(ax, (110.0, 20.0), "b") is not None
    assert sph.text(ax, coord, "c") is not None
    assert sph.annotate(ax, "d", 110.0, 20.0) is not None
    assert sph.annotate(ax, "e", (110.0, 20.0)) is not None
    assert sph.annotate(ax, "f", coord) is not None


def test_annotate_uses_xycoords_not_transform():
    """matplotlib's annotate takes the transform via xycoords; passing it as
    transform= silently mis-places the label. The wrapper hides that."""
    fig, ax = _allsky()
    ann = sph.annotate(ax, "M31", (110.0, 20.0))
    assert ann.xycoords is not None
    assert ann.xycoords is not ax.transData


# ---- ax.sky_* delegating methods ----

def test_sky_methods_attached_and_delegate():
    fig, ax = _allsky()
    for verb in ("scatter", "plot", "text", "annotate", "contour", "hist2d"):
        assert hasattr(ax, f"sky_{verb}"), verb
    a = np.asarray(ax.sky_scatter([100.0], [10.0]).get_offsets())
    b = np.asarray(sph.scatter(ax, [100.0], [10.0]).get_offsets())
    assert np.allclose(a, b)


def test_sky_methods_not_patched_onto_foreign_axes():
    """Attached only to axes sph builds — never monkeypatched onto WCSAxes
    globally, which would leak sph behavior into every astropy user's axes."""
    fig = plt.figure()
    assert not hasattr(fig.add_subplot(111), "sky_scatter")


# ---- mesh / field verbs ----

def test_mesh_verbs_draw():
    fig, ax = _allsky()
    lo, la = np.meshgrid(np.linspace(60, 120, 12), np.linspace(-20, 20, 10))
    v = np.hypot(lo - 90, la)
    assert sph.contour(ax, lo, la, v, levels=4) is not None
    assert sph.contourf(ax, lo, la, v, levels=4) is not None
    assert sph.pcolormesh(ax, lo, la, v) is not None


def test_hist2d_draws():
    fig, ax = _allsky()
    rng = np.random.default_rng(0)
    assert sph.hist2d(ax, rng.uniform(60, 120, 200),
                      rng.uniform(-20, 20, 200), bins=6) is not None


# ---- misuse raises rather than misbinding ----

def test_mixed_skycoord_and_arrays_raises():
    fig, ax = _allsky()
    with pytest.raises(TypeError):
        sph.scatter(ax, SkyCoord([1.0], [2.0], unit="deg"), [3.0])


def test_fill_between_rejects_skycoord():
    """fill_between needs two latitude bounds, which a SkyCoord can't carry."""
    fig, ax = _allsky()
    with pytest.raises(TypeError):
        sph.fill_between(ax, SkyCoord([1.0], [2.0], unit="deg"), 0.0, 1.0)


# ---- sph.to_lonlat ----

def test_to_lonlat_preserves_frame_by_default():
    """A bare call must NOT silently convert. Galactic in, galactic out."""
    c = SkyCoord(l=120.0, b=30.0, unit="deg", frame="galactic")
    lon, lat = sph.to_lonlat(c)
    assert lon == pytest.approx(120.0)
    assert lat == pytest.approx(30.0)


def test_to_lonlat_scalar_returns_floats():
    lon, lat = sph.to_lonlat(SkyCoord(83.6, 22.0, unit="deg"))
    assert isinstance(lon, float) and isinstance(lat, float)


def test_to_lonlat_array_returns_arrays():
    c = SkyCoord([10.0, 20.0], [30.0, 40.0], unit="deg")
    lon, lat = sph.to_lonlat(c)
    assert lon.shape == (2,) and lat.shape == (2,)
    assert lon == pytest.approx([10.0, 20.0])


def test_to_lonlat_frame_converts():
    c = SkyCoord(l=0.0, b=0.0, unit="deg", frame="galactic")
    lon, lat = sph.to_lonlat(c, frame="icrs")
    # the galactic center in ICRS
    assert lon == pytest.approx(266.4, abs=0.1)
    assert lat == pytest.approx(-28.9, abs=0.1)


def test_to_lonlat_ax_uses_axes_frame():
    """ax= answers 'what numbers do I pass to this plot?'."""
    fig, ax = _allsky(frame="galactic")
    c = SkyCoord(266.405, -28.936, unit="deg")  # galactic center in ICRS
    lon, lat = sph.to_lonlat(c, ax=ax)
    assert lon % 360 == pytest.approx(0.0, abs=0.1) or \
        lon % 360 == pytest.approx(360.0, abs=0.1)
    assert lat == pytest.approx(0.0, abs=0.1)


def test_to_lonlat_frame_beats_ax():
    fig, ax = _allsky(frame="galactic")
    c = SkyCoord(l=0.0, b=0.0, unit="deg", frame="galactic")
    lon, lat = sph.to_lonlat(c, frame="icrs", ax=ax)
    assert lon == pytest.approx(266.4, abs=0.1)


def test_to_lonlat_passes_degrees_through():
    lon, lat = sph.to_lonlat([10.0, 20.0], [30.0, 40.0])
    assert lon == pytest.approx([10.0, 20.0])
    assert lat == pytest.approx([30.0, 40.0])


def test_to_lonlat_degrees_ignore_frame():
    """Plain numbers are taken at face value, not reinterpreted."""
    lon, lat = sph.to_lonlat([10.0], [30.0], frame="galactic")
    assert lon == pytest.approx([10.0])


def test_to_lonlat_misuse_raises():
    with pytest.raises(TypeError):
        sph.to_lonlat(SkyCoord([1.0], [2.0], unit="deg"), [3.0])
    with pytest.raises(TypeError):
        sph.to_lonlat([1.0, 2.0])


# ---- view / limits API (set_extent, set_xlim/ylim, zoom_to, set_view) ----

from skyplothelper.plotting import _short_lon_pair  # noqa: E402


def _car(center_lon=0.0, center_lat=0.0, npix=720):
    """A flat plate-carree planet frame (FITS CAR) with a real WCS."""
    ax = sph.make_planet_frame(111, body="earth", projection="CAR",
                               center_LONdeg=center_lon, center_LATdeg=center_lat,
                               npix=npix, grid=False)
    ax.figure.canvas.draw()
    return ax


def _pixel_of(ax, lon, lat):
    """World (deg) -> pixel/data coords via the standard WCSAxes world path.

    Independent of the implementation under test: the documented idiom is the
    'world' transform (world->display) composed with transData^-1 (display->
    pixel). Used to check that a point lands inside/outside the view window.
    """
    disp = ax.get_transform("world").transform([[lon, lat]])
    return ax.transData.inverted().transform(disp)[0]


def _in_view(ax, lon, lat):
    x, y = _pixel_of(ax, lon, lat)
    (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
    return x0 <= x <= x1 and y0 <= y <= y1


def test_short_lon_pair_takes_the_shorter_arc():
    # a 20-deg box straddling 180 must sweep 20 deg, not 340
    a, b = _short_lon_pair(170.0, 190.0)
    assert b - a == pytest.approx(20.0)
    a, b = _short_lon_pair(170.0, -170.0)   # -170 == 190
    assert b - a == pytest.approx(20.0)
    # west-longitude box [160 W, 60 W] -> east [200, 300], a 100-deg span
    a, b = _short_lon_pair(200.0, 300.0)
    assert b - a == pytest.approx(100.0)


def test_set_extent_frames_the_box_on_car():
    ax = _car()
    box = sph.set_extent(ax, [-40, 20, -10, 30])
    assert len(box) == 4
    assert ax.get_xlim()[0] < ax.get_xlim()[1]
    assert ax.get_ylim()[0] < ax.get_ylim()[1]
    # a point inside the box is in view; a far-away point is not
    assert _in_view(ax, -10, 10)
    assert not _in_view(ax, 150, -60)
    # the view is a genuine crop, not the whole 720-px frame
    assert (ax.get_xlim()[1] - ax.get_xlim()[0]) < 700


def test_set_extent_lon_west_matches_east():
    box_e = sph.set_extent(_car(), [-125, -66, 24, 50])
    box_w = sph.set_extent(_car(), [125, 66, 24, 50], lon_west=True)
    assert np.allclose(box_e, box_w, atol=1.0)


def test_set_extent_pad_widens_the_window():
    tight = sph.set_extent(_car(), [-20, 20, -10, 10])
    padded = sph.set_extent(_car(), [-20, 20, -10, 10], pad=8)
    assert (padded[1] - padded[0]) > (tight[1] - tight[0])
    assert (padded[3] - padded[2]) > (tight[3] - tight[2])


def test_set_xlim_set_ylim_exact_on_car():
    ax = _car()
    x0, x1 = sph.set_xlim(ax, -30, 30)
    # on CAR the meridians -30/+30 map exactly to the x-limits (to <1 px)
    px = sorted(_pixel_of(ax, lon, 0)[0] for lon in (-30, 30))
    assert min(x0, x1) == pytest.approx(px[0], abs=1.0)
    assert max(x0, x1) == pytest.approx(px[1], abs=1.0)
    y0, y1 = sph.set_ylim(ax, -15, 45)
    py = sorted(_pixel_of(ax, 0, lat)[1] for lat in (-15, 45))
    assert min(y0, y1) == pytest.approx(py[0], abs=1.0)
    assert max(y0, y1) == pytest.approx(py[1], abs=1.0)


def test_zoom_to_frames_all_points():
    ax = _car()
    lon = np.array([-120.0, -100.0, -80.0])
    lat = np.array([20.0, 35.0, 45.0])
    sph.zoom_to(ax, lon, lat, pad=3)
    for lo, la in zip(lon, lat):
        assert _in_view(ax, lo, la)
    # a point well outside the cluster is excluded
    assert not _in_view(ax, 60, -40)


def test_zoom_to_accepts_skycoord():
    # a SkyCoord in another frame is converted into the axes frame first
    ax = sph.make_wcs_frame(111, projection="MOL", center=0, frame="icrs")
    ax.figure.canvas.draw()
    sc = SkyCoord([120.0, 130.0], [20.0, 30.0], unit="deg", frame="galactic")
    sph.zoom_to(ax, sc, pad=2)
    assert ax.get_xlim()[0] < ax.get_xlim()[1]
    # the framed points (in ICRS) are in view
    assert _in_view(ax, sc.icrs.ra.deg[0], sc.icrs.dec.deg[0])


def test_set_view_centers_on_target():
    ax = _car()
    sph.set_view(ax, (-100, 30), (40, 20))
    assert _in_view(ax, -100, 30)          # the center is in view
    assert _in_view(ax, -100 + 15, 30 + 8)  # inside the half-fov
    assert not _in_view(ax, -100 + 40, 30)  # outside the half-fov


def test_set_extent_frame_conversion_shifts_the_window():
    """Passing frame='galactic' must interpret the box in galactic and land it
    at the corresponding place on an ICRS map (not the same pixels as ICRS)."""
    ax_icrs = sph.make_wcs_frame(111, projection="MOL", center=0, frame="icrs")
    ax_icrs.figure.canvas.draw()
    same = sph.set_extent(ax_icrs, [-10, 10, -5, 5])
    ax_icrs2 = sph.make_wcs_frame(111, projection="MOL", center=0, frame="icrs")
    ax_icrs2.figure.canvas.draw()
    gal = sph.set_extent(ax_icrs2, [-10, 10, -5, 5], frame="galactic")
    assert not np.allclose(same, gal, atol=5.0)


def test_set_extent_on_globe_drops_offlimb_and_errors_when_all_hidden():
    ax = sph.make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    ax.figure.canvas.draw()
    # a near-side box gives a finite crop
    box = sph.set_extent(ax, [-30, 30, -20, 20])
    assert all(np.isfinite(box))
    # a box entirely on the FAR side (around the antimeridian) has no finite
    # pixels -> clear error rather than a garbage window
    with pytest.raises(ValueError):
        sph.set_extent(ax, [178, 182, -2, 2])


def test_view_methods_are_attached():
    ax = _car()
    assert hasattr(ax, "sky_set_extent")
    ax.sky_set_extent([-40, 20, -10, 30])
    assert _in_view(ax, -10, 10)
