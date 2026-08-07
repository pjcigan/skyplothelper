"""Tests for skyplothelper.globe.frame."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.visualization.wcsaxes import WCSAxes
from astropy.wcs import WCS

from skyplothelper.globe.frame import (
    TiltedEarthFrame,
    euler_to_fits_ortho,
    make_globe_angles,
    make_globe_frame,
    quaternion_to_fits_ortho,
)


def _axis_quat(angle_deg, axis):
    """Unit quaternion (w, x, y, z) for a rotation about a cardinal axis."""
    h = np.radians(angle_deg) / 2.0
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([np.cos(h), *(np.sin(h) * np.array(v))])


def _qmul(q1, q2):
    """Hamilton product of two (w, x, y, z) quaternions (no scipy)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _euler_quat(rotation, obliquity, perspective):
    """The (w, x, y, z) quaternion encoding the same orientation as the
    ZXZ euler angles euler_to_fits_ortho consumes (built from axis-angle, so
    the test has no scipy dependency)."""
    return _qmul(_qmul(_axis_quat(-rotation, "z"), _axis_quat(-obliquity, "x")),
                 _axis_quat(-perspective, "z"))


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_tilted_earth_frame_class_exists():
    assert TiltedEarthFrame.__name__ == "TiltedEarthFrame"


def test_euler_to_fits_ortho_runs():
    """Should produce (center_lon, center_lat, lonpole) for a rotation state."""
    out = euler_to_fits_ortho(rotation=0.0, obliquity=23.44, perspective=45.0)
    # The function returns center_lon, center_lat, lonpole
    assert len(out) == 3
    for v in out:
        assert np.isfinite(float(v))


def test_quaternion_identity_is_equatorial():
    """The identity quaternion → the untilted equatorial view (0, 0, 0)."""
    out = quaternion_to_fits_ortho([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(out, [0.0, 0.0, 0.0], atol=1e-9)


def test_quaternion_matches_euler_for_same_orientation():
    """quaternion_to_fits_ortho agrees with euler_to_fits_ortho when the
    quaternion encodes the same orientation."""
    rot, obl, persp = 60.0, 23.44, 10.0
    q = _euler_quat(rot, obl, persp)                       # (w, x, y, z)
    assert np.allclose(quaternion_to_fits_ortho(q),
                       euler_to_fits_ortho(rot, obl, persp), atol=1e-9)


def test_quaternion_scalar_first_vs_last():
    """scalar_first=False reads (x, y, z, w); same orientation → same result."""
    q_wxyz = _euler_quat(40.0, 50.0, -15.0)
    q_xyzw = np.array([*q_wxyz[1:], q_wxyz[0]])            # roll w to the end
    assert np.allclose(
        quaternion_to_fits_ortho(q_wxyz, scalar_first=True),
        quaternion_to_fits_ortho(q_xyzw, scalar_first=False), atol=1e-12)


def test_quaternion_batch_and_normalization():
    """An (N, 4) batch returns 3 length-N arrays; non-unit quaternions are
    normalized internally."""
    qs = np.array([_euler_quat(a, 20.0, 5.0) for a in (0.0, 30.0, 90.0)])
    lons, lats, poles = quaternion_to_fits_ortho(qs)
    assert lons.shape == lats.shape == poles.shape == (3,)
    # scaling a quaternion doesn't change the rotation it represents
    assert np.allclose(quaternion_to_fits_ortho(qs[1] * 3.7),
                       quaternion_to_fits_ortho(qs[1]), atol=1e-9)


def test_quaternion_rejects_bad_shape():
    with pytest.raises(ValueError, match="4 components"):
        quaternion_to_fits_ortho([1.0, 0.0, 0.0])


def test_quaternion_scipy_bridge_matches_euler():
    """Locks the documented SciPy bridge: because euler_to_fits_ortho uses
    passive matrices, a SciPy (active) ZXZ quaternion of the NEGATED angles
    matches it. Skipped when SciPy (an optional dep) is unavailable."""
    Rotation = pytest.importorskip(
        "scipy.spatial.transform", reason="scipy optional").Rotation
    a, b, c = 60.0, 23.44, 10.0
    q = Rotation.from_euler("ZXZ", [-a, -b, -c], degrees=True).as_quat()  # x,y,z,w
    assert np.allclose(quaternion_to_fits_ortho(q, scalar_first=False),
                       euler_to_fits_ortho(a, b, c), atol=1e-6)


def test_make_globe_angles_returns_array():
    """Generates a sequence of (rotation, obliquity, perspective) angles."""
    angles = make_globe_angles(
        start_angles_deg=(0.0, 23.44, 45.0),
        n_steps=10,
    )
    angles = np.asarray(angles)
    # Output is (n_steps, 3) — rotation/obliquity/perspective per step
    assert angles.shape[0] == 10 or angles.shape[1] == 10


def test_make_globe_angles_totals_match_rates():
    """spin_total / prec_total / nut_cycles are convenience aliases: they must
    reproduce the equivalent per-step rate exactly."""
    n = 40
    by_total = make_globe_angles((10., 24., 45.), n, spin_total=360.,
                                 prec_total=80., nut_cycles=2., nut_amp=5.,
                                 out_system="euler")
    by_rate = make_globe_angles((10., 24., 45.), n, spin_rate=360. / n,
                                prec_rate=80. / n, nut_rate=360. * 2. / n,
                                nut_amp=5., out_system="euler")
    for a, b in zip(by_total, by_rate):
        np.testing.assert_allclose(a, b)
    # A full 360 spin is endpoint-exclusive so the loop closes seamlessly:
    # the last frame is one step short of a full turn back to the start.
    phis = np.asarray(by_total[0])
    assert np.isclose(phis[-1] - phis[0], 360. * (n - 1) / n)


def test_make_globe_angles_total_and_rate_conflict():
    """Passing both a *_rate and its *_total (or nut_rate + nut_cycles) is
    ambiguous and rejected."""
    with pytest.raises(ValueError):
        make_globe_angles((0., 24., 45.), 10, spin_rate=5., spin_total=360.)
    with pytest.raises(ValueError):
        make_globe_angles((0., 24., 45.), 10, prec_rate=2., prec_total=90.)
    with pytest.raises(ValueError):
        make_globe_angles((0., 24., 45.), 10, nut_rate=30., nut_cycles=1.)


def test_make_globe_frame_basic():
    plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    assert isinstance(ax, WCSAxes)


# ---- direction (longitude orientation) ----

def _globe_dirs(**kw):
    """Return (east_side, north_side) on screen for a centered globe."""
    res = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0,
                           return_header=True, **kw)
    hdr = res[-1] if isinstance(res, (tuple, list)) else res
    w = WCS(hdr)
    cx, cy = w.world_to_pixel_values(0, 0)
    xe, _ = w.world_to_pixel_values(30, 0)
    _, yn = w.world_to_pixel_values(0, 30)
    return ("RIGHT" if xe > cx else "LEFT", "UP" if yn > cy else "DOWN")


def test_make_globe_frame_default_is_sky():
    """The globe defaults to the astronomical looking-out view (east to the
    left, north up), consistent with the other frames."""
    assert _globe_dirs() == ("LEFT", "UP")


def test_make_globe_frame_geographic_for_earth():
    """direction='geographic' gives the from-outside Earth view (east right),
    north up — what the Earth helpers / ITRS globes want."""
    assert _globe_dirs(direction="geographic") == ("RIGHT", "UP")


def test_make_globe_frame_direction_aliases():
    assert _globe_dirs(direction="geo") == ("RIGHT", "UP")
    assert _globe_dirs(direction="astro") == ("LEFT", "UP")


def _planet_dirs(**kw):
    from skyplothelper.globe.frame import make_planet_frame
    res = make_planet_frame(111, center_LONdeg=0, center_LATdeg=0,
                            return_header=True, **kw)
    hdr = res[-1] if isinstance(res, (tuple, list)) else res
    w = WCS(hdr)
    cx, cy = w.world_to_pixel_values(0, 0)
    xe, _ = w.world_to_pixel_values(30, 0)
    _, yn = w.world_to_pixel_values(0, 30)
    return ("RIGHT" if xe > cx else "LEFT", "UP" if yn > cy else "DOWN",
            hdr["CTYPE1"])


def test_make_planet_frame_defaults_geographic_bodyfixed():
    """Earth/planet globes default to geographic (east-right) + a body-fixed
    TLON/TLAT frame, unlike the sky-default make_globe_frame."""
    side, vert, ctype = _planet_dirs()
    assert (side, vert) == ("RIGHT", "UP")
    assert ctype.startswith("TLON")


def test_make_planet_frame_nonearth_body_is_generic_bodyfixed():
    """A non-Earth body still gets the generic body-fixed lon/lat frame."""
    _, _, ctype = _planet_dirs(body="mars")
    assert ctype.startswith("TLON")


def test_make_planet_frame_direction_override():
    """direction= still overrides the geographic default."""
    assert _planet_dirs(direction="sky")[0] == "LEFT"


def _globe_lon_unit(**kw):
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0, **kw)
    return str(ax.coords[0].get_format_unit())


def test_make_globe_frame_lon_units_override():
    """lon_units forces longitude units regardless of frame/direction."""
    assert _globe_lon_unit() == "hourangle"                    # sky ICRS auto
    assert _globe_lon_unit(lon_units="degrees") == "deg"
    assert _globe_lon_unit(direction="geo") == "deg"           # geo auto
    assert _globe_lon_unit(direction="geo", lon_units="hours") == "hourangle"


def test_make_globe_frame_accepts_tuple_and_subplotspec():
    """make_globe_frame / make_planet_frame accept int, (nrows, ncols, index)
    tuples (grids beyond 9 panels), and SubplotSpec — parity with
    make_wcs_frame."""
    import matplotlib.gridspec as gridspec

    from skyplothelper.globe.frame import make_planet_frame

    fig = plt.figure(figsize=(12, 9))
    ax_int = make_globe_frame(111)
    assert isinstance(ax_int, WCSAxes)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 9))
    ax_tuple = make_globe_frame((3, 4, 11), center_LONdeg=0)   # panel 11 of 12
    assert isinstance(ax_tuple, WCSAxes)
    gs = gridspec.GridSpec(3, 4, figure=fig)
    ax_spec = make_globe_frame(gs[2, 3], center_LONdeg=0)
    assert isinstance(ax_spec, WCSAxes)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 9))
    ax_planet = make_planet_frame((3, 4, 12))
    assert isinstance(ax_planet, WCSAxes)
    plt.close(fig)


# ---- grid-spacing kwarg unification (1.2.1) ----

import warnings  # noqa: E402

import skyplothelper as sph  # noqa: E402


def _n_lon_labels(ax):
    ax.figure.canvas.draw()
    txt = ax.coords[0].ticklabels.text
    return len([s for v in txt.values() for s in v])


def _spacing_deprecations(fn):
    """Run *fn*; return only OUR spacing DeprecationWarnings (ignoring unrelated
    library warnings like mpl's PyparsingDeprecationWarning during draw)."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        result = fn()
    return result, [x for x in rec if "is deprecated; use" in str(x.message)]


def test_globe_canonical_lon_spacing_no_warning():
    ax, dep = _spacing_deprecations(
        lambda: make_globe_frame(center_LONdeg=0, lon_spacing=15, lat_spacing=15))
    assert not dep
    # 15-deg spacing -> more longitude labels than the 30-deg default
    assert _n_lon_labels(ax) > _n_lon_labels(make_globe_frame(center_LONdeg=0))
    plt.close("all")


def test_globe_deg_spacing_alias_warns_but_works():
    with pytest.warns(DeprecationWarning, match="lon_deg_spacing"):
        ax = make_globe_frame(center_LONdeg=0, lon_deg_spacing=15)
    assert _n_lon_labels(ax) > 8
    plt.close("all")


def test_globe_default_spacing_unchanged_no_warning():
    _, dep = _spacing_deprecations(lambda: make_globe_frame(center_LONdeg=0))
    assert not dep
    plt.close("all")


@pytest.mark.parametrize("projection", ["SIN", "CAR"])
def test_planet_frame_lon_spacing_one_name_all_projections(projection):
    """The same lon_spacing name works whether make_planet_frame routes to the
    globe (SIN) or the flat (CAR) builder — the 1.2.1 fix."""
    def build(sp):
        # a fresh figure per build so the second doesn't replace the first at 111
        return sph.make_planet_frame(111, fig=plt.figure(), projection=projection,
                                     center_LONdeg=0, lon_spacing=sp, lat_spacing=sp)
    ax_tight, dep = _spacing_deprecations(lambda: build(15))
    assert not dep                              # canonical name -> no deprecation
    tight = _n_lon_labels(ax_tight)
    assert tight > _n_lon_labels(build(45))     # 15 deg -> more lines than 45 deg
    plt.close("all")
