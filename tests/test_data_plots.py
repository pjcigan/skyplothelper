"""Smoke tests for skyplothelper.data_plots."""

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from skyplothelper.data_plots import (
    _wrap_break_lonlat,
    plot_displacement,
    plot_sky_vectors,
)
from skyplothelper.wcs_frame import make_wcs_frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_plot_sky_vectors_smoke():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(180.0, 0.0), fig=fig)
    plot_sky_vectors(
        ax,
        lon=np.array([180.0]),
        lat=np.array([0.0]),
        dlon=np.array([1.0]),
        dlat=np.array([1.0]),
        units="arcsec",
    )


def test_plot_displacement_smoke():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(180.0, 0.0), fig=fig)
    plot_displacement(
        ax,
        lon1=np.array([180.0]),
        lat1=np.array([0.0]),
        lon2=np.array([180.001]),
        lat2=np.array([0.001]),
    )


def test_wrap_break_inserts_nan_at_seam():
    """A polyline straddling the wrap meridian gets a NaN break; one that
    stays on one side is returned unbroken."""
    # 5° → 355° the short way (through the 0/360 seam) at center=180.
    lon = np.array([5.0, 3.0, 1.0, -1.0, -3.0, -5.0])
    lat = np.zeros_like(lon)
    wl, _ = _wrap_break_lonlat(lon, lat, center=180.0)
    assert np.isnan(wl).any()

    lon2 = np.array([100.0, 110.0, 120.0])
    wl2, _ = _wrap_break_lonlat(lon2, np.zeros(3), center=180.0)
    assert not np.isnan(wl2).any()


def test_plot_displacement_geodesic_returns_shaft_and_head():
    """Geodesic mode draws a shaft Line2D plus an arrowhead annotation."""
    from matplotlib.lines import Line2D
    from matplotlib.text import Annotation
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    artists = plot_displacement(ax, 100.0, 10.0, 110.0, 15.0)
    assert any(isinstance(a, Line2D) for a in artists)
    assert any(isinstance(a, Annotation) for a in artists)


def test_plot_displacement_geodesic_false_is_legacy_annotate():
    """The opt-out draws a single annotate arrow per source (no shaft)."""
    from matplotlib.lines import Line2D
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    artists = plot_displacement(ax, 100.0, 10.0, 110.0, 15.0, geodesic=False)
    assert len(artists) == 1
    assert not any(isinstance(a, Line2D) for a in artists)


def test_plot_displacement_seam_crossing_shaft_is_broken():
    """A source on the wrap seam (lon=0 at center=180) displaced across it
    yields a shaft whose wrapped path carries a NaN break — so it draws off
    one frame edge and onto the other instead of streaking across."""
    from matplotlib.lines import Line2D
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    artists = plot_displacement(ax, 0.0, 0.0, 358.0, 0.0)
    shafts = [a for a in artists if isinstance(a, Line2D)]
    assert shafts
    assert np.isnan(np.asarray(shafts[0].get_xdata(), dtype=float)).any()


def test_plot_catalog_arrays_positional_teaching_error():
    """plot_catalog(ax, ra, dec) — the common misuse — raises a clear TypeError
    (dec binds to ra_col=), not a cryptic AttributeError."""
    import skyplothelper as sph
    ax = make_wcs_frame(111, projection="AIT", center=180)
    ra = np.array([180.0, 181.0])
    dec = np.array([10.0, 11.0])
    with pytest.raises(TypeError, match="column NAME"):
        sph.plot_catalog(ax, ra, dec)


def test_plot_catalog_bare_array_teaching_error():
    """plot_catalog(ax, single_ndarray) raises a clear TypeError about the
    catalog type, not a KeyError from column auto-detection."""
    import skyplothelper as sph
    ax = make_wcs_frame(111, projection="AIT", center=180)
    with pytest.raises(TypeError, match="named columns"):
        sph.plot_catalog(ax, np.array([180.0, 181.0]))


def test_plot_catalog_autodetects_de_icrs():
    """VizieR's Gaia dec column DE_ICRS auto-detects (case-insensitive,
    shared registry with skyplothelper.catalog)."""
    from astropy.table import Table

    import skyplothelper as sph
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    tbl = Table({"RA_ICRS": [180.0, 181.0], "DE_ICRS": [10.0, 11.0],
                 "Gmag": [15.0, 16.0]})
    sph.plot_catalog(ax, tbl)                 # must not KeyError
    # lowercase spelling resolves via the case-insensitive match too
    tbl2 = Table({"ra_icrs": [180.0], "de_icrs": [10.0]})
    sph.plot_catalog(ax, tbl2)


def test_plot_catalog_galactic_lb_still_autodetects():
    """The galactic l/b aliases survive the shared-registry refactor."""
    from astropy.table import Table

    import skyplothelper as sph
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", center=0, frame="galactic",
                        fig=fig)
    sph.plot_catalog(ax, Table({"l": [10.0, 20.0], "b": [5.0, -5.0]}))


# ============================================================
# plot_sky_vectors: ``scale`` is degrees on sky per unit vector
# ============================================================
#
# Regression: the degree displacement was handed straight to
# ``quiver(..., scale_units='xy')``, which measures length in the axes' data
# units -- WCS pixels, not degrees. Every arrow therefore picked up a spurious
# factor of the pixel scale, so arrows collapsed on any frame whose cdelt was
# not ~1 (a zoomed field, or the default all-sky grid at higher ``npix``).
#
# The rendered on-sky length is ``hypot(U, V) * deg_per_pixel``, so asserting on
# the quiver's U/V pins the contract exactly, without rasterizing.

def _deg_per_pix(ax):
    from astropy.wcs.utils import proj_plane_pixel_scales
    return float(np.mean(np.abs(
        np.asarray(proj_plane_pixel_scales(ax.wcs.celestial))[:2])))


def _vector_frames():
    return [
        ("AIT all-sky", dict(projection="AIT", center=180)),
        ("AIT npix", dict(projection="AIT", center=180, npix=(1200, 600))),
        ("TAN fov=12", dict(projection="TAN", center_lon=0, center_lat=0,
                            fov_deg=12)),
        ("TAN fov=40", dict(projection="TAN", center_lon=0, center_lat=0,
                            fov_deg=40)),
    ]


@pytest.mark.parametrize("name,kw", _vector_frames())
def test_plot_sky_vectors_scale_is_degrees_on_sky(name, kw):
    """scale=S with a unit vector renders S degrees, on every frame."""
    from skyplothelper.data_plots import plot_sky_vectors
    from skyplothelper.wcs_frame import make_wcs_frame
    fig = plt.figure(figsize=(5, 5))
    ax = make_wcs_frame(111, fig=fig, **kw)
    lon0, lat0 = ax.wcs.wcs.crval
    res = plot_sky_vectors(ax, [lon0], [lat0], [0.0], [1.0], units="deg",
                           scale=6.0)
    q = res.quiver
    length_deg = float(np.hypot(q.U[0], q.V[0])) * _deg_per_pix(ax)
    assert length_deg == pytest.approx(6.0, rel=1e-6), name
    plt.close(fig)


@pytest.mark.parametrize("name,kw", _vector_frames())
def test_plot_sky_vectors_auto_target_deg_hits_target(name, kw):
    """scale='auto' with auto_target_deg=T renders T degrees, on every frame."""
    from skyplothelper.data_plots import plot_sky_vectors
    from skyplothelper.wcs_frame import make_wcs_frame
    fig = plt.figure(figsize=(5, 5))
    ax = make_wcs_frame(111, fig=fig, **kw)
    lon0, lat0 = ax.wcs.wcs.crval
    res = plot_sky_vectors(ax, [lon0], [lat0], [0.0], [1.0], units="deg",
                           scale="auto", auto_target_deg=3.0)
    q = res.quiver
    length_deg = float(np.hypot(q.U[0], q.V[0])) * _deg_per_pix(ax)
    assert length_deg == pytest.approx(3.0, rel=1e-6), name
    plt.close(fig)


def test_plot_sky_vectors_length_invariant_to_npix_and_fov():
    """The same request renders the same on-sky length regardless of the
    frame's pixel scale -- raising npix must not shrink the arrows."""
    from skyplothelper.data_plots import plot_sky_vectors
    from skyplothelper.wcs_frame import make_wcs_frame
    lengths = []
    for _name, kw in _vector_frames():
        fig = plt.figure(figsize=(5, 5))
        ax = make_wcs_frame(111, fig=fig, **kw)
        lon0, lat0 = ax.wcs.wcs.crval
        res = plot_sky_vectors(ax, [lon0], [lat0], [0.0], [1.0], units="deg",
                               scale=5.0)
        q = res.quiver
        lengths.append(float(np.hypot(q.U[0], q.V[0])) * _deg_per_pix(ax))
        plt.close(fig)
    assert np.allclose(lengths, 5.0, rtol=1e-6), lengths


def test_wcs_deg_per_pixel_handles_cd_matrix_wcs():
    """cdelt is [1, 1] on a CD-matrix header (the scale lives in CD), so the
    pixel scale must come from proj_plane_pixel_scales, not cdelt."""
    from astropy.wcs import WCS

    from skyplothelper.data_plots import _wcs_deg_per_pixel
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [0.0, 0.0]
    w.wcs.crpix = [50.0, 50.0]
    w.wcs.cd = [[-0.001, 0.0], [0.0, 0.001]]
    with warnings.catch_warnings():
        # astropy warns exactly because cdelt is meaningless here, which is
        # the point of the assertion below.
        warnings.simplefilter("ignore", RuntimeWarning)
        assert np.allclose(np.abs(w.wcs.cdelt[:2]), 1.0)     # the trap
    assert _wcs_deg_per_pixel(w) == pytest.approx(0.001, rel=1e-9)


# ============================================================
# SkyVectorResult.scale / .deg_per_pix  +  sky_quiverkey
# ============================================================
#
# plot_sky_vectors hands ax.quiver its lengths in frame *pixels* (degrees /
# pixel-scale), so the quiver's U/V are pixels, not degrees. A raw quiverkey
# value in "the degrees scale is documented in" therefore comes out short by
# exactly deg_per_pix -- ~11% on the default all-sky grid, silently. The
# result now carries the resolved scale + deg_per_pix, and sky_quiverkey does
# the conversion so a key can't disagree with its field.

from skyplothelper.data_plots import (  # noqa: E402
    _VECTOR_UNIT_FACTORS,
    sky_quiverkey,
)


def _pm_frame(**kw):
    fig = plt.figure()
    ax = make_wcs_frame(111, fig=fig, **kw)
    return fig, ax


@pytest.mark.parametrize("name,kw", [
    ("AIT all-sky", dict(projection="AIT", center=180)),
    ("TAN fov=12", dict(projection="TAN", center_lon=0, center_lat=0,
                        fov_deg=12)),
])
def test_quiver_uv_is_in_pixel_units(name, kw):
    """Pin the unit contract: V == value * ufac * scale / deg_per_pix. If a
    refactor flips the quiver back to degree units, this fails."""
    fig, ax = _pm_frame(**kw)
    res = plot_sky_vectors(ax, [ax.wcs.wcs.crval[0]], [ax.wcs.wcs.crval[1]],
                           [0.0], [1000.0], units="mas", scale=20000)
    ufac = _VECTOR_UNIT_FACTORS["mas"]
    expected = 1000.0 * ufac * 20000 / res.deg_per_pix
    assert float(np.ravel(res.quiver.V)[0]) == pytest.approx(expected, rel=1e-9)
    plt.close(fig)


def test_sky_vector_result_stashes_scale_and_deg_per_pix():
    fig, ax = _pm_frame(projection="AIT", center=180)
    res = plot_sky_vectors(ax, [180], [0], [0.0], [1.0], units="mas",
                           scale=1234.0)
    assert res.scale == 1234.0
    from astropy.wcs.utils import proj_plane_pixel_scales
    dpp = float(np.mean(np.abs(
        np.asarray(proj_plane_pixel_scales(ax.wcs.celestial))[:2])))
    assert res.deg_per_pix == pytest.approx(dpp, rel=1e-9)
    assert len(res) == 4          # 4-field NamedTuple
    plt.close(fig)


def test_sky_vector_result_scale_auto_is_readable():
    """scale='auto' resolves to a number the caller can now read back."""
    fig, ax = _pm_frame(projection="AIT", center=180)
    res = plot_sky_vectors(ax, [180, 180], [0, 10], [1.0, 1.0], [1.0, 1.0],
                           units="mas", scale="auto", auto_target_deg=3.0)
    assert isinstance(res.scale, float) and res.scale > 0
    plt.close(fig)


@pytest.mark.parametrize("kw", [
    dict(projection="AIT", center=180),
    dict(projection="TAN", center_lon=0, center_lat=0, fov_deg=12),
])
def test_sky_quiverkey_matches_the_arrow_length(kw):
    """The key's U equals the drawn arrow's V for the same magnitude, on any
    frame -- which is the whole point: the key can't be short by deg_per_pix."""
    fig, ax = _pm_frame(**kw)
    lon0, lat0 = ax.wcs.wcs.crval
    res = plot_sky_vectors(ax, [lon0], [lat0], [0.0], [1000.0], units="mas",
                           scale=20000)
    qk = sky_quiverkey(res, ax, 0.9, 0.95, 1000, "1000 mas", units="mas")
    assert qk.U == pytest.approx(float(np.ravel(res.quiver.V)[0]), rel=1e-9)
    plt.close(fig)


def test_sky_quiverkey_unit_independent():
    """scale multiplies the degree-magnitude, so the same physical size in a
    different unit gives the same key length."""
    fig, ax = _pm_frame(projection="AIT", center=180)
    res = plot_sky_vectors(ax, [180], [0], [0.0], [1000.0], units="mas",
                           scale=20000)
    a = sky_quiverkey(res, ax, 0.9, 0.95, 1000, "1000 mas", units="mas")
    b = sky_quiverkey(res, ax, 0.9, 0.90, 1.0, "1 arcsec", units="arcsec")
    assert a.U == pytest.approx(b.U, rel=1e-9)   # 1000 mas == 1 arcsec
    plt.close(fig)


def test_sky_quiverkey_works_with_scale_auto_without_scale_arg():
    fig, ax = _pm_frame(projection="AIT", center=180)
    res = plot_sky_vectors(ax, [180, 180], [0, 10], [1.0, 1.0], [1.0, 1.0],
                           units="mas", scale="auto", auto_target_deg=3.0)
    qk = sky_quiverkey(res, ax, 0.9, 0.95, 1, "1 mas", units="mas")
    expected = 1 * _VECTOR_UNIT_FACTORS["mas"] * res.scale / res.deg_per_pix
    assert qk.U == pytest.approx(expected, rel=1e-9)
    plt.close(fig)


def test_sky_quiverkey_rejects_unknown_units():
    fig, ax = _pm_frame(projection="AIT", center=180)
    res = plot_sky_vectors(ax, [180], [0], [0.0], [1.0], units="mas", scale=1)
    with pytest.raises(ValueError, match="Unknown units"):
        sky_quiverkey(res, ax, 0.9, 0.9, 1, "x", units="furlongs")
    plt.close(fig)


# ---- SkyCoord catalogs ----

def _allsky(frame="ICRS"):
    import skyplothelper as sph
    return sph.allsky_figure(projection="AIT", center=180, frame=frame)


def _offsets(res):
    import numpy as np
    art = res[0] if isinstance(res, tuple) else res
    return np.asarray(art.get_offsets())


def test_plot_catalog_accepts_skycoord_array():
    """SkyCoord is the natural output of cone_search/region_search, so the
    search -> plot pipeline must not require hand-unwrapping .ra/.dec."""
    import numpy as np
    from astropy.coordinates import SkyCoord

    import skyplothelper as sph
    fig, ax = _allsky()
    sc = SkyCoord([83.6, 84.0, 85.0], [22.0, 22.5, 23.0], unit="deg")
    off = _offsets(sph.plot_catalog(ax, sc))
    assert np.allclose(off[:, 0], [83.6, 84.0, 85.0])
    plt.close(fig)


def test_plot_catalog_skycoord_scalar():
    from astropy.coordinates import SkyCoord

    import skyplothelper as sph
    fig, ax = _allsky()
    assert sph.plot_catalog(ax, SkyCoord(83.6, 22.0, unit="deg")) is not None
    plt.close(fig)


def test_plot_catalog_skycoord_converts_to_axes_frame():
    """A galactic SkyCoord must convert onto an ICRS axes, and stay put on a
    galactic axes — the frame is resolved from the axes, not assumed ICRS."""
    import numpy as np
    from astropy.coordinates import SkyCoord

    import skyplothelper as sph
    gal = SkyCoord([120.0, 121.0], [30.0, 31.0], unit="deg", frame="galactic")

    fig, ax = _allsky()                      # ICRS axes -> converted
    off = _offsets(sph.plot_catalog(ax, gal))
    assert np.allclose(off[:, 0], gal.icrs.ra.deg)
    plt.close(fig)

    fig, ax = _allsky(frame="galactic")      # galactic axes -> unchanged
    off = _offsets(sph.plot_catalog(ax, gal))
    assert np.allclose(off[:, 0], [120.0, 121.0])
    plt.close(fig)


# ---- SkyCoord forms for the vector/displacement plotters ----

def test_plot_sky_vectors_skycoord_with_keyword_deltas():
    """A SkyCoord takes the `lon` slot; deltas come by keyword. Deliberately
    NOT the positional-shift trick — this signature would need three shifts."""
    from astropy.coordinates import SkyCoord

    import skyplothelper as sph
    fig, ax = _allsky()
    sc = SkyCoord([80.0, 100.0], [10.0, 20.0], unit="deg")
    assert sph.plot_sky_vectors(ax, sc, dlon=[100.0, 100.0],
                                dlat=[50.0, 50.0]) is not None
    # positional arrays unchanged
    assert sph.plot_sky_vectors(ax, [80.0, 100.0], [10.0, 20.0],
                                [100.0, 100.0], [50.0, 50.0]) is not None
    plt.close(fig)


def test_plot_sky_vectors_rejects_mixed_forms():
    from astropy.coordinates import SkyCoord

    import skyplothelper as sph
    fig, ax = _allsky()
    with pytest.raises(TypeError):
        sph.plot_sky_vectors(ax, SkyCoord([80.0], [10.0], unit="deg"), [1.0],
                             dlon=[1.0], dlat=[1.0])
    plt.close(fig)


def test_plot_displacement_two_skycoords():
    """Two SkyCoords replace all four coordinate args — the same shape as
    Ruler.from_world(coord1, coord2)."""
    from astropy.coordinates import SkyCoord

    import skyplothelper as sph
    fig, ax = _allsky()
    a = sph.plot_displacement(ax, SkyCoord([80.0], [10.0], unit="deg"),
                              SkyCoord([90.0], [20.0], unit="deg"))
    b = sph.plot_displacement(ax, [80.0], [10.0], [90.0], [20.0])
    assert len(a) == len(b)
    plt.close(fig)


def test_plot_displacement_requires_complete_input():
    import skyplothelper as sph
    fig, ax = _allsky()
    with pytest.raises(TypeError):
        sph.plot_displacement(ax, [80.0], [10.0])
    plt.close(fig)


def test_plot_sky_vectors_warns_when_arrows_are_subpixel():
    """A field of sub-pixel arrows renders as dots and looks like the arrows
    failed to draw — but nothing errored, the magnitudes were just read in the
    wrong unit (`units` defaults to arcsec, so degree-scale amplitudes come out
    ~3600x too small). Warn instead of drawing something invisible.
    """
    import warnings

    import numpy as np

    import skyplothelper as sph
    fig, ax = _allsky()
    glon, glat = np.meshgrid(np.arange(0, 360, 20), np.arange(-75, 76, 15))
    dlon, dlat = sph.vsh_field(glon.ravel(), glat.ravel(), {"D_3": 1.0})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sph.plot_sky_vectors(ax, glon.ravel(), glat.ravel(), dlon, dlat,
                             scale=30)
    assert [w for w in caught if "shorter than one pixel" in str(w.message)]
    plt.close(fig)


@pytest.mark.parametrize("kwargs", [
    {"scale": 30, "units": "deg"},   # unit matched to the data
    {"scale": "auto"},               # auto-sized
])
def test_plot_sky_vectors_no_warning_when_visible(kwargs):
    import warnings

    import numpy as np

    import skyplothelper as sph
    fig, ax = _allsky()
    glon, glat = np.meshgrid(np.arange(0, 360, 20), np.arange(-75, 76, 15))
    dlon, dlat = sph.vsh_field(glon.ravel(), glat.ravel(), {"D_3": 1.0})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sph.plot_sky_vectors(ax, glon.ravel(), glat.ravel(), dlon, dlat,
                             **kwargs)
    assert not [w for w in caught
                if "shorter than one pixel" in str(w.message)]
    plt.close(fig)
