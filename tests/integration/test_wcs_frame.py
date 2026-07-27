"""wcs_frame return-value verification (all 32 projections).

Picks up where the canonical ``tests/test_wcs_frame.py`` leaves off:
the canonical suite parametrizes over a hand-picked subset of the 32
registered projections. This file exhaustively exercises the registry
— every
FITS code and every non-FITS projection in
``_PROJECTION_REGISTRY`` — plus the offset-WCS helpers and
``center_lon`` / ``center_lat`` keyword path that the canonical suite
doesn't cover.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.io.fits import Header
from astropy.visualization.wcsaxes import WCSAxes
from astropy.wcs import WCS

from skyplothelper.projections.registry import (
    _PROJECTION_REGISTRY,
    _resolve_projection,
    list_projections,
)
from skyplothelper.wcs_frame import (
    WCS_to_offsetWCS,
    dummy_allsky_hdr,
    dummy_offset_hdr,
    dummy_ortho_hdr,
    dummy_standard_hdr,
    make_wcs_frame,
    offset_coord_WCS,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ============================================================
# Exhaustive coverage of the projection registry
# ============================================================

# Some non-allsky FITS projections need a (lon, lat) tuple to construct
# a sensible field of view.
_FIELD_PROJ_CENTER = (180.0, 0.0)
_NON_ALLSKY_KEYS = {k for k, v in _PROJECTION_REGISTRY.items() if not v.allsky}

# Conic projections (COD/COE/COO/COP) require a PV2_1 standard-
# parallel parameter; ``make_wcs_frame`` supplies a sensible default
# (PV2_1=45) so they all construct successfully.
_NEEDS_PV_PARAMS = {"cod", "coe", "coo", "cop"}

_ALL_PROJ_KEYS = sorted(_PROJECTION_REGISTRY.keys())


@pytest.mark.parametrize("proj_key", _ALL_PROJ_KEYS)
def test_make_wcs_frame_constructs_for_every_registered_projection(proj_key):
    """Every entry in _PROJECTION_REGISTRY round-trips through
    ``make_wcs_frame``. Conic / Bonne projections rely on the
    PV2_1=45 default ``make_wcs_frame`` applies for them."""
    fig = plt.figure()
    center = _FIELD_PROJ_CENTER if proj_key in _NON_ALLSKY_KEYS else 180
    ax = make_wcs_frame(111, projection=proj_key, center=center, fig=fig)
    assert isinstance(ax, WCSAxes)


@pytest.mark.parametrize("proj_key", sorted(_NEEDS_PV_PARAMS))
def test_conic_default_pv2_1_constructs(proj_key):
    """COD/COE/COO/COP construct successfully with
    the PV2_1=45 default supplied by ``make_wcs_frame``."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection=proj_key, center=(180.0, 30.0),
                        fig=fig)
    assert isinstance(ax, WCSAxes)
    # Verify the PV card landed on the WCS
    pv = {(i, m): v for i, m, v in ax.wcs.wcs.get_pv()}
    # astropy's get_pv returns dict keyed by (axis, m); we asked for axis 2 m=1
    assert (2, 1) in pv, f"{proj_key}: PV2_1 not present on WCS"
    assert pv[(2, 1)] == 45.0


@pytest.mark.parametrize("proj_key", sorted(_NEEDS_PV_PARAMS))
def test_conic_user_pv2_1_overrides_default(proj_key):
    """User-supplied ``pv2_1`` overrides the default and lands on the
    WCS so different values produce different pixel mappings."""
    fig1 = plt.figure()
    ax1 = make_wcs_frame(111, projection=proj_key, center=(180.0, 30.0),
                         pv2_1=30.0, fig=fig1)
    fig2 = plt.figure()
    ax2 = make_wcs_frame(111, projection=proj_key, center=(180.0, 30.0),
                         pv2_1=60.0, fig=fig2)
    pv1 = {(i, m): v for i, m, v in ax1.wcs.wcs.get_pv()}
    pv2 = {(i, m): v for i, m, v in ax2.wcs.wcs.get_pv()}
    assert pv1[(2, 1)] == 30.0
    assert pv2[(2, 1)] == 60.0
    # Different standard parallels → different pixel mappings somewhere.
    # Sample a grid of points; at least one should differ. (Single-
    # point checks fail for projections where the chosen point happens
    # to be invariant under the parallel change.)
    test_lons = np.array([170., 175., 180., 185., 190.])
    test_lats = np.array([10., 25., 40., 55., 70.])
    LL, BB = np.meshgrid(test_lons, test_lats)
    x1, y1 = ax1.wcs.world_to_pixel_values(LL.ravel(), BB.ravel())
    x2, y2 = ax2.wcs.world_to_pixel_values(LL.ravel(), BB.ravel())
    assert not np.allclose(np.column_stack([x1, y1]),
                           np.column_stack([x2, y2])), \
        f"{proj_key}: pv2_1 overrides have no effect on pixel mapping"


def test_conic_pv2_2_spreads_two_standard_parallels():
    """Conics accept PV2_2 to specify the spread between two
    standard parallels. PV2_2=0 (default omission) → single
    parallel; non-zero values produce different projection geometry."""
    fig1 = plt.figure()
    ax1 = make_wcs_frame(111, projection='COD', center=(180.0, 30.0),
                         pv2_1=45.0, fig=fig1)
    fig2 = plt.figure()
    ax2 = make_wcs_frame(111, projection='COD', center=(180.0, 30.0),
                         pv2_1=45.0, pv2_2=15.0, fig=fig2)
    pv1 = {(i, m): v for i, m, v in ax1.wcs.wcs.get_pv()}
    pv2 = {(i, m): v for i, m, v in ax2.wcs.wcs.get_pv()}
    assert (2, 2) not in pv1 or pv1.get((2, 2), 0.0) == 0.0
    assert pv2[(2, 2)] == 15.0


@pytest.mark.parametrize("proj_key", ["AIT", "MOL", "SIN", "TAN", "CAR"])
def test_pv_kwargs_silently_ignored_for_non_conic_projections(proj_key):
    """Defensive check: passing pv2_1/pv2_2 to a
    projection that doesn't use them must NOT change the WCS pixel
    mapping. The PV cards should never be injected for AIT/MOL/etc.
    """
    center = (180.0, 0.0) if proj_key in ('SIN', 'TAN') else 180.0
    fig0 = plt.figure()
    ax0 = make_wcs_frame(111, projection=proj_key, center=center, fig=fig0)
    fig1 = plt.figure()
    ax1 = make_wcs_frame(111, projection=proj_key, center=center,
                         pv2_1=99.0, pv2_2=33.0, fig=fig1)
    test_lons = np.array([0.0, 90.0, 180.0, 270.0])
    test_lats = np.array([-30.0, 0.0, 30.0, 60.0])
    x0, y0 = ax0.wcs.world_to_pixel_values(test_lons, test_lats)
    x1, y1 = ax1.wcs.world_to_pixel_values(test_lons, test_lats)
    np.testing.assert_array_equal(x0, x1)
    np.testing.assert_array_equal(y0, y1)


def test_registry_has_32_entries():
    """Sanity check: the registry ships the full 32-projection set,
    including BON on a rectangular (kapteyn-style) frame."""
    assert len(_PROJECTION_REGISTRY) == 32


def test_list_projections_lists_every_registered_projection(capsys):
    """list_projections() prints to stdout; descriptions are unique
    so verify each shows up in the captured output."""
    list_projections()
    out = capsys.readouterr().out
    for info in _PROJECTION_REGISTRY.values():
        descr = info.description.replace(" [non-FITS]", "")
        assert descr in out, (
            f"list_projections() output does not mention {descr!r}"
        )


# ============================================================
# Alias dispatch (case + separator tolerance)
# ============================================================

@pytest.mark.parametrize("alias, expected_fits", [
    ("AIT", "AIT"),
    ("ait", "AIT"),
    ("Aitoff", "AIT"),
    ("hammer-aitoff", "AIT"),
    ("hammer_aitoff", "AIT"),
    ("MOLLWEIDE", "MOL"),
    ("plate-carree", "CAR"),
    ("plate_carree", "CAR"),
    ("equirectangular", "CAR"),
    ("orthographic", "SIN"),
])
def test_alias_resolution_is_lenient(alias, expected_fits):
    _key, info = _resolve_projection(alias)
    assert info.fits_code == expected_fits


# ============================================================
# center_lon / center_lat keyword path
# ============================================================

def test_make_wcs_frame_center_lon_keyword():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", center_lon=120, fig=fig)
    assert ax.wcs.wcs.crval[0] == pytest.approx(120.0)


def test_make_wcs_frame_center_lon_lat_keywords_for_globe():
    fig = plt.figure()
    ax = make_wcs_frame(
        111, projection="SIN", center_lon=83.6, center_lat=22.0, fig=fig,
    )
    assert ax.wcs.wcs.crval[0] == pytest.approx(83.6)
    assert ax.wcs.wcs.crval[1] == pytest.approx(22.0)


# ============================================================
# frame= parameter propagates to CTYPE
# ============================================================

@pytest.mark.parametrize("frame, expected_ctype1_prefix", [
    ("ICRS", "RA--"),
    ("Galactic", "GLON"),
])
def test_make_wcs_frame_respects_frame_parameter(frame, expected_ctype1_prefix):
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", center=0, frame=frame, fig=fig)
    assert ax.wcs.wcs.ctype[0].startswith(expected_ctype1_prefix)


# ============================================================
# return_hdr=True returns (ax, hdr)
# ============================================================

def test_return_hdr_returns_tuple():
    fig = plt.figure()
    out = make_wcs_frame(111, projection="MOL", return_hdr=True, fig=fig)
    assert isinstance(out, tuple) and len(out) == 2
    ax, hdr = out
    assert isinstance(ax, WCSAxes)
    assert isinstance(hdr, Header)
    assert hdr["CTYPE1"].endswith("MOL")


# ============================================================
# Header generators
# ============================================================

def test_dummy_allsky_hdr_carries_through_to_wcs():
    hdr = dummy_allsky_hdr(center_LONdeg=42, projection="MOL")
    wcs = WCS(hdr)
    # Round-trip a center pixel through the WCS to confirm the header is valid.
    pix = np.array([[hdr["CRPIX1"] - 1, hdr["CRPIX2"] - 1]])
    world = wcs.wcs_pix2world(pix, 0)[0]
    assert world[0] == pytest.approx(42.0, abs=1e-6)


def test_dummy_ortho_hdr_carries_through_to_wcs():
    hdr = dummy_ortho_hdr(center_LONdeg=120, center_LATdeg=30)
    wcs = WCS(hdr)
    assert wcs.wcs.ctype[0].endswith("SIN")
    assert wcs.wcs.crval[0] == pytest.approx(120)
    assert wcs.wcs.crval[1] == pytest.approx(30)


def test_dummy_offset_hdr_returns_offset_units():
    """Offset header puts CTYPE in xoffset/yoffset and CUNIT in arcsec."""
    hdr = dummy_offset_hdr(centercoords_deg=(180.0, 0.0), offset_units="arcsec")
    assert "xoffset" in hdr["CTYPE1"].lower() or hdr["CUNIT1"] == "arcsec"


def test_dummy_standard_hdr_basic_validity():
    hdr = dummy_standard_hdr(centercoords_deg=(0.0, 0.0))
    wcs = WCS(hdr)
    assert wcs.wcs.crval[0] == pytest.approx(0.0)
    assert wcs.wcs.crval[1] == pytest.approx(0.0)


# ============================================================
# Offset-WCS round-trip: convert sky WCS → offset WCS, then back
# ============================================================

def test_WCS_to_offsetWCS_round_trip():
    """WCS_to_offsetWCS produces a WCS centered on the input coords with
    offset CTYPEs; round-tripping the centerpoint through both WCSes
    should yield (0, 0) in the offset frame."""
    hdr = dummy_standard_hdr(
        centercoords_deg=(180.0, 30.0), cdelts=(-0.001, 0.001),
        cunit="deg", projection="TAN",
    )
    sky_wcs = WCS(hdr)
    offset_wcs = WCS_to_offsetWCS(
        sky_wcs, centercoords=(180.0, 30.0), offset_units="arcsec",
    )
    # The (0, 0) offset world coord must map back to the WCS center
    # pixel (CRPIX-1 in 0-based indexing).
    offset_xy = offset_wcs.wcs_world2pix([[0.0, 0.0]], 0)[0]
    assert offset_xy[0] == pytest.approx(offset_wcs.wcs.crpix[0] - 1,
                                         abs=1e-6)
    assert offset_xy[1] == pytest.approx(offset_wcs.wcs.crpix[1] - 1,
                                         abs=1e-6)
    # Round-tripping in offset world coords must yield zero offset
    sky_at_center = offset_wcs.wcs_pix2world(
        [[offset_wcs.wcs.crpix[0] - 1, offset_wcs.wcs.crpix[1] - 1]], 0,
    )[0]
    assert sky_at_center[0] == pytest.approx(0.0, abs=1e-6)
    assert sky_at_center[1] == pytest.approx(0.0, abs=1e-6)


def test_offset_coord_WCS_returns_wcs_with_offset_ctypes():
    """offset_coord_WCS takes a sky header and yields an offset WCS."""
    hdr = dummy_standard_hdr(
        centercoords_deg=(83.6, 22.0), cdelts=(-0.001, 0.001),
        cunit="deg", projection="TAN",
    )
    offset_wcs = offset_coord_WCS(
        hdr, centercoords=(83.6, 22.0), offset_units="arcsec",
    )
    assert isinstance(offset_wcs, WCS)
    # The offset WCS should be in arcsec units
    assert offset_wcs.wcs.cunit[0].to_string().startswith("arcsec") or \
           offset_wcs.wcs.cunit[0].name == "arcsec"


# ============================================================
# fov_deg convenience (field-of-view → cdelt)
# ============================================================

def test_make_wcs_frame_fov_deg_computes_cdelt():
    """``fov_deg=`` is a convenience alternative to ``cdelt``;
    ``cdelt = fov_deg / max(npix)`` should be applied automatically."""
    fig = plt.figure()
    ax = make_wcs_frame(111, "TAN", center=(180.0, 0.0),
                        fov_deg=0.5, npix=400, fig=fig)
    # max axis = 400, so cdelt = 0.5/400 = 0.00125 (sign depends on lon).
    assert abs(abs(ax.wcs.wcs.cdelt[0]) - 0.5 / 400) < 1e-12
    assert abs(ax.wcs.wcs.cdelt[1] - 0.5 / 400) < 1e-12


def test_make_wcs_frame_fov_deg_and_cdelt_both_raises():
    """Passing both ``fov_deg`` and ``cdelt`` is ambiguous — raise."""
    fig = plt.figure()
    with pytest.raises(TypeError, match="both"):
        make_wcs_frame(111, "TAN", center=(180.0, 0.0),
                       fov_deg=0.5, cdelt=0.001, fig=fig)


def test_make_wcs_frame_tan_field_from_fov_deg():
    """Build a TAN field axes from ``center`` + ``fov_deg``."""
    fig = plt.figure(figsize=(8, 8))
    ax = make_wcs_frame(
        111, "TAN", center=(83.633, 22.015), fov_deg=0.5, npix=500, fig=fig)
    assert isinstance(ax, WCSAxes)
    assert ax.wcs.wcs.ctype[0].endswith("TAN")
    assert abs(abs(ax.wcs.wcs.cdelt[0]) - 0.5 / 500) < 1e-12


def test_make_wcs_frame_accepts_existing_axes():
    """``subplotnumber=`` accepts an existing matplotlib Axes —
    swap it for a WCSAxes at the same SubplotSpec position. Useful
    for ``plt.subplots`` workflows."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    plain_ax = axes[0, 1]
    plain_axes_in_fig = list(fig.axes)
    assert plain_ax in plain_axes_in_fig

    sky_ax = make_wcs_frame(plain_ax, "AIT", center=180)
    # New axes is in the figure
    assert sky_ax.figure is fig
    assert sky_ax in fig.axes
    # Old axes was removed
    assert plain_ax not in fig.axes
    # Got a WCSAxes
    assert isinstance(sky_ax, WCSAxes)


def test_make_wcs_frame_accepts_gridspec_axes():
    """``subplotnumber=`` Axes built via ``GridSpec`` works the same
    way."""
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    plain_ax = fig.add_subplot(gs[1, 2])
    sky_ax = make_wcs_frame(plain_ax, "MOL", center=0)
    assert isinstance(sky_ax, WCSAxes)
    assert plain_ax not in fig.axes
    assert sky_ax in fig.axes


def test_make_wcs_frame_existing_axes_cross_figure_raises():
    """Passing both an explicit ``fig=`` and an Axes from a
    different figure is ambiguous — raise."""
    fig_a = plt.figure()
    fig_b = plt.figure()
    plain_ax_a = fig_a.add_subplot(111)
    with pytest.raises(ValueError, match="different figure"):
        make_wcs_frame(plain_ax_a, "AIT", fig=fig_b)


def test_make_wcs_frame_existing_axes_no_subplotspec_raises():
    """Pre-existing Axes from ``fig.add_axes(rect)`` has no
    SubplotSpec — raise with a helpful redirect."""
    fig = plt.figure()
    free_ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    with pytest.raises(ValueError, match="SubplotSpec"):
        make_wcs_frame(free_ax, "AIT")
