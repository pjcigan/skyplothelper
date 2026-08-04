"""Filled regions on a SIN-projection globe must close along the visible-
hemisphere *limb*, not chord across the disk.

Regression for the globe limb-fill bug: a filled polygon (a landmass, a plate,
a large cap) that spills past the visible hemisphere used to be closed with a
straight chord — the far-side vertices project to NaN and only the finite ones
were joined. On a globe the ``center±180`` antimeridian is not a visible seam
(it is on the far side), so the fix routes globe region-fills through the whole-
ring + limb-bridge path, walking the frame silhouette (the limb) across each
off-frame gap. Flat all-sky frames project every point to a finite pixel and are
unaffected.

The visible signature of the old bug was a filled area that was either a tiny
sliver (chord cutting most of the region away) or the complement; these tests
pin the area to the plausible visible fraction instead.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import skyplothelper as sph  # noqa: E402
from skyplothelper.geometry._projector import WCSAxesProjector  # noqa: E402
from skyplothelper.geometry.shapes import geodesic_circle  # noqa: E402


def _globe(center_lon=0.0, center_lat=0.0):
    fig = plt.figure(figsize=(5, 5))
    ax = sph.make_globe_frame(111, center_LONdeg=center_lon,
                              center_LATdeg=center_lat)
    return fig, ax


def _cap_frac(ax, lon, lat, radius_deg):
    proj = WCSAxesProjector(ax)
    lons, lats = geodesic_circle(lon, lat, radius_deg, 240)
    geom = proj.project_polygon(lons, lats, clip="d3", lat_center=lat,
                                radius_deg=radius_deg)
    if geom is None or geom.is_empty:
        return 0.0
    return geom.area / proj.frame_polygon.area


def test_footprint_inside_hemisphere_unchanged():
    """A cap wholly inside the visible hemisphere fills a plain disk (the case
    that always worked — must stay working)."""
    fig, ax = _globe()
    frac = _cap_frac(ax, 0, 0, 40)
    # A 40-deg cap at the near point covers a moderate central disk.
    assert 0.1 < frac < 0.55, frac
    plt.close(fig)


@pytest.mark.parametrize("radius", [60, 75, 88])
def test_cap_crossing_limb_is_a_large_lune_not_a_chord(radius):
    """A cap centered off the near point so it crosses the limb must fill a
    substantial lune — not a chord sliver (old bug ⇒ near-0) nor the
    complement."""
    fig, ax = _globe()
    frac = _cap_frac(ax, 35, 20, radius)
    assert 0.05 < frac < 0.98, f"radius={radius} frac={frac}"
    plt.close(fig)


@pytest.mark.parametrize("radius", [95, 110, 150, 179])
def test_cap_enclosing_whole_visible_hemisphere_fills_disk(radius):
    """A cap larger than 90 deg centered on the near point encloses the whole
    visible hemisphere. Its boundary ring is entirely far-side (no limb
    crossings), but the fill is the whole disk — the winding at the near point
    (_encloses_direction) catches this and returns the frame silhouette."""
    fig, ax = _globe()
    frac = _cap_frac(ax, 0, 0, radius)
    assert frac > 0.97, f"radius={radius} frac={frac}"
    plt.close(fig)


@pytest.mark.parametrize("radius", [20, 40, 60])
def test_far_side_cap_at_antipode_stays_empty(radius):
    """The complement of the enclose case: a small cap centered on the *far*
    point (the antipode of the globe center) is entirely on the back of the
    globe and must render empty — the winding sign (~-2pi, antipode) keeps
    _encloses_direction from wrongly filling the disk."""
    fig, ax = _globe()
    frac = _cap_frac(ax, 180, 0, radius)
    assert frac < 0.02, f"radius={radius} frac={frac}"
    plt.close(fig)


def test_offcenter_globe_limb_fill():
    """The limb fill also works on a globe centered off the equator."""
    fig, ax = _globe(center_lon=-60, center_lat=30)
    frac2 = _cap_frac(ax, -10, 10, 70)      # a limb-crosser
    assert 0.05 < frac2 < 0.98, frac2
    plt.close(fig)


def test_plot_land_on_globe_renders_smoke():
    """End-to-end: filled land on a globe renders patches (skips if the Earth
    data isn't present — the conftest hook turns that into a skip)."""
    from matplotlib.patches import PathPatch
    fig = plt.figure(figsize=(5, 5))
    ax = sph.make_planet_frame(111, body="earth", center_LONdeg=20,
                               center_LATdeg=20)
    sph.plot_land(ax, facecolor="#cbb994")
    n = sum(isinstance(p, PathPatch) for p in ax.patches)
    assert n >= 1
    plt.close(fig)
