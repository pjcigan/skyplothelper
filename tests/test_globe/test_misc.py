"""Smoke tests for nightshade, insets, baselines, boundaries."""

import matplotlib

matplotlib.use("Agg")


import matplotlib.pyplot as plt
import pytest

from skyplothelper._compat import coord_ticklabels
from skyplothelper.globe.baselines import _format_baseline_length, _normalize_sites
from skyplothelper.globe.boundaries import (
    _BOUNDARY_DATA_URLS,
    _find_data_file,
    fetch_boundary_data,
)
from skyplothelper.globe.frame import make_globe_frame, make_planet_frame
from skyplothelper.globe.insets import (
    _east_increases_right,
    reproject_inset_axes,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---- baselines ----

def test_normalize_sites_dict():
    """_normalize_sites returns a list of (name, lon, lat) tuples."""
    sites = {"VLA": (-107.6184, 34.0784), "GBT": (-79.8398, 38.4331)}
    out = _normalize_sites(sites)
    names = {entry[0] for entry in out}
    assert names == {"VLA", "GBT"}


def test_format_baseline_length_runs():
    s = _format_baseline_length(3000.0, unit="km")
    assert "3000" in s or "km" in s


# ---- boundaries / fetch ----

def test_boundary_data_urls_populated():
    """Registry has entries for the standard boundary files."""
    expected = {
        "coastlines.npz", "tectonic_plates.npz", "time_zones.npz",
    }
    assert expected == set(_BOUNDARY_DATA_URLS.keys())


def test_find_data_file_missing_returns_none():
    assert _find_data_file("definitely_does_not_exist.npz") is None


def test_fetch_boundary_data_unknown_filename_raises():
    with pytest.raises(ValueError, match="Unknown boundary data file"):
        fetch_boundary_data("not_a_real_file.npz")


def test_fetch_boundary_data_local_mirror(tmp_path):
    """A directory passed as base_url is used as an offline mirror:
    files are copied from there with no network access."""
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "coastlines.npz").write_bytes(b"fake-npz-bytes")
    dest = tmp_path / "dest"

    out = fetch_boundary_data(
        filename="coastlines.npz",
        dest=str(dest),
        base_url=str(mirror),
        progress=False,
    )
    assert out == [str(dest / "coastlines.npz")]
    assert (dest / "coastlines.npz").read_bytes() == b"fake-npz-bytes"


def test_fetch_boundary_data_local_mirror_missing_warns(tmp_path):
    """Mirror without the requested file warns and returns empty (the
    FileNotFoundError is caught and retried, then surfaced as a warning)."""
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    with pytest.warns(UserWarning, match="Could not fetch"):
        out = fetch_boundary_data(
            filename="coastlines.npz",
            dest=str(tmp_path / "dest"),
            base_url=str(mirror),
            retries=0,
            progress=False,
        )
    assert out == []


def test_fetch_boundary_data_hash_mismatch_discards(tmp_path, monkeypatch):
    """A registered hash that doesn't match the mirrored file discards the
    download rather than keeping corrupt data."""
    from skyplothelper.globe import boundaries

    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / "coastlines.npz").write_bytes(b"wrong-content")
    monkeypatch.setitem(
        boundaries._BOUNDARY_DATA_URLS,
        "coastlines.npz",
        ("http://example.invalid/coastlines.npz", "0" * 64),
    )
    dest = tmp_path / "dest"
    with pytest.warns(UserWarning, match="Hash mismatch"):
        out = fetch_boundary_data(
            filename="coastlines.npz",
            dest=str(dest),
            base_url=str(mirror),
            retries=0,
            progress=False,
        )
    assert out == []
    assert not (dest / "coastlines.npz").exists()


# ---- insets ----

def test_reproject_inset_axes_smoke():
    fig = plt.figure()
    parent = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    inset = reproject_inset_axes(
        parent, rect=[0.65, 0.05, 0.3, 0.3], projection="TAN",
        center=(10, 10), size=10.0, npix=60,
    )
    assert inset is not None


def test_reproject_inset_axes_bounds_from_projected_limb():
    """Inset limits come from the projected angular edge, not linear size/cdelt.
    On SIN, a size=170 deg cap's limb sits at (180/pi)*sin(85 deg)/cdelt px —
    not the linear 85/cdelt (which over-extends by ~pi/2)."""
    import numpy as np
    fig = plt.figure()
    parent = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    inset = reproject_inset_axes(
        parent, rect=[0.6, 0.05, 0.35, 0.35], projection="SIN",
        center=(120, -20), size=170.0, npix=400, auto_fontsize=False)
    (y0, y1) = inset.get_ylim()
    half = (y1 - y0) / 2
    cdelt = abs(inset.wcs.wcs.cdelt[1])
    true_limb = (180 / np.pi) * np.sin(np.radians(85.0)) / cdelt
    linear = 0.5 * 170.0 / cdelt
    assert half == pytest.approx(true_limb, rel=1e-3)
    assert half < 0.8 * linear          # clearly tighter than the old linear bound


def test_reproject_inset_axes_facecolor_applies():
    """facecolor= now paints the inset (was a constructor no-op)."""
    import matplotlib.colors as mcolors
    fig = plt.figure()
    parent = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    ins = reproject_inset_axes(
        parent, rect=[0.6, 0.05, 0.3, 0.3], projection="TAN",
        center=(10, 10), size=5.0, npix=60, facecolor="red",
        auto_fontsize=False)
    assert mcolors.to_hex(ins.patch.get_facecolor()) == mcolors.to_hex("red")


def test_reproject_inset_axes_bg_color_artist():
    """bg_color adds a low-zorder background artist (survives transparent save)."""
    fig = plt.figure()
    parent = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    ins = reproject_inset_axes(
        parent, rect=[0.6, 0.05, 0.3, 0.3], projection="TAN",
        center=(10, 10), size=5.0, npix=60, bg_color="#3355aa",
        auto_fontsize=False)
    bg = [p for p in ins.patches if p.get_gid() == "_sph_inset_bg"]
    assert len(bg) == 1
    assert bg[0].get_zorder() < 0            # behind all inset content


def test_reproject_inset_axes_bg_color_follows_elliptical_frame():
    """The bg artist follows the frame shape (many-vertex ellipse for AIT)."""
    from astropy.visualization.wcsaxes.frame import EllipticalFrame
    fig = plt.figure()
    parent = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    ins = reproject_inset_axes(
        parent, rect=[0.55, 0.1, 0.35, 0.35], projection="AIT",
        center=(120, -20), size=40.0, npix=200, bg_color="navy",
        frame_class=EllipticalFrame, auto_fontsize=False)
    bg = [p for p in ins.patches if p.get_gid() == "_sph_inset_bg"][0]
    assert bg.get_path().vertices.shape[0] > 20   # ellipse, not a rectangle


def _inset_east_right(parent, **kw):
    inset = reproject_inset_axes(
        parent, rect=[0.65, 0.05, 0.3, 0.3], projection="TAN",
        center=(10, 10), size=10.0, npix=60, **kw)
    return _east_increases_right(inset.wcs)


def test_inset_inherits_parent_direction():
    """By default ('inherit') an inset matches the parent's on-screen east
    direction, so a geographic parent doesn't get a mirrored astro inset."""
    fig = plt.figure()
    geo_parent = make_planet_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    assert _inset_east_right(geo_parent) is True   # geo parent → geo inset

    fig2 = plt.figure()
    sky_parent = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig2.canvas.draw()
    assert _inset_east_right(sky_parent) is False  # sky parent → sky inset


def test_inset_direction_explicit_override():
    """An explicit direction= overrides the inherited parent orientation."""
    fig = plt.figure()
    geo_parent = make_planet_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    assert _inset_east_right(geo_parent, direction="sky") is False
    assert _inset_east_right(geo_parent, direction="geo") is True


# ---- inset tick styling (curved globe-like insets get in-frame labels) ----

def _has_overlay_ticklabels(ax):
    """True if the axes carries skyplothelper in-frame overlay tick labels."""
    return any(getattr(t, "_sph_overlay_ticklabel", False) for t in ax.texts)


def test_sin_inset_gets_in_frame_ticks_by_default():
    """A SIN (orthographic, frame_shape 'circular') inset inherits astropy's
    poorly-rendered native curved-frame ticks; tick_style='auto' routes it to
    the same clean in-frame overlay labels make_globe_frame uses, and suppresses
    the native ones."""
    fig = plt.figure(figsize=(6, 6))
    parent = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    inset = reproject_inset_axes(
        parent, rect=[0.55, 0.1, 0.4, 0.4], projection="SIN",
        center=(120, -20), size=40.0, npix=200)
    fig.canvas.draw()
    assert _has_overlay_ticklabels(inset)               # in-frame labels drawn
    assert inset._sph_is_allsky is False                # treated as a field
    # native tick labels suppressed so they don't double up with the overlay
    assert not coord_ticklabels(inset.coords[0]).get_visible()
    assert not coord_ticklabels(inset.coords[1]).get_visible()


def test_tan_inset_keeps_native_ticks():
    """A TAN (rectilinear) inset renders cleanly with astropy native ticks, so
    'auto' leaves it untouched — no overlay labels, native labels visible."""
    fig = plt.figure(figsize=(6, 6))
    parent = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    inset = reproject_inset_axes(
        parent, rect=[0.6, 0.05, 0.35, 0.35], projection="TAN",
        center=(10, 10), size=10.0, npix=100)
    fig.canvas.draw()
    assert not _has_overlay_ticklabels(inset)
    assert coord_ticklabels(inset.coords[0]).get_visible()


def test_sin_inset_tick_style_native_opt_out():
    """tick_style='native' forces bare astropy ticks even on a SIN inset."""
    fig = plt.figure(figsize=(6, 6))
    parent = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    inset = reproject_inset_axes(
        parent, rect=[0.55, 0.1, 0.4, 0.4], projection="SIN",
        center=(120, -20), size=40.0, npix=200, tick_style="native")
    fig.canvas.draw()
    assert not _has_overlay_ticklabels(inset)
