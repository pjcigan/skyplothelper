"""overlays.annotations return-value verification.

Verifies that the annotation helpers (``add_compass``, ``add_axis_inlay``,
``add_sizebar`` / ``add_sizebar_asec``, ``add_bandlabels``,
``add_colorbar``, ``add_contour_overlay``, ``style_ax_colors``)
return / mutate as expected.

Beam-related coverage lives in :mod:`tests.test_beam` (the
:class:`Beam` class).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.io import fits
from matplotlib.colorbar import Colorbar
from matplotlib.contour import QuadContourSet
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from skyplothelper.overlays.annotations import (
    add_axis_inlay,
    add_bandlabels,
    add_colorbar,
    add_compass,
    add_contour_overlay,
    add_sizebar,
    add_sizebar_asec,
    style_ax_colors,
)
from skyplothelper.wcs_frame import make_wcs_frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def tan_axes():
    """A small TAN field, the natural target for these annotations."""
    fig = plt.figure(figsize=(6, 6))
    ax = make_wcs_frame(111, projection="TAN", center=(180.0, 0.0), fig=fig)
    fig.canvas.draw()
    return fig, ax


def _tan_header_with_beam():
    """A FITS header carrying BMAJ/BMIN/BPA + CDELT for beam-from-header tests."""
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = 100
    hdr["NAXIS2"] = 100
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRPIX1"] = 50.5
    hdr["CRPIX2"] = 50.5
    hdr["CRVAL1"] = 180.0
    hdr["CRVAL2"] = 0.0
    hdr["CDELT1"] = -1.0 / 3600.0
    hdr["CDELT2"] = 1.0 / 3600.0
    hdr["BMAJ"] = 5.0 / 3600.0
    hdr["BMIN"] = 3.0 / 3600.0
    hdr["BPA"] = 30.0
    return hdr


# ============================================================
# add_colorbar
# ============================================================

def test_add_colorbar_returns_colorbar(tan_axes):
    fig, ax = tan_axes
    img = np.linspace(0, 1, 100).reshape(10, 10)
    im = ax.imshow(img, transform=ax.get_transform("pixel"))
    cb = add_colorbar(im, ax=ax, label="value")
    assert isinstance(cb, Colorbar)


def test_add_colorbar_label_propagates(tan_axes):
    fig, ax = tan_axes
    img = np.linspace(0, 1, 100).reshape(10, 10)
    im = ax.imshow(img, transform=ax.get_transform("pixel"))
    cb = add_colorbar(im, ax=ax, label="my label")
    # Colorbar label is on the y axis (vertical default)
    assert cb.ax.get_ylabel() == "my label"


# ============================================================
# add_contour_overlay
# ============================================================

def test_add_contour_overlay_returns_quadcontourset(tan_axes):
    fig, ax = tan_axes
    lon_grid, lat_grid = np.meshgrid(
        np.linspace(179.97, 180.03, 20),
        np.linspace(-0.03, 0.03, 20),
    )
    values = np.sin(np.radians(lon_grid - 180) * 100)
    cs = add_contour_overlay(ax, lon_grid, lat_grid, values, levels=5)
    assert isinstance(cs, QuadContourSet)


def test_add_contour_overlay_filled_returns_quadcontourset(tan_axes):
    fig, ax = tan_axes
    lon_grid, lat_grid = np.meshgrid(
        np.linspace(179.97, 180.03, 20),
        np.linspace(-0.03, 0.03, 20),
    )
    values = np.sin(np.radians(lon_grid - 180) * 100)
    cs = add_contour_overlay(ax, lon_grid, lat_grid, values,
                             levels=6, filled=True)
    assert isinstance(cs, QuadContourSet)


# ============================================================
# add_sizebar / add_sizebar_asec
# ============================================================

def test_add_sizebar_returns_anchoredsizebar(tan_axes):
    fig, ax = tan_axes
    sb = add_sizebar(ax, length_pixels=20, label="20 px")
    assert isinstance(sb, AnchoredSizeBar)


def test_add_sizebar_asec_returns_anchoredsizebar(tan_axes):
    """add_sizebar_asec uses the header's CDELT to convert to pixels and
    then delegates to add_sizebar."""
    fig, ax = tan_axes
    hdr = _tan_header_with_beam()
    sb = add_sizebar_asec(ax, hdr, length_asec=10.0, label="10″")
    assert isinstance(sb, AnchoredSizeBar)


# ============================================================
# add_compass
# ============================================================

def test_add_compass_returns_artist_list(tan_axes):
    fig, ax = tan_axes
    arts = add_compass(ax)
    assert isinstance(arts, list)
    # The compass adds at least 2 arrows (N, E) + 2 labels = 4-ish artists
    assert len(arts) >= 4


def test_add_compass_custom_labels_propagate(tan_axes):
    """Custom N/E labels should appear in the resulting Text artists."""
    fig, ax = tan_axes
    arts = add_compass(ax, north_label="↑", east_label="←")
    label_strings = [a.get_text() for a in arts if hasattr(a, "get_text")]
    assert "↑" in label_strings
    assert "←" in label_strings


# ============================================================
# add_axis_inlay
# ============================================================

def test_add_axis_inlay_returns_artist_list(tan_axes):
    fig, ax = tan_axes
    arts = add_axis_inlay(ax, lon_label="RA", lat_label="Dec")
    assert isinstance(arts, list)
    assert len(arts) >= 4  # arrows + labels


def test_add_axis_inlay_labels_propagate(tan_axes):
    fig, ax = tan_axes
    arts = add_axis_inlay(ax, lon_label="ℓ", lat_label="b")
    label_strings = [a.get_text() for a in arts if hasattr(a, "get_text")]
    assert "ℓ" in label_strings
    assert "b" in label_strings


# ============================================================
# add_bandlabels
# ============================================================

def test_add_bandlabels_creates_annotations(tan_axes):
    """``add_bandlabels`` returns None but adds annotations to the axes;
    we count them via ax.texts (Annotation is a subclass of Text)."""
    fig, ax = tan_axes
    n_before = len(ax.texts)
    add_bandlabels(ax, labels=["U", "B", "V"], labcolors=["red", "green", "blue"])
    n_after = len(ax.texts)
    assert n_after - n_before == 3


# ============================================================
# style_ax_colors
# ============================================================

def test_style_ax_colors_propagates_to_spines():
    """``style_ax_colors`` mutates the axes; verify spine colors changed."""
    fig, ax = plt.subplots()
    style_ax_colors(ax, color="red")
    # All four spines should now be red
    for spine in ax.spines.values():
        # Edgecolor as RGBA tuple — red has R=1, G=0, B=0
        ec = spine.get_edgecolor()
        assert ec[0] == pytest.approx(1.0)
        assert ec[1] == pytest.approx(0.0)
        assert ec[2] == pytest.approx(0.0)
