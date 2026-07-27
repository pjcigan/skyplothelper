"""figures (allsky_figure / offset_figure / projection_gallery).

Verifies the high-level multi-call wrappers in ``skyplothelper.figures``
return the expected (fig, ax) shapes and propagate the documented
parameters (frame, fov_deg, projection list).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from astropy.visualization.wcsaxes import WCSAxes
from matplotlib.figure import Figure

from skyplothelper.figures import (
    allsky_figure,
    offset_figure,
    projection_gallery,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ============================================================
# allsky_figure
# ============================================================

def test_allsky_figure_returns_fig_ax_tuple():
    fig, ax = allsky_figure(projection="AIT", center=180)
    assert isinstance(fig, Figure)
    assert isinstance(ax, WCSAxes)


def test_allsky_figure_propagates_frame_to_wcs():
    fig, ax = allsky_figure(projection="AIT", center=0, frame="Galactic")
    assert ax.wcs.wcs.ctype[0].startswith("GLON")


def test_allsky_figure_default_figsize_is_landscape():
    """Default figsize should be a wide rectangle suitable for an AIT all-sky."""
    fig, ax = allsky_figure()
    w, h = fig.get_size_inches()
    assert w > h, f"expected landscape default, got {w}×{h}"


# ============================================================
# offset_figure
# ============================================================

def test_offset_figure_returns_fig_ax_tuple():
    fig, ax = offset_figure(center=(180.0, 30.0), fov_deg=0.5)
    assert isinstance(fig, Figure)
    assert isinstance(ax, WCSAxes)


def test_offset_figure_centers_axes_on_input_coords():
    fig, ax = offset_figure(center=(83.6, 22.0), fov_deg=0.1, projection="TAN")
    assert ax.wcs.wcs.crval[0] == pytest.approx(83.6)
    assert ax.wcs.wcs.crval[1] == pytest.approx(22.0)


def test_offset_figure_fov_controls_pixel_scale():
    """fov_deg / npix becomes the per-pixel CDELT — verify the scaling."""
    fig_a, ax_a = offset_figure(center=(0.0, 0.0), fov_deg=1.0, npix=200)
    fig_b, ax_b = offset_figure(center=(0.0, 0.0), fov_deg=0.1, npix=200)
    # |cdelt| must scale with fov for a fixed npix
    cdelt_a = abs(ax_a.wcs.wcs.cdelt[0])
    cdelt_b = abs(ax_b.wcs.wcs.cdelt[0])
    assert cdelt_a / cdelt_b == pytest.approx(10.0, rel=1e-3), (
        f"expected 10× ratio, got {cdelt_a!r}/{cdelt_b!r}"
    )


# ============================================================
# projection_gallery
# ============================================================

def test_projection_gallery_default_returns_fig_and_axes_list():
    """The default 6-projection set returns 6 successful axes; the
    default list is ['AIT', 'MOL', 'SFL', 'CAR', 'PAR', 'PCO']."""
    fig, axes = projection_gallery()
    assert isinstance(fig, Figure)
    assert len(axes) == 6, f"expected 6 successful panels, got {len(axes)}"
    assert all(isinstance(ax, WCSAxes) for ax in axes)


def test_projection_gallery_custom_projections_list():
    """A custom projection list of constructable projections should
    produce exactly that many axes."""
    fig, axes = projection_gallery(projections=["AIT", "MOL", "SFL", "CAR"])
    assert len(axes) == 4
    fits_codes = [ax.wcs.wcs.ctype[0][-3:] for ax in axes]
    assert "AIT" in fits_codes
    assert "MOL" in fits_codes


def test_projection_gallery_center_lon_lat_tuple():
    # Regression: passing center_lon + center_lat makes `center` a (lon, lat)
    # tuple, which used to crash inside healpix_to_celestial (center_deg - 180
    # on a tuple). Only the longitude feeds the all-sky grid now.
    fig, axes = projection_gallery(projections=["AIT", "MOL"],
                                   center_lon=180, center_lat=30)
    assert len(axes) == 2


def test_projection_gallery_ncols_controls_layout():
    """The grid uses ceil(n / ncols) rows."""
    fig, axes = projection_gallery(
        projections=["AIT", "MOL", "SFL", "CAR"], ncols=2,
    )
    # ncols=2 with 4 projections → 2 rows × 2 cols
    # Each subplot should occupy 1/2 of the figure horizontally
    bboxes = [ax.get_position() for ax in axes]
    widths = [bb.width for bb in bboxes]
    # No subplot should span the entire figure width with ncols=2
    assert max(widths) < 0.6, (
        f"expected subplot widths < 0.6 with ncols=2, got max={max(widths)!r}"
    )
