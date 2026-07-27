"""Geometry pipeline regression tests.

Pins a high-latitude-polygon bug: polygons
that touch / wrap the lon=0/360 seam project to sphere-spanning
polygons through the default ``_project_shape`` pipeline. The D3-
style ``_antimeridian_clip`` + ``_stitch_and_project`` pipeline
handles them correctly.

Root cause: at high latitude the AIT (and related) projection has a
1-px discontinuity across the lon=0/360 wrap that
``_detect_pixel_jumps_circular`` flags as a "jump." The "1 jump
non-pole" branch of ``_project_shape`` then calls
``_close_segment_centroid`` which walks the frame boundary, even
though the segment's endpoints (e.g. x=165, y=179 and x=193, y=179
for nside=32 RING pix 23 on AIT center=180) are nowhere near the
elliptical limb. The boundary walk produces an ~870-vertex polygon
spanning the whole top of the frame. The fix is a
circular-lon-coverage discriminator that recognizes the singularity
and skips the boundary walk.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from skyplothelper.geometry.shapes import add_spherical_polygon
from skyplothelper.wcs_frame import make_wcs_frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# nside=32 RING pix 23 — a polar tile whose corners straddle lon=0/360.
# Canonical reproduction: ``add_spherical_polygon`` (default path)
# returns x=[-0.5, 359.5] (full frame); ``clip='d3'`` returns a
# tight ~10° patch at x=[161.7, 170.3].
PIX23_LONS = np.array([360.0, 330.0, 337.5, 360.0])
PIX23_LATS = np.array([87.076, 85.613, 84.150, 85.613])


# Counter-example: high-lat antimeridian-touching tile (nside=32 RING
# pix 50) whose vertices stay on one side of the lon=0/360 wrap. This
# one renders correctly via *both* paths and shouldn't regress.
PIX50_LONS = np.array([180.0, 180.0, 195.0, 198.0])
PIX50_LATS = np.array([84.150, 82.690, 81.220, 82.690])


def _bbox(patches):
    xs = np.concatenate([p.get_path().vertices[:, 0] for p in patches])
    ys = np.concatenate([p.get_path().vertices[:, 1] for p in patches])
    return xs.min(), xs.max(), ys.min(), ys.max()


def _frame_extent(ax):
    return (ax.get_xlim()[1] - ax.get_xlim()[0],
            ax.get_ylim()[1] - ax.get_ylim()[0])


# ============================================================
# Polar wrap-spanning tile — the canonical 7.6 reproduction
# ============================================================

def test_polar_wrap_tile_default_path_stays_bounded():
    """nside=32 RING pix 23 via default ``add_spherical_polygon``.

    Pinning the Option B fix: added a
    circular-lon-coverage discriminator to ``_project_shape``'s
    1-jump branch so the high-lat WCS singularity at center±180°
    no longer routes through ``_close_segment_centroid``'s frame-
    boundary walk.
    """
    fig = plt.figure(figsize=(8, 4))
    ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    patches = add_spherical_polygon(ax, PIX23_LONS, PIX23_LATS)
    xmin, xmax, ymin, ymax = _bbox(patches)
    fxw, fyw = _frame_extent(ax)
    assert (xmax - xmin) < 0.5 * fxw, \
        f"polygon x-span {xmax - xmin:.1f} too large; frame is {fxw:.1f}"
    assert (ymax - ymin) < 0.5 * fyw, \
        f"polygon y-span {ymax - ymin:.1f} too large; frame is {fyw:.1f}"


def test_polar_wrap_tile_d3_clip_stays_bounded():
    """Positive control: same input through the D3-clip pipeline.

    Pinning that ``_antimeridian_clip`` + ``_stitch_and_project``
    produces correct output here — this is the working path the
    HEALPix helpers route through.
    """
    fig = plt.figure(figsize=(8, 4))
    ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    patches = add_spherical_polygon(ax, PIX23_LONS, PIX23_LATS,
                                    clip='d3')
    xmin, xmax, ymin, ymax = _bbox(patches)
    fxw, fyw = _frame_extent(ax)
    assert (xmax - xmin) < 0.5 * fxw
    assert (ymax - ymin) < 0.5 * fyw


def test_polar_wrap_tile_via_plot_healpix_sparse():
    """End-to-end test: ``plot_healpix_sparse`` renders pix 23 as a
    compact patch on AIT center=180.

    Pinning the Option A fix: routed the
    HEALPix patches backend through ``_antimeridian_clip`` +
    ``_stitch_and_project`` so high-lat wrap-spanning tiles don't
    trip ``_project_shape``'s frame-boundary closure.
    """
    from skyplothelper.healpix import plot_healpix_sparse

    fig = plt.figure(figsize=(8, 4))
    ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    art = plot_healpix_sparse([23], [1.0], nside=32, ax=ax,
                              cmap="viridis", set_extent=False)
    fxw, fyw = _frame_extent(ax)
    paths = art.get_paths()
    xs = np.concatenate([p.vertices[:, 0] for p in paths])
    ys = np.concatenate([p.vertices[:, 1] for p in paths])
    assert (xs.max() - xs.min()) < 0.5 * fxw, \
        (f"plot_healpix_sparse x-span {xs.max() - xs.min():.1f} too "
         f"large; frame is {fxw:.1f}")
    assert (ys.max() - ys.min()) < 0.5 * fyw


# ============================================================
# Counter-example — must not regress
# ============================================================

def test_high_lat_non_wrap_tile_default_compact():
    """Pix 50: high-lat tile that touches the antimeridian but doesn't
    wrap across the lon=0/360 seam. Renders compactly via both paths.
    Guards against any regression when the 7.6 fix lands.
    """
    fig = plt.figure(figsize=(8, 4))
    ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    patches = add_spherical_polygon(ax, PIX50_LONS, PIX50_LATS)
    xmin, xmax, _, _ = _bbox(patches)
    fxw, _ = _frame_extent(ax)
    assert (xmax - xmin) < 0.2 * fxw


def test_high_lat_non_wrap_tile_d3_clip_compact():
    fig = plt.figure(figsize=(8, 4))
    ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    patches = add_spherical_polygon(ax, PIX50_LONS, PIX50_LATS,
                                    clip='d3')
    xmin, xmax, _, _ = _bbox(patches)
    fxw, _ = _frame_extent(ax)
    assert (xmax - xmin) < 0.2 * fxw
