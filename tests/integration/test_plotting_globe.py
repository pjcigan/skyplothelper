"""globe plotting return-value verification.

Verifies that the globe plotting helpers (``plot_scatter_globe``,
``plot_line_globe``, ``plot_pcolormesh_globe``, ``plot_contour_globe``,
``imscatter`` / ``imscatter_rotated``) return the right kind of artist
with the right contents after the merge.

See ``tests/integration/README.md`` for context.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.contour import QuadContourSet
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox

from skyplothelper.globe.frame import make_globe_frame
from skyplothelper.globe.plotting import (
    imscatter,
    imscatter_rotated,
    plot_contour_globe,
    plot_line_globe,
    plot_pcolormesh_globe,
    plot_scatter_globe,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def globe_axes():
    fig = plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    return fig, ax


# ============================================================
# plot_scatter_globe
# ============================================================

def test_plot_scatter_globe_returns_pathcollection(globe_axes):
    fig, ax = globe_axes
    sc = plot_scatter_globe(ax, lons=[0, 10, 20], lats=[0, 5, -5])
    assert isinstance(sc, PathCollection), \
        f"expected PathCollection, got {type(sc).__name__}"


def test_plot_scatter_globe_color_propagates(globe_axes):
    fig, ax = globe_axes
    sc = plot_scatter_globe(ax, lons=[0], lats=[0], color="red")
    # PathCollection facecolor — single scatter point colored red
    fc = sc.get_facecolor()
    assert fc.shape[0] == 1
    # red is (1, 0, 0, 1)
    assert fc[0][0] == pytest.approx(1.0)
    assert fc[0][1] == pytest.approx(0.0)


def test_plot_scatter_globe_hemisphere_only_filters_antipodes(globe_axes):
    fig, ax = globe_axes
    # lat=0, lon=170 is antipodal-ish to center (0, 0); lon=10 is visible.
    # All-antipodal case: a single point at lon=180 (antipode of center)
    # should be filtered, returning None.
    sc = plot_scatter_globe(
        ax, lons=[180.0], lats=[0.0],
        hemisphere_only=True,
    )
    assert sc is None, "antipodal-only points should be filtered to None"


def test_plot_scatter_globe_hemisphere_only_keeps_visible(globe_axes):
    fig, ax = globe_axes
    # Visible point at center + antipodal point — should keep one
    sc = plot_scatter_globe(
        ax, lons=[0.0, 180.0], lats=[0.0, 0.0],
        hemisphere_only=True,
    )
    assert sc is not None
    assert sc.get_offsets().shape[0] == 1, \
        "hemisphere_only should drop the antipodal point"


def test_plot_scatter_globe_hemisphere_off_keeps_all(globe_axes):
    fig, ax = globe_axes
    sc = plot_scatter_globe(
        ax, lons=[0.0, 180.0, 90.0], lats=[0.0, 0.0, 45.0],
        hemisphere_only=False,
    )
    assert sc is not None
    assert sc.get_offsets().shape[0] == 3


# ============================================================
# plot_line_globe
# ============================================================

def test_plot_line_globe_returns_line2d_list(globe_axes):
    fig, ax = globe_axes
    lines = plot_line_globe(ax, lons=[0, 10, 20], lats=[0, 5, -5])
    assert isinstance(lines, list)
    assert len(lines) >= 1
    assert all(isinstance(line, Line2D) for line in lines)


def test_plot_line_globe_color_propagates(globe_axes):
    fig, ax = globe_axes
    lines = plot_line_globe(ax, lons=[0, 10], lats=[0, 5], color="blue")
    # First line's color should be blue
    color = lines[0].get_color()
    # color is "blue" or RGB tuple normalized to (0, 0, 1, ...)
    if isinstance(color, str):
        assert color == "blue"
    else:
        assert color[0] == pytest.approx(0.0)
        assert color[2] == pytest.approx(1.0)


# ============================================================
# plot_pcolormesh_globe
# ============================================================

def test_plot_pcolormesh_globe_returns_quadmesh(globe_axes):
    fig, ax = globe_axes
    lon_grid, lat_grid = np.meshgrid(
        np.linspace(-30, 30, 10), np.linspace(-30, 30, 10),
    )
    data = np.sin(np.radians(lon_grid))
    mesh = plot_pcolormesh_globe(ax, lon_grid, lat_grid, data)
    assert isinstance(mesh, QuadMesh), \
        f"expected QuadMesh, got {type(mesh).__name__}"
    # And it's a ScalarMappable so colorbars work
    assert isinstance(mesh, ScalarMappable)


# ============================================================
# plot_contour_globe
# ============================================================

def test_plot_contour_globe_returns_quadcontourset(globe_axes):
    fig, ax = globe_axes
    lon_grid, lat_grid = np.meshgrid(
        np.linspace(-30, 30, 10), np.linspace(-30, 30, 10),
    )
    data = np.sin(np.radians(lon_grid)) * np.cos(np.radians(lat_grid))
    cs = plot_contour_globe(ax, lon_grid, lat_grid, data, levels=5)
    assert isinstance(cs, QuadContourSet)


def test_plot_contour_globe_filled_returns_quadcontourset(globe_axes):
    fig, ax = globe_axes
    lon_grid, lat_grid = np.meshgrid(
        np.linspace(-30, 30, 10), np.linspace(-30, 30, 10),
    )
    data = np.sin(np.radians(lon_grid)) * np.cos(np.radians(lat_grid))
    cs = plot_contour_globe(ax, lon_grid, lat_grid, data, levels=5, filled=True)
    assert isinstance(cs, QuadContourSet)


# ============================================================
# imscatter / imscatter_rotated
# ============================================================

def _tiny_image():
    """Create a 4x4 RGB array so imscatter has something to embed."""
    return np.zeros((4, 4, 3), dtype=float)


def test_imscatter_returns_list_of_annotationbbox():
    fig, ax = plt.subplots()
    img = _tiny_image()
    arts = imscatter([1.0, 2.0, 3.0], [0.5, 0.5, 0.5], img, ax=ax)
    assert isinstance(arts, list)
    assert len(arts) == 3
    assert all(isinstance(a, AnnotationBbox) for a in arts)


def test_imscatter_rotated_returns_list_of_annotationbbox():
    fig, ax = plt.subplots()
    img = _tiny_image()
    arts = imscatter_rotated(
        [1.0, 2.0], [0.5, 0.5], img, rotations=[30.0, 60.0], ax=ax,
    )
    assert isinstance(arts, list)
    assert len(arts) == 2
    assert all(isinstance(a, AnnotationBbox) for a in arts)


def test_imscatter_rotated_positive_is_ccw_on_screen():
    """The ``rotations`` arg is CCW on screen. The documented static-icon
    aiming recipe (``rotations = aim_angles(...)['aim_angle'] - rest_angle``)
    relies on this sign, so lock it: a bright block on the RIGHT half, rotated
    +90 deg, must end up in the TOP half of the drawn stamp."""
    fig, ax = plt.subplots()
    img = np.zeros((21, 21, 4))
    img[..., 3] = 1.0                    # fully opaque support
    img[8:13, 13:20, :3] = 1.0           # bright block, right of center (+x)
    art = imscatter_rotated([0.0], [0.0], img, rotations=[90.0], ax=ax,
                            autoscale=False)[0]
    rot = art.offsetbox.get_data()       # the ndimage-rotated stamp
    lum = rot[..., :3].mean(2) * (rot[..., 3] > 0.3)
    half = rot.shape[0] // 2
    top, bottom = lum[:half].sum(), lum[half:].sum()   # row 0 = top on screen
    assert top > 3 * bottom


def test_imscatter_autoscale_true_preserves_preset_limits():
    # default autoscale=True must NOT clobber limits set beforehand
    fig, ax = plt.subplots()
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 10.0)
    imscatter([1.0, 2.0], [5.0, 5.0], _tiny_image(), ax=ax)
    assert ax.get_xlim() == (0.0, 10.0)
    assert ax.get_ylim() == (0.0, 10.0)


def test_imscatter_autoscale_false_leaves_view_untouched():
    fig, ax = plt.subplots()
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 10.0)
    imscatter([1.0, 2.0], [5.0, 5.0], _tiny_image(), ax=ax, autoscale=False)
    assert ax.get_xlim() == (0.0, 10.0)
    assert ax.get_ylim() == (0.0, 10.0)


def test_imscatter_autoscale_true_fits_on_fresh_axes():
    # with no preset limits, autoscale=True still expands to include the icons
    fig, ax = plt.subplots()
    imscatter([100.0, 200.0], [50.0, 50.0], _tiny_image(), ax=ax)
    assert ax.get_xlim()[1] > 10.0  # view grew toward the icon coords


def test_imscatter_rotated_clips_float_rgb_no_log_spam(caplog):
    import logging
    fig, ax = plt.subplots()
    # high-contrast float RGB so cubic-spline rotation overshoots [0, 1]
    img = np.zeros((16, 16, 3), dtype=float)
    img[4:12, 4:12, :] = 1.0
    with caplog.at_level(logging.WARNING, logger="matplotlib.image"):
        imscatter_rotated([1.0], [1.0], img, rotations=[30.0], ax=ax)
        fig.canvas.draw()
    assert not any("Clipping input data" in r.getMessage()
                   for r in caplog.records)


def test_imscatter_globe_accepts_n2_lonlat():
    ccrs = pytest.importorskip("cartopy.crs")
    from skyplothelper.globe.plotting import imscatter_globe
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=ccrs.Orthographic(0, 30))
    img = np.zeros((4, 4, 3), dtype=float)
    coords = np.array([[10.0, 20.0], [-30.0, 45.0]])  # (N, 2) lon/lat
    left, right = imscatter_globe(ax, coords, ccrs.PlateCarree(), 0.0, img,
                                  zoom=0.5)
    assert isinstance(left, list) and isinstance(right, list)
    assert len(left) + len(right) == 2


def test_imscatter_globe_bad_shape_raises():
    ccrs = pytest.importorskip("cartopy.crs")
    from skyplothelper.globe.plotting import imscatter_globe
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=ccrs.Orthographic(0, 30))
    img = np.zeros((4, 4, 3), dtype=float)
    with pytest.raises(ValueError, match="shape"):
        imscatter_globe(ax, np.zeros(5), ccrs.PlateCarree(), 0.0, img)


# ============================================================
# imscatter_rotated: aim_at / rest_angle / flip
# ============================================================

def _asym_icon():
    """Asymmetric RGBA icon, so a horizontal mirror is detectable."""
    img = np.zeros((9, 9, 4), dtype=float)
    img[2:5, 5:8, :] = 1.0
    img[..., 3] = 1.0
    return img


def _square_axes():
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    fig.canvas.draw()
    return fig, ax


def test_imscatter_rotated_aim_at_matches_aim_angles_recipe():
    """aim_at solves rotation = aim_angle - rest_angle, i.e. exactly the
    recipe aim_angles() documents for raster icons."""
    import skyplothelper as sph
    from skyplothelper.globe.plotting import _rotate_image_ccw
    fig, ax = _square_axes()
    icon, rest, target = _asym_icon(), 0.0, (90.0, 50.0)
    phi = sph.aim_angles(ax, (50, 50), target,
                         target_coords="data")["aim_angle"]
    arts = imscatter_rotated([50], [50], icon, aim_at=target,
                             rest_angle=rest, ax=ax)
    np.testing.assert_allclose(arts[0].offsetbox.get_data(),
                               _rotate_image_ccw(icon, phi - rest))


def test_imscatter_rotated_aim_at_target_coords_defaults_to_data():
    """aim_at tuples are data coords by default (unlike aim_angles), matching
    this function's own x/y."""
    fig, ax = _square_axes()
    icon = _asym_icon()
    a = imscatter_rotated([50], [50], icon, aim_at=(90.0, 50.0), ax=ax)
    b = imscatter_rotated([50], [50], icon, aim_at=(90.0, 50.0),
                          target_coords="data", ax=ax)
    np.testing.assert_allclose(a[0].offsetbox.get_data(),
                               b[0].offsetbox.get_data())


def test_imscatter_rotated_flip_auto_mirrors_far_side_target():
    """flip='auto' mirrors when the target and the boresight straddle
    vertical, so the icon leans toward the target instead of rolling past
    upright and reading upside-down."""
    fig, ax = _square_axes()
    icon = _asym_icon()
    # target at aim_angle=0 (to the right); dish boresight 130 -> straddles 90
    kw = dict(aim_at=(95.0, 50.0), rest_angle=130.0, ax=ax)
    auto = imscatter_rotated([50], [50], icon, **kw)[0].offsetbox.get_data()
    forced = imscatter_rotated([50], [50], icon, flip=True,
                               **kw)[0].offsetbox.get_data()
    unflipped = imscatter_rotated([50], [50], icon, flip=False,
                                  **kw)[0].offsetbox.get_data()
    np.testing.assert_allclose(auto, forced)   # 'auto' chose to flip here
    assert (auto.shape != unflipped.shape
            or not np.allclose(auto, unflipped))


def test_imscatter_rotated_aim_at_and_rotations_mutually_exclusive():
    fig, ax = _square_axes()
    with pytest.raises(ValueError, match="not both"):
        imscatter_rotated([50], [50], _asym_icon(), rotations=[0.0],
                          aim_at=(90.0, 50.0), ax=ax)


def test_imscatter_rotated_bad_flip_raises():
    fig, ax = _square_axes()
    with pytest.raises(ValueError, match="flip"):
        imscatter_rotated([50], [50], _asym_icon(), aim_at=(90.0, 50.0),
                          flip="maybe", ax=ax)


# ============================================================
# array-like zoom (the raster counterpart of scatter(s=...))
# ============================================================

@pytest.mark.parametrize("fn", [imscatter, imscatter_rotated])
def test_imscatter_array_zoom_sizes_each_icon(fn):
    fig, ax = plt.subplots()
    arts = fn([1.0, 2.0], [1.0, 2.0], _tiny_image(), zoom=[1.0, 3.0], ax=ax)
    assert [a.offsetbox.get_zoom() for a in arts] == [1.0, 3.0]


@pytest.mark.parametrize("fn", [imscatter, imscatter_rotated])
def test_imscatter_scalar_zoom_broadcasts(fn):
    fig, ax = plt.subplots()
    arts = fn([1.0, 2.0], [1.0, 2.0], _tiny_image(), zoom=2.5, ax=ax)
    assert [a.offsetbox.get_zoom() for a in arts] == [2.5, 2.5]


@pytest.mark.parametrize("fn", [imscatter, imscatter_rotated])
def test_imscatter_zoom_length_mismatch_raises(fn):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="zoom"):
        fn([1.0, 2.0], [1.0, 2.0], _tiny_image(), zoom=[1.0, 2.0, 3.0], ax=ax)


# ============================================================
# imscatter_globe: rest_angle
# ============================================================

_CLON = 0.0


@pytest.mark.parametrize("clon", [0.0, 45.0, 90.0, 180.0, 270.0, 315.0])
def test_imscatter_globe_hemisphere_split_west_is_left(clon):
    """A site west of the central meridian goes to the (unmirrored) left
    branch, and one east of it to the mirrored right branch — for *every*
    center longitude. The old wrapped-longitude comparison was unsatisfiable
    whenever the west-of-center interval crossed 0 (any clon < 180), which
    silently sent every icon down the right-hand branch."""
    ccrs = pytest.importorskip("cartopy.crs")
    from skyplothelper.globe.plotting import imscatter_globe
    icon = _asym_icon()
    coords = np.array([[clon - 40.0, 5.0], [clon + 40.0, 5.0]])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=ccrs.Orthographic(clon, 0))
    left, right = imscatter_globe(ax, coords, ccrs.PlateCarree(), clon, icon)
    assert len(left) == 1 and len(right) == 1


def test_imscatter_globe_rest_angle_default_matches_historical_rule():
    """rest_angle=45 (the default) reproduces the historical
    ``rotations = 90 - lat`` rule, so existing figures are unchanged."""
    ccrs = pytest.importorskip("cartopy.crs")
    from skyplothelper.globe.plotting import _rotate_image_ccw, imscatter_globe
    icon = _asym_icon()
    lats = [10.0, 55.0]
    coords = np.array([[-60.0, lats[0]], [-30.0, lats[1]]])  # west of clon=0
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=ccrs.Orthographic(_CLON, 0))
    left, _ = imscatter_globe(ax, coords, ccrs.PlateCarree(), _CLON, icon)
    assert len(left) == 2
    for art, lat in zip(left, lats):
        np.testing.assert_allclose(art.offsetbox.get_data(),
                                   _rotate_image_ccw(icon, 90.0 - lat))


def test_imscatter_globe_rest_angle_corrects_boresight():
    """A non-45 boresight (e.g. the bundled dish at 130) is corrected by
    subtracting (rest_angle - 45) from the upright rotation."""
    ccrs = pytest.importorskip("cartopy.crs")
    from skyplothelper.globe.plotting import _rotate_image_ccw, imscatter_globe
    icon, lat, rest = _asym_icon(), 10.0, 130.0
    coords = np.array([[-60.0, lat]])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=ccrs.Orthographic(_CLON, 0))
    left, _ = imscatter_globe(ax, coords, ccrs.PlateCarree(), _CLON, icon,
                              rest_angle=rest)
    np.testing.assert_allclose(
        left[0].offsetbox.get_data(),
        _rotate_image_ccw(icon, (90.0 - lat) - (rest - 45.0)))


def test_imscatter_globe_array_zoom_indexed_per_hemisphere():
    ccrs = pytest.importorskip("cartopy.crs")
    from skyplothelper.globe.plotting import imscatter_globe
    icon = _asym_icon()
    coords = np.array([[-60.0, 10.0], [-30.0, 55.0]])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=ccrs.Orthographic(_CLON, 0))
    left, _ = imscatter_globe(ax, coords, ccrs.PlateCarree(), _CLON, icon,
                              zoom=[2.0, 5.0])
    assert [a.offsetbox.get_zoom() for a in left] == [2.0, 5.0]
