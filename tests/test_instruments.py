"""Tests for procedural instrument markers (add_antenna_marker,
add_telescope_marker, add_dome_marker)."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea  # noqa: E402
from matplotlib.patches import Circle, PathPatch, Polygon  # noqa: E402
from matplotlib.path import Path  # noqa: E402

import skyplothelper as sph  # noqa: E402
from skyplothelper.overlays.instruments import (  # noqa: E402
    add_antenna_marker,
    add_dome_marker,
    add_telescope_marker,
    aim_angles,
)


@pytest.fixture
def plain_axes():
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    yield fig, ax
    plt.close(fig)


@pytest.fixture
def wcs_axes():
    fig = plt.figure(figsize=(5, 5))
    ax = sph.make_wcs_frame(111, projection="CAR", center=0, fig=fig)
    fig.canvas.draw()
    yield fig, ax
    plt.close(fig)


def _drawn_patches(anchor):
    """Return the patches inside an AnchoredOffsetbox's DrawingArea."""
    assert isinstance(anchor, AnchoredOffsetbox)
    da = anchor.get_child()
    assert isinstance(da, DrawingArea)
    return list(da.get_children())


# ============================================================
# add_antenna_marker
# ============================================================

def test_antenna_marker_returns_anchored_offsetbox(plain_axes):
    fig, ax = plain_axes
    out = add_antenna_marker(ax, (50, 50), dish_elev=45)
    assert isinstance(out, AnchoredOffsetbox)
    patches = _drawn_patches(out)
    # base (Polygon) + pole (Polygon) + dish (PathPatch closed bezier)
    # + 2 struts (PathPatch x2) + focal dot (Circle) = 6 patches
    assert len(patches) == 6
    assert sum(isinstance(p, Polygon) for p in patches) == 2
    assert sum(isinstance(p, PathPatch) for p in patches) == 3
    assert sum(isinstance(p, Circle) for p in patches) == 1


def test_antenna_marker_dish_back_stays_near_pivot(plain_axes):
    """With the pivot just inside the back of the bowl, the bezier
    curve's actual midpoint (the deepest visible part of the bowl)
    should be close to a consistent position across dish_elev values
    — proving the bowl rotates around its own back, not its rim."""
    fig, ax = plain_axes
    a0 = add_antenna_marker(ax, (50, 50), dish_elev=0, rotation=0)
    a90 = add_antenna_marker(ax, (50, 50), dish_elev=90, rotation=0)

    def _bezier_midpoint(anchor):
        """Compute the t=0.5 point of the dish bezier from its
        Path vertices."""
        for p in _drawn_patches(anchor):
            if isinstance(p, PathPatch):
                path = p.get_path()
                codes = list(path.codes) if path.codes is not None else []
                if (len(codes) == 4
                        and codes[1] == path.CURVE3
                        and codes[3] == path.CLOSEPOLY):
                    rim_top = path.vertices[0]
                    vertex = path.vertices[1]
                    rim_bot = path.vertices[2]
                    return (0.25 * rim_top + 0.5 * vertex
                             + 0.25 * rim_bot)
        raise AssertionError("dish path not found")

    mid_0 = _bezier_midpoint(a0)
    mid_90 = _bezier_midpoint(a90)
    # Both should land near the same point (within ~1 px) since the
    # pivot is inside the bowl just forward of the back.
    np.testing.assert_allclose(mid_0, mid_90, atol=1.5)


def test_antenna_marker_dish_elev_moves_focus(plain_axes):
    fig, ax = plain_axes
    a0 = add_antenna_marker(ax, (50, 50), dish_elev=0)
    a90 = add_antenna_marker(ax, (50, 50), dish_elev=90)
    f0 = np.array(next(p for p in _drawn_patches(a0)
                        if isinstance(p, Circle)).center)
    f90 = np.array(next(p for p in _drawn_patches(a90)
                         if isinstance(p, Circle)).center)
    assert not np.allclose(f0, f90)


def test_antenna_marker_world_coord_on_wcs_axes(wcs_axes):
    fig, ax = wcs_axes
    out = add_antenna_marker(ax, (180.0, -30.0), coord_type="world",
                              dish_elev=45, size=20)
    assert isinstance(out, AnchoredOffsetbox)


# ============================================================
# add_telescope_marker
# ============================================================

def test_telescope_marker_returns_anchored_offsetbox(plain_axes):
    fig, ax = plain_axes
    out = add_telescope_marker(ax, (50, 50), tube_elev=45)
    assert isinstance(out, AnchoredOffsetbox)
    patches = _drawn_patches(out)
    assert len(patches) == 4
    assert sum(isinstance(p, Polygon) for p in patches) == 2
    assert sum(isinstance(p, PathPatch) for p in patches) == 2


def test_telescope_marker_tube_elev_changes_polygon_vertices(plain_axes):
    fig, ax = plain_axes
    t0 = add_telescope_marker(ax, (30, 30), tube_elev=0)
    t90 = add_telescope_marker(ax, (60, 60), tube_elev=90)
    polys_0 = [p for p in _drawn_patches(t0) if isinstance(p, Polygon)]
    polys_90 = [p for p in _drawn_patches(t90) if isinstance(p, Polygon)]
    tube_0 = max(polys_0, key=lambda p: p.get_xy().std())
    tube_90 = max(polys_90, key=lambda p: p.get_xy().std())
    assert not np.allclose(tube_0.get_xy(), tube_90.get_xy())


def test_telescope_marker_world_coord_on_wcs_axes(wcs_axes):
    fig, ax = wcs_axes
    out = add_telescope_marker(ax, (100.0, 20.0), coord_type="world",
                                tube_elev=30)
    assert isinstance(out, AnchoredOffsetbox)


# ============================================================
# add_dome_marker
# ============================================================

def test_dome_marker_at_zero_azim_returns_three_patches(plain_axes):
    """At slit_azim=0 the slit is full-width and visible: base
    (Polygon) + dome (PathPatch) + slit (PathPatch) = 3 patches."""
    fig, ax = plain_axes
    out = add_dome_marker(ax, (50, 50), slit_azim=0)
    patches = _drawn_patches(out)
    assert len(patches) == 3
    assert sum(isinstance(p, Polygon) for p in patches) == 1
    assert sum(isinstance(p, PathPatch) for p in patches) == 2


def test_dome_marker_at_back_azim_drops_slit(plain_axes):
    """When slit_azim is on the back side (|cos azim| ≈ 0) the slit
    isn't drawn — only base + dome remain."""
    fig, ax = plain_axes
    for back_azim in (100, 180, -120):
        out = add_dome_marker(ax, (50, 50), slit_azim=back_azim)
        patches = _drawn_patches(out)
        assert len(patches) == 2, (
            f"slit_azim={back_azim} should hide the slit, got "
            f"{len(patches)} patches")


def test_dome_marker_slit_shifts_off_center_with_azim(plain_axes):
    """As slit_azim rotates away from 0, the slit's base center
    should shift horizontally (R·sin azim along the dome equator)."""
    fig, ax = plain_axes

    def _slit_base_x(anchor):
        for p in _drawn_patches(anchor):
            if isinstance(p, PathPatch):
                path = p.get_path()
                codes = list(path.codes) if path.codes is not None else []
                if (len(codes) >= 6
                        and codes[0] == path.MOVETO
                        and codes[-1] == path.CLOSEPOLY
                        and codes[1] == path.CURVE3):
                    base_left = path.vertices[0]
                    base_right = path.vertices[4]
                    return 0.5 * (base_left[0] + base_right[0])
        raise AssertionError("slit path not found")

    x_0 = _slit_base_x(add_dome_marker(ax, (50, 50), slit_azim=0))
    x_60 = _slit_base_x(add_dome_marker(ax, (50, 50), slit_azim=60))
    assert abs(x_60 - x_0) > 1.0


def test_dome_marker_slit_base_width_shrinks_with_azim(plain_axes):
    fig, ax = plain_axes

    def _slit_base_width(anchor):
        for p in _drawn_patches(anchor):
            if isinstance(p, PathPatch):
                path = p.get_path()
                codes = list(path.codes) if path.codes is not None else []
                if (len(codes) >= 6
                        and codes[1] == path.CURVE3
                        and codes[-1] == path.CLOSEPOLY):
                    return float(np.linalg.norm(
                        path.vertices[4] - path.vertices[0]))
        raise AssertionError("slit path not found")

    w_0 = _slit_base_width(add_dome_marker(ax, (50, 50), slit_azim=0))
    w_60 = _slit_base_width(add_dome_marker(ax, (50, 50), slit_azim=60))
    # cos(60°) = 0.5 → slit base is half as wide.
    assert w_60 == pytest.approx(0.5 * w_0, rel=0.05)


def test_dome_marker_world_coord_on_wcs_axes(wcs_axes):
    fig, ax = wcs_axes
    out = add_dome_marker(ax, (45.0, 60.0), coord_type="world",
                           slit_azim=20)
    assert isinstance(out, AnchoredOffsetbox)


# ============================================================
# Shared behavior
# ============================================================

@pytest.mark.parametrize("marker_fn,kw", [
    (add_antenna_marker, dict(dish_elev=45)),
    (add_telescope_marker, dict(tube_elev=30)),
    (add_dome_marker, dict(slit_azim=15)),
])
def test_marker_stroke_color_applies_path_effects(plain_axes, marker_fn, kw):
    fig, ax = plain_axes
    out = marker_fn(ax, (50, 50), stroke_color="yellow", stroke_lw=3.0,
                    **kw)
    patches = _drawn_patches(out)
    assert any(p.get_path_effects() for p in patches)


@pytest.mark.parametrize("marker_fn,kw", [
    (add_antenna_marker, dict(dish_elev=45)),
    (add_telescope_marker, dict(tube_elev=30)),
    (add_dome_marker, dict(slit_azim=15)),
])
def test_marker_zorder_propagates(plain_axes, marker_fn, kw):
    fig, ax = plain_axes
    out = marker_fn(ax, (50, 50), zorder=42, **kw)
    assert out.get_zorder() == 42


@pytest.mark.parametrize("marker_fn,kw", [
    (add_antenna_marker, dict(dish_elev=45)),
    (add_telescope_marker, dict(tube_elev=30)),
    (add_dome_marker, dict(slit_azim=15)),
])
def test_marker_remove_takes_it_off(plain_axes, marker_fn, kw):
    fig, ax = plain_axes
    n_before = len(ax.artists)
    out = marker_fn(ax, (50, 50), **kw)
    assert len(ax.artists) == n_before + 1
    out.remove()
    assert len(ax.artists) == n_before


def test_markers_exported_from_top_level():
    """All three markers must be accessible via sph.<name>."""
    assert sph.add_antenna_marker is add_antenna_marker
    assert sph.add_telescope_marker is add_telescope_marker
    assert sph.add_dome_marker is add_dome_marker


# ============================================================
# Marker aiming — aim_angles + aim_at=
# ============================================================

def _drawn_bowl_direction(anchor):
    """Screen angle (deg CCW) the drawn antenna dish bowl opens toward,
    read from the real drawn geometry: dish back-vertex -> focal dot."""
    da = anchor.get_child()
    dish = next(p for p in da.get_children() if isinstance(p, PathPatch)
                and p.get_path().codes is not None
                and Path.CURVE3 in p.get_path().codes)
    focus = next(p for p in da.get_children() if isinstance(p, Circle))
    vertex = dish.get_path().vertices[1]        # the CURVE3 control = bowl back
    d = np.asarray(focus.get_center()) - np.asarray(vertex)
    return float(np.degrees(np.arctan2(d[1], d[0]))) % 360.0


def _drawn_tube_direction(anchor):
    """Screen angle the drawn telescope tube points (eyepiece -> objective)."""
    da = anchor.get_child()
    # tube is the larger-area Polygon (vs the small eyepiece bump)
    polys = [p for p in da.get_children() if isinstance(p, Polygon)]
    tube = max(polys, key=lambda p: _poly_area(p.get_xy()))
    v = tube.get_xy()[:4]
    obj_mid = (v[1] + v[2]) / 2.0
    eye_mid = (v[0] + v[3]) / 2.0
    d = obj_mid - eye_mid
    return float(np.degrees(np.arctan2(d[1], d[0]))) % 360.0


def _poly_area(v):
    v = np.asarray(v)
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _angdiff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


@pytest.fixture
def square_axes():
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    fig.canvas.draw()
    yield fig, ax
    plt.close(fig)


@pytest.mark.parametrize("target", [(90, 50), (50, 90), (20, 20), (80, 15)])
def test_aim_angles_aimed_antenna_bowl_points_at_target(square_axes, target):
    """The DRAWN dish bowl opens toward the target (aimed mode)."""
    fig, ax = square_axes
    m = ax.transData.transform((50, 50))
    t = ax.transData.transform(target)
    phi = np.degrees(np.arctan2(t[1] - m[1], t[0] - m[0])) % 360.0
    r = sph.aim_angles(ax, (50, 50), target, marker="antenna",
                       target_coords="data")
    out = add_antenna_marker(ax, (50, 50), dish_elev=r["dish_elev"],
                             rotation=r["rotation"], size=40)
    fig.canvas.draw()
    assert _angdiff(_drawn_bowl_direction(out), phi) < 0.5


@pytest.mark.parametrize("target", [(90, 50), (50, 90), (20, 20), (80, 15)])
def test_aim_at_telescope_tube_points_at_target(square_axes, target):
    """aim_at= on the telescope makes the DRAWN tube point at the target."""
    fig, ax = square_axes
    m = ax.transData.transform((50, 50))
    t = ax.transData.transform(target)
    phi = np.degrees(np.arctan2(t[1] - m[1], t[0] - m[0])) % 360.0
    out = add_telescope_marker(ax, (50, 50), aim_at=target,
                               target_coords="data", size=40)
    fig.canvas.draw()
    assert _angdiff(_drawn_tube_direction(out), phi) < 0.5


def test_aim_angles_dict_keys_per_marker(square_axes):
    fig, ax = square_axes
    ra = sph.aim_angles(ax, (50, 50), (80, 80), marker="antenna",
                        target_coords="data")
    assert set(ra) == {"rotation", "dish_elev", "aim_angle", "radial_angle"}
    assert ra["radial_angle"] is None            # aimed mode uses no center
    rt = sph.aim_angles(ax, (50, 50), (80, 80), marker="telescope",
                        target_coords="data")
    assert "tube_elev" in rt


def test_aim_planted_base_upright_dish_on_target(square_axes):
    """Planted: base points along the outward radial (rho); bowl at target."""
    fig, ax = square_axes
    center, site, target = (50, 50), (50, 85), (90, 85)
    m = ax.transData.transform(site)
    g = ax.transData.transform(center)
    t = ax.transData.transform(target)
    rho = np.degrees(np.arctan2(m[1] - g[1], m[0] - g[0])) % 360.0
    phi = np.degrees(np.arctan2(t[1] - m[1], t[0] - m[0])) % 360.0
    r = sph.aim_angles(ax, site, target, marker="antenna", mode="planted",
                       globe_center=center, target_coords="data")
    out = add_antenna_marker(ax, site, dish_elev=r["dish_elev"],
                             rotation=r["rotation"], size=40)
    fig.canvas.draw()
    assert _angdiff((90 + r["rotation"]) % 360, rho) < 0.5   # base upright
    assert _angdiff(_drawn_bowl_direction(out), phi) < 0.5   # bowl on target
    assert _angdiff(r["radial_angle"], rho) < 0.5


def test_aim_planted_max_tilt_clamps_below_horizon(square_axes):
    """With flip=False, a below-horizon target is clamped to max_tilt."""
    fig, ax = square_axes
    # site at top, center below => local up = +90 deg; target straight down.
    r = sph.aim_angles(ax, (50, 85), (50, 20), marker="antenna",
                       mode="planted", globe_center=(50, 50),
                       target_coords="data", max_tilt=80.0, flip=False)
    out = add_antenna_marker(ax, (50, 85), dish_elev=r["dish_elev"],
                             rotation=r["rotation"], size=40)
    fig.canvas.draw()
    bowl = _drawn_bowl_direction(out)
    # up is 90; clamped 80 deg toward the target => 170 (not 270 = straight down)
    assert _angdiff(bowl, 170.0) < 1.0


def test_aim_planted_flip_points_dish_at_behind_horizon_target(square_axes):
    """flip='auto' (default) lets the dish reach a behind-horizon target while
    keeping the pier on the local vertical (base flipped 180 deg)."""
    fig, ax = square_axes
    site, center, target = (50, 85), (50, 50), (50, 20)
    m = ax.transData.transform(site)
    t = ax.transData.transform(target)
    phi = np.degrees(np.arctan2(t[1] - m[1], t[0] - m[0])) % 360.0  # = 270
    r = sph.aim_angles(ax, site, target, marker="antenna", mode="planted",
                       globe_center=center, target_coords="data")  # flip auto
    out = add_antenna_marker(ax, site, dish_elev=r["dish_elev"],
                             rotation=r["rotation"], size=40)
    fig.canvas.draw()
    # dish reaches the target (no clamp), and the base is flipped: base-up is
    # the inward normal (rho + 180 = 270), not the outward normal (90).
    assert _angdiff(_drawn_bowl_direction(out), phi) < 0.5
    assert _angdiff((90 + r["rotation"]) % 360, 270.0) < 0.5


def test_aim_planted_flip_false_keeps_base_outward(square_axes):
    """flip=False never flips: the base always points along the outward
    normal even when that tangles the dish across the mount."""
    fig, ax = square_axes
    r = sph.aim_angles(ax, (50, 85), (50, 20), marker="antenna",
                       mode="planted", globe_center=(50, 50),
                       target_coords="data", flip=False, max_tilt=180.0)
    assert _angdiff((90 + r["rotation"]) % 360, 90.0) < 0.5  # outward up

    r2 = sph.aim_angles(ax, (50, 85), (50, 20), marker="antenna",
                        mode="planted", globe_center=(50, 50),
                        target_coords="data", flip=True)
    assert _angdiff((90 + r2["rotation"]) % 360, 270.0) < 0.5  # forced flip


def test_aim_angles_invalid_flip(square_axes):
    fig, ax = square_axes
    with pytest.raises(ValueError, match="flip"):
        sph.aim_angles(ax, (50, 85), (90, 85), mode="planted",
                       globe_center=(50, 50), target_coords="data",
                       flip="sometimes")


def test_aim_at_overrides_explicit_elev_and_rotation(square_axes):
    """aim_at wins over any dish_elev/rotation passed alongside it."""
    fig, ax = square_axes
    target = (90, 50)
    ref = add_antenna_marker(ax, (50, 50), aim_at=target,
                             target_coords="data", size=40)
    override = add_antenna_marker(ax, (50, 50), aim_at=target, dish_elev=0,
                                  rotation=123, target_coords="data", size=40)
    fig.canvas.draw()
    assert _angdiff(_drawn_bowl_direction(ref),
                    _drawn_bowl_direction(override)) < 1e-6


def test_aim_angles_planted_requires_globe_center(square_axes):
    fig, ax = square_axes
    with pytest.raises(ValueError, match="globe_center"):
        sph.aim_angles(ax, (50, 85), (90, 85), mode="planted",
                       target_coords="data")


def test_aim_angles_invalid_mode_and_target_coords(square_axes):
    fig, ax = square_axes
    with pytest.raises(ValueError, match="aim_mode"):
        sph.aim_angles(ax, (50, 50), (80, 80), mode="sideways",
                       target_coords="data")
    with pytest.raises(ValueError, match="target_coords"):
        sph.aim_angles(ax, (50, 50), (80, 80), target_coords="galactic")


def test_aim_angles_exported():
    assert sph.aim_angles is aim_angles


# --- rest_elev: aiming an instrument that does not rest at the zenith -------

def _aim_axes():
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    fig.canvas.draw()
    return ax


_AIM_TARGETS = [(400, 300), (250, 480), (80, 120), (460, 260), (120, 460)]
_REST_POSES = [90.0, 75.0, 45.0, 30.0, 0.0, 120.0]


@pytest.mark.parametrize("target", _AIM_TARGETS)
def test_rest_elev_90_reproduces_the_previous_behavior(target):
    """The solver assumed a zenith rest pose outright, so 90 must be a no-op.
    That is what makes this a generalization rather than a change."""
    ax = _aim_axes()
    a = sph.aim_angles(ax, (250, 250), target, marker="antenna",
                       coord_type="pixel")
    b = sph.aim_angles(ax, (250, 250), target, marker="antenna",
                       coord_type="pixel", rest_elev=90.0)
    plt.close("all")
    assert a["rotation"] == pytest.approx(b["rotation"], abs=1e-12)
    assert a["dish_elev"] == pytest.approx(b["dish_elev"], abs=1e-12)


@pytest.mark.parametrize("marker", ["antenna", "telescope"])
@pytest.mark.parametrize("rest_elev", _REST_POSES)
def test_collecting_element_lands_on_target_for_any_rest_pose(marker,
                                                              rest_elev):
    """The point of the solve: the dish bowl / tube objective must end up
    pointing at the target, whatever the rest pose.

    The two markers do NOT share a rotation convention — the antenna's bowl
    counts ``rotation`` twice, the telescope's tube once — so this is checked
    per marker rather than assumed to follow from the rotation alone.
    """
    from skyplothelper.overlays.instruments import _wrap180
    ax = _aim_axes()
    for target in _AIM_TARGETS:
        r = sph.aim_angles(ax, (250, 250), target, marker=marker,
                           coord_type="pixel", rest_elev=rest_elev)
        if marker == "antenna":
            pointing = r["dish_elev"] + 2.0 * r["rotation"]
        else:
            pointing = r["tube_elev"] + r["rotation"]
        assert _wrap180(pointing - r["aim_angle"]) == pytest.approx(0.0,
                                                                    abs=1e-9)
    plt.close("all")


@pytest.mark.parametrize("rest_elev", _REST_POSES)
def test_rest_elev_matches_the_reference_formulas(rest_elev):
    """rotation = aim - rest_elev; dish_elev = 2*rest_elev - aim."""
    from skyplothelper.overlays.instruments import _wrap180
    ax = _aim_axes()
    r = sph.aim_angles(ax, (250, 250), (400, 300), marker="antenna",
                       coord_type="pixel", rest_elev=rest_elev)
    plt.close("all")
    aim = r["aim_angle"]
    assert _wrap180(r["rotation"] - (aim - rest_elev)) == pytest.approx(
        0.0, abs=1e-9)
    assert _wrap180(r["dish_elev"] - (2.0 * rest_elev - aim)) == pytest.approx(
        0.0, abs=1e-9)


def test_rest_elev_honored_in_planted_mode():
    """'planted' keeps the pier on the local vertical and tilts only the dish,
    so it takes a different branch to the rest pose than 'aimed' does."""
    from skyplothelper.overlays.instruments import _wrap180
    ax = _aim_axes()
    r = sph.aim_angles(ax, (250, 250), (400, 300), marker="antenna",
                       coord_type="pixel", mode="planted",
                       globe_center=(250, 150), rest_elev=45.0)
    plt.close("all")
    pointing = r["dish_elev"] + 2.0 * r["rotation"]
    assert _wrap180(pointing - r["aim_angle"]) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("adder", ["add_antenna_marker",
                                   "add_telescope_marker"])
def test_rest_elev_reaches_the_marker_wrappers(adder):
    """aim_at= must thread rest_elev through, not just aim_angles()."""
    ax = _aim_axes()
    a = getattr(sph, adder)(ax, (250, 250), aim_at=(400, 300))
    b = getattr(sph, adder)(ax, (250, 250), aim_at=(400, 300), rest_elev=40.0)
    verts_a = [p.get_path().vertices.copy() for p in a.get_child().get_children()]
    verts_b = [p.get_path().vertices.copy() for p in b.get_child().get_children()]
    plt.close("all")
    assert any(va.shape != vb.shape or not np.allclose(va, vb)
               for va, vb in zip(verts_a, verts_b))


def test_instrument_marker_stroke_reaches_label():
    """A stroked antenna/telescope/dome marker strokes its label too, not only
    the sprite outline (12.15 unification fold-in)."""
    import matplotlib.pyplot as plt

    import skyplothelper as sph
    for fn in (sph.add_antenna_marker, sph.add_telescope_marker,
               sph.add_dome_marker):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        box = fn(ax, (5, 5), coord_type="pixel", label="site",
                 stroke_color="k", stroke_lw=3)
        assert box.label_artist is not None
        assert box.label_artist.get_path_effects(), f"{fn.__name__} label unstroked"
        plt.close(fig)


def test_instrument_marker_no_stroke_leaves_label_plain():
    import matplotlib.pyplot as plt

    import skyplothelper as sph
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    box = sph.add_antenna_marker(ax, (5, 5), coord_type="pixel", label="site")
    assert not box.label_artist.get_path_effects()
    plt.close(fig)


def test_markers_base_anchored_feet_align():
    """All three markers place their base foot at the anchor coord, so a row
    of markers at one y lands their feet on one ground line."""
    import matplotlib.pyplot as plt

    import skyplothelper as sph
    fig, ax = plt.subplots()
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 5)
    boxes = [fn(ax, (x, 3.0), coord_type="pixel", size=48)
             for x, fn in ((0.5, sph.add_antenna_marker),
                           (1.5, sph.add_telescope_marker),
                           (2.5, sph.add_dome_marker))]
    for box, x in zip(boxes, (0.5, 1.5, 2.5)):
        bx, by = box.anchors.base
        assert abs(bx - x) < 1e-6 and abs(by - 3.0) < 1e-6
    plt.close(fig)


def test_marker_anchors_pivot_above_base():
    """The pivot (hinge/mount/dome-center) sits above the base foot."""
    import matplotlib.pyplot as plt

    import skyplothelper as sph
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 5)
    for fn in (sph.add_antenna_marker, sph.add_telescope_marker,
               sph.add_dome_marker):
        a = fn(ax, (0.5, 2.0), coord_type="pixel", size=48).anchors
        assert a.pivot[1] > a.base[1]                 # hinge above foot
        # pivot_offset is a stable display-point offset, up the pier
        assert a.pivot_offset[1] > 0
    plt.close(fig)


def test_marker_anchors_sight_line_origin_beyond_pivot():
    """sight_line_origin nudges past the pivot along the aim direction."""
    import matplotlib.pyplot as plt
    import numpy as np

    import skyplothelper as sph
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 5)
    a = sph.add_antenna_marker(ax, (0.5, 2.0), coord_type="pixel",
                               size=48).anchors
    # aim straight up (90 deg): origin should sit above the pivot
    origin = a.sight_line_origin(90.0, into_bowl=0.55)
    assert origin[1] > a.pivot[1]
    # display-coord variant is a 2-tuple of finite floats
    od = a.sight_line_origin(90.0, coords="display")
    assert len(od) == 2 and all(np.isfinite(od))
    plt.close(fig)


def test_marker_anchors_is_public():
    import skyplothelper as sph
    from skyplothelper.overlays.instruments import MarkerAnchors
    assert sph.MarkerAnchors is MarkerAnchors
