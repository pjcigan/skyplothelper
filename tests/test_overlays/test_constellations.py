"""Smoke tests for skyplothelper.overlays.constellations."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

import skyplothelper.overlays.constellations as _constellations_mod
from skyplothelper.overlays.constellations import (
    _CONSTELLATION_CENTERS,
    _CONSTELLATION_NAMES,
    _load_constellation_lines,
    add_constellation_boundaries,
    add_constellation_labels,
    add_constellation_lines,
    add_constellation_polygon,
    list_constellations,
)
from skyplothelper.wcs_frame import make_wcs_frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_load_constellation_lines_caches_bundled_data():
    # Regression: the bundled-data path reassigned data_file before the
    # `if data_file is None` cache-store check, so the in-memory cache was
    # never populated and the npz re-parsed on every call. It must now cache.
    _constellations_mod._constellation_lines_cache = None
    first = _load_constellation_lines()
    assert _constellations_mod._constellation_lines_cache is not None
    second = _load_constellation_lines()
    assert first is second  # served from cache, not re-loaded


def test_constellation_dicts_consistent():
    """All abbreviations in CENTERS should also have a NAME."""
    centers_keys = set(_CONSTELLATION_CENTERS.keys())
    names_keys = set(_CONSTELLATION_NAMES.keys())
    # Some entries may differ slightly; both should be 88 IAU constellations.
    assert len(centers_keys) == 88
    assert len(names_keys) == 88


def test_list_constellations_runs(capsys):
    list_constellations()
    out = capsys.readouterr().out
    assert "Andromeda" in out or "AND" in out


def test_add_constellation_boundaries_smoke():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    add_constellation_boundaries(ax)
    fig.canvas.draw()


def test_add_constellation_labels_smoke():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    add_constellation_labels(ax, labels="abbr")
    fig.canvas.draw()


# ===== Stroke kwarg uniformity =====

def test_add_constellation_labels_stroke_off_by_default():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    texts = add_constellation_labels(ax, labels="abbr")
    assert all(not t.get_path_effects() for t in texts)


def test_add_constellation_labels_stroke_on():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    texts = add_constellation_labels(ax, labels="abbr",
                                       stroke_color="white", stroke_lw=2.0)
    assert all(len(t.get_path_effects()) == 1 for t in texts)


def test_add_constellation_boundaries_stroke_on():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    lines = add_constellation_boundaries(ax, stroke_color="k", stroke_lw=2.0)
    # At least one segment, and each line has the stroke applied.
    assert len(lines) > 0
    assert all(len(ln.get_path_effects()) == 1 for ln in lines[:5])


def test_add_constellation_lines_stroke_on():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    lines = add_constellation_lines(ax, constellations=["ORI"],
                                     stroke_color="white", stroke_lw=2.5)
    assert len(lines) > 0
    assert all(len(ln.get_path_effects()) == 1 for ln in lines)


def test_add_constellation_polygon_stroke_on():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    patches = add_constellation_polygon(ax, "UMI",
                                          stroke_color="k", stroke_lw=2.5)
    assert len(patches) > 0
    assert all(len(p.get_path_effects()) == 1 for p in patches)


# ============================================================
# #15 constellation-tutorial fixes: frame-crossing fill, non-ICRS
# frames, globe far-side label culling
# ============================================================

def _poly_area(patches):
    import numpy as np
    tot = 0.0
    for p in patches:
        v = p.get_path().vertices
        if len(v) >= 3:
            x, y = v[:, 0], v[:, 1]
            tot += 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return tot


def test_constellation_polygon_not_inverted_on_frame_crossing_field():
    """Orion (dec -11..+23) exceeds a 28 deg TAN field, so its boundary exits
    the frame. The default fill must be the interior, not the complement — the
    d3 clip inverted it; auto now routes to project_shape on bounded fields."""
    fig = plt.figure(figsize=(6, 6))
    ax = make_wcs_frame(111, projection="TAN", center=(83, 2), fov_deg=28,
                        fig=fig)
    fig.canvas.draw()
    interior = _poly_area(add_constellation_polygon(ax, "Ori"))
    plt.close(fig)
    fig = plt.figure(figsize=(6, 6))
    ax = make_wcs_frame(111, projection="TAN", center=(83, 2), fov_deg=28,
                        fig=fig)
    fig.canvas.draw()
    inverted = _poly_area(add_constellation_polygon(ax, "Ori",
                                                    clip="d3"))
    plt.close(fig)
    # The correct interior fill (Orion taller than the frame) is much larger
    # than the inverted "outside Orion" region.
    assert interior > 1.5 * inverted


def test_constellation_polygon_lands_correctly_on_galactic_frame():
    """RA/Dec corner data must re-express into the axes' native frame, so a
    galactic-frame axes places Orion in the galactic anticenter (l~200, b~-15),
    not at the raw RA/Dec read as l/b (l=83, b=+5)."""
    import numpy as np
    fig = plt.figure(figsize=(10, 5))
    ax = make_wcs_frame(111, projection="AIT", center=0, frame="galactic",
                        fig=fig)
    fig.canvas.draw()
    patches = add_constellation_polygon(ax, "ORI")
    verts = np.vstack([p.get_path().vertices for p in patches
                       if len(p.get_path().vertices) >= 3])
    cx, cy = verts.mean(0)
    gl, gb = ax.wcs.wcs_pix2world([cx], [cy], 0)
    assert 180 < gl[0] < 225        # galactic longitude of Orion, not l=83
    assert -30 < gb[0] < 0          # southern galactic lat, not b=+5


def test_constellation_lines_land_in_native_frame_on_galactic_axes():
    """On a galactic frame the lines are converted to native (l, b) and drawn
    through 'world', so an Orion asterism sits at galactic l~200 (Orion), not at
    the raw RA (83) read as l."""
    import numpy as np
    fig = plt.figure(figsize=(10, 5))
    ax = make_wcs_frame(111, projection="AIT", center=0, frame="galactic",
                        fig=fig)
    fig.canvas.draw()
    lines = add_constellation_lines(ax, constellations=["ORI"])
    assert lines
    lon = np.concatenate([np.asarray(ln.get_xdata(), float) for ln in lines])
    lon = lon[np.isfinite(lon)]
    assert 170.0 < float(np.median(lon % 360.0)) < 220.0   # galactic Orion


def test_constellation_boundaries_no_seam_streak_on_galactic_frame():
    """The antimeridian split runs in NATIVE coords, so boundary segments
    straddling the galactic seam (l=±180 on a center=0 frame) don't sweep
    across the map interior. Regression for Bug D — the equatorial-RA split
    left southern segments streaking on non-ICRS frames."""
    import numpy as np
    fig = plt.figure(figsize=(11, 5))
    ax = make_wcs_frame(111, projection="AIT", center=0, frame="galactic",
                        fig=fig)
    fig.canvas.draw()
    lines = add_constellation_boundaries(ax)
    fig.canvas.draw()
    lc = lines[0]                       # single LineCollection of all chords
    tr = lc.get_transform()
    worst = 0.0
    for seg in lc.get_segments():       # each chord's endpoints, per-segment
        disp = tr.transform(seg)
        disp = disp[np.isfinite(disp).all(1)]
        if len(disp) >= 2:
            worst = max(worst, float(np.abs(np.diff(disp[:, 0])).max()))
    assert worst < 0.5 * fig.bbox.width    # no boundary sweeps across the map
    plt.close(fig)


def test_constellation_labels_cull_far_side_on_globe(recwarn):
    """Far-side label centers on a globe are unprojectable (NaN); they must be
    skipped rather than emit 'posx and posy should be finite values'."""
    import numpy as np

    from skyplothelper.globe.frame import make_globe_frame
    fig = plt.figure(figsize=(6, 6))
    ax = make_globe_frame(111, center_LONdeg=83, center_LATdeg=10)
    fig.canvas.draw()
    texts = add_constellation_labels(ax)
    fig.canvas.draw()
    # Fewer than all 88 (far side dropped), and every drawn label is finite.
    assert 0 < len(texts) < 88
    tr = ax.get_transform("icrs")
    for t in texts:
        x, y = t.get_position()
        px, py = tr.transform((x, y))
        assert np.isfinite(px) and np.isfinite(py)
    plt.close(fig)


def test_boundary_densification_is_tunable():
    """step_deg was fixed at 0.5 — the loader already cached on it, but the
    drawer never exposed it."""
    import skyplothelper as sph
    fig, ax = sph.allsky_figure(projection="AIT")
    coarse = sph.add_constellation_boundaries(ax, step_deg=5.0)
    fine = sph.add_constellation_boundaries(ax, step_deg=0.5)

    def nsegs(arts):
        return sum(len(a.get_segments())
                   for a in arts if hasattr(a, "get_segments"))

    assert nsegs(fine) > nsegs(coarse) * 3
    plt.close(fig)
