"""Seam / frame-edge sweep: overlays must survive an ARBITRARY frame center.

This class of bug has bitten repeatedly: a helper assumes a particular center
longitude (often 0 or 180), renders correctly there, and streaks a line across
the entire map at other centers. Inspecting code for hard-coded centers is
unreliable — the assumption hides in wrapping arithmetic — so this tests it
empirically instead.

Method: for each frame center, place the test content straddling *that frame's
own seam* (at ``center + 180``), draw, and measure the largest display-space
step between points matplotlib actually connects. A seam bug shows up as a step
spanning most of the canvas.

Two properties make the sweep trustworthy:

* Content is positioned **relative to each center's seam**, so every case is a
  real crossing. Fixed sky content would only cross for one center and the rest
  of the matrix would be vacuously green.
* A **positive control** (raw ``ax.plot``, which genuinely streaks) must fire at
  every center. If it stops firing, the detector has broken and the other
  assertions mean nothing.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import skyplothelper as sph  # noqa: E402

CENTERS = [0, 45, 90, 180, 270, 359]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _paths_of(artist):
    out = []
    try:
        if hasattr(artist, "get_xydata"):
            out.append((np.asarray(artist.get_xydata(), float),
                        artist.get_transform()))
        elif hasattr(artist, "get_paths"):
            for p in artist.get_paths():
                out.append((np.asarray(p.vertices, float),
                            artist.get_transform()))
        elif hasattr(artist, "get_path"):
            out.append((np.asarray(artist.get_path().vertices, float),
                        artist.get_transform()))
    except Exception:
        pass
    return out


def _worst_step_fraction(artists, width):
    """Largest display step between CONNECTED points, as a fraction of width.

    Pairs separated by a NaN are skipped, not filtered: filtering would measure
    straight across an intentional break and hide the very thing under test.
    """
    worst = 0.0
    for art in artists:
        for verts, tr in _paths_of(art):
            if verts.size == 0:
                continue
            try:
                disp = tr.transform(verts)
            except Exception:
                continue
            for i in range(len(disp) - 1):
                p, q = disp[i], disp[i + 1]
                if np.isfinite(p).all() and np.isfinite(q).all():
                    worst = max(worst, abs(q[0] - p[0]))
    return worst / width if width else 0.0


def _snapshot(ax):
    return set(map(id, ax.lines)) | set(map(id, ax.patches)) | \
        set(map(id, ax.collections))


def _added(ax, before):
    return [a for a in list(ax.lines) + list(ax.patches) + list(ax.collections)
            if id(a) not in before]


def _measure(center, draw):
    """Draw content straddling this frame's seam; return worst step fraction."""
    fig, ax = sph.allsky_figure(projection="AIT", center=center)
    fig.canvas.draw()
    seam = (center + 180.0) % 360.0
    before = _snapshot(ax)
    draw(ax, seam)
    fig.canvas.draw()
    frac = _worst_step_fraction(_added(ax, before),
                                float(ax.get_window_extent().width))
    plt.close(fig)
    return frac


# name -> callable(ax, seam_lon)
SEAM_CASES = {
    "plot": lambda ax, s: sph.plot(
        ax, (s + np.linspace(-20, 20, 41)) % 360, np.linspace(-5, 5, 41)),
    "geodesic_circle": lambda ax, s: sph.add_geodesic_circle(
        ax, s % 360, 0.0, 25.0),
    "spherical_polygon": lambda ax, s: sph.add_spherical_polygon(
        ax, [(s - 10) % 360, (s + 10) % 360, (s + 10) % 360, (s - 10) % 360],
        [-15, -15, 15, 15]),
    "lonlat_box": lambda ax, s: sph.add_lonlat_box(
        ax, -15, 15, (s - 10) % 360, (s + 10) % 360),
    "great_circle": lambda ax, s: sph.add_great_circle(ax, s % 360, 20.0),
    "rectangle": lambda ax, s: sph.add_rectangle(
        ax, s % 360, 0.0, width=30, height=20),
    "ellipse": lambda ax, s: sph.add_ellipse(
        ax, s % 360, 0.0, semi_major=20, semi_minor=10),
    "annulus": lambda ax, s: sph.add_annulus(
        ax, s % 360, 0.0, inner_radius=8, outer_radius=18),
    "great_circle_band": lambda ax, s: sph.add_great_circle_band(
        ax, s % 360, 20.0, half_width=8),
    "longitude_band": lambda ax, s: sph.add_longitude_band(
        ax, (s - 10) % 360, (s + 10) % 360),
}


@pytest.mark.parametrize("center", CENTERS)
@pytest.mark.parametrize("name", sorted(SEAM_CASES))
def test_overlay_survives_arbitrary_frame_center(name, center):
    """No overlay may streak across the map, whatever the frame center is."""
    frac = _measure(center, SEAM_CASES[name])
    assert frac < 0.5, (
        f"{name} at center={center} drew a connected step spanning "
        f"{frac:.0%} of the axes — a seam/frame-edge streak, usually an "
        "assumed center longitude.")


@pytest.mark.parametrize("center", CENTERS)
def test_seam_detector_actually_fires(center):
    """Positive control for the sweep above.

    Raw ``ax.plot`` across the seam genuinely streaks. If this stops failing,
    the detector is broken and every assertion above is vacuously true.
    """
    def _raw(ax, s):
        return ax.plot((s + np.linspace(-20, 20, 41)) % 360, np.zeros(41),
                       transform=ax.get_transform("world"))

    assert _measure(center, _raw) > 0.5, (
        "the seam detector no longer fires on a known-streaking path; "
        "the sweep can no longer catch regressions")


@pytest.mark.parametrize("lat", [75.0, 85.0])
def test_split_at_seam_catches_near_pole_crossing(lat):
    """Regression guard for the hybrid ``_split_at_seam``.

    On a pseudocylindrical frame (MOL/Robinson) that narrows toward the poles, a
    high-latitude seam crossing jumps only the *local* width — well under half
    the canvas — so the display-space detector alone MISSES it and the line
    streaks (the coastline near-pole streak). The analytic center-relative
    detector must catch it regardless of on-screen width.

    Asserts both halves so the test fails if the analytic branch is ever
    dropped: (1) the display jump really is < the 0.5-width threshold here, and
    (2) a NaN break is nonetheless inserted between the straddling vertices.
    """
    from skyplothelper.plotting import _split_at_seam
    fig, ax = sph.allsky_figure(projection="MOL", center=0)
    fig.canvas.draw()
    width = float(ax.get_window_extent().width)
    lon = np.array([170.0, 175.0, -175.0, -170.0])   # steps across the ±180 seam
    la = np.full_like(lon, lat)
    disp = ax.get_transform("world").transform(np.column_stack([lon, la]))
    display_jump = abs(disp[2, 0] - disp[1, 0]) / width
    assert display_jump < 0.5, (
        f"pick a higher latitude — display jump {display_jump:.2f} would be "
        "caught by the display detector alone, so this wouldn't guard the fix")
    out_lon, _ = _split_at_seam(ax, lon, la)
    assert np.count_nonzero(np.isnan(out_lon)) == 1, (
        "near-pole seam crossing was not broken — the analytic detector in "
        "_split_at_seam regressed and the line will streak")
    plt.close(fig)
