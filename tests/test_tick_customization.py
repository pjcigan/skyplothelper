"""Tests for per-axis / interpretive tick-label customization.

Covers the ``which`` knob on :func:`format_ticklabels`, and the
``stroke_*`` / per-axis / interpretive-rotation additions to
:func:`add_overlay_ticks` and :meth:`CoordinateOverlay.render_labels`.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import skyplothelper as sph  # noqa: E402
from skyplothelper._compat import coord_ticklabels  # noqa: E402
from skyplothelper.ticks import format_ticklabels  # noqa: E402
from skyplothelper.wcs_frame import make_wcs_frame  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _native_labels(ax, i):
    ax.figure.canvas.draw()
    text = coord_ticklabels(ax.coords[i]).text
    out = []
    for side in ("b", "t", "l", "r"):
        out.extend(t for t in text.get(side, []) if t)
    return out


def _overlay_labels(ax, kind=None):
    ax.figure.canvas.draw()
    return [t for t in ax.texts
            if getattr(t, "_sph_overlay_ticklabel", False)
            and (kind is None or getattr(t, "_sph_overlay_kind", None) == kind)]


# ---------------------------------------------------------------------------
# format_ticklabels(which=...)
# ---------------------------------------------------------------------------

def test_which_lon_leaves_latitude_untouched():
    """which='lon' + decimal_plain drops the ° from longitude only; the
    latitude labels keep the degree symbol from the prior 'decimal' pass."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(180, 30),
                        frame="galactic", fig=fig)
    format_ticklabels(ax, style="decimal")               # both get °
    format_ticklabels(ax, style="decimal_plain", which="lon")
    lon = _native_labels(ax, 0)
    lat = _native_labels(ax, 1)
    assert lon and lat
    assert all("°" not in x for x in lon), f"lon should lose °, got {lon}"
    assert any("°" in x for x in lat), f"lat should keep °, got {lat}"


def test_which_lat_leaves_longitude_untouched():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(180, 30),
                        frame="galactic", fig=fig)
    format_ticklabels(ax, style="decimal")
    format_ticklabels(ax, style="decimal_plain", which="lat")
    lon = _native_labels(ax, 0)
    lat = _native_labels(ax, 1)
    assert any("°" in x for x in lon), f"lon should keep °, got {lon}"
    assert all("°" not in x for x in lat), f"lat should lose °, got {lat}"


@pytest.mark.parametrize("alias", ["lon", "LON", "ra", "RA", "longitude", "l"])
def test_which_lon_aliases(alias):
    """Case-insensitive lon aliases all resolve to the longitude axis."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(180, 30),
                        frame="galactic", fig=fig)
    format_ticklabels(ax, style="decimal")
    format_ticklabels(ax, style="decimal_plain", which=alias)
    assert all("°" not in x for x in _native_labels(ax, 0))
    assert any("°" in x for x in _native_labels(ax, 1))


@pytest.mark.parametrize("alias", ["lat", "LAT", "dec", "Dec", "latitude", "b"])
def test_which_lat_aliases(alias):
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(180, 30),
                        frame="galactic", fig=fig)
    format_ticklabels(ax, style="decimal")
    format_ticklabels(ax, style="decimal_plain", which=alias)
    assert any("°" in x for x in _native_labels(ax, 0))
    assert all("°" not in x for x in _native_labels(ax, 1))


def test_which_both_is_default_behavior():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(180, 30),
                        frame="galactic", fig=fig)
    format_ticklabels(ax, style="decimal")
    format_ticklabels(ax, style="decimal_plain", which="both")
    assert all("°" not in x for x in _native_labels(ax, 0))
    assert all("°" not in x for x in _native_labels(ax, 1))


def test_which_invalid_raises():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(180, 30), fig=fig)
    with pytest.raises(ValueError, match="which must be"):
        format_ticklabels(ax, which="diagonal")


# ---------------------------------------------------------------------------
# add_overlay_ticks stroke + per-axis styling
# ---------------------------------------------------------------------------

def test_overlay_stroke_applies_path_effects():
    fig = plt.figure(figsize=(8, 4))
    ax = make_wcs_frame(111, projection="AIT", center=(0, 0), fig=fig)
    sph.add_overlay_ticks(ax, stroke_lw=3, stroke_color="w")
    labels = _overlay_labels(ax)
    assert labels
    assert all(t.get_path_effects() for t in labels), \
        "every overlay label should carry the stroke path_effects"


def test_overlay_stroke_requires_both_params():
    """A stroke needs both lw and color; one alone is a no-op (no effects)."""
    fig = plt.figure(figsize=(8, 4))
    ax = make_wcs_frame(111, projection="AIT", center=(0, 0), fig=fig)
    sph.add_overlay_ticks(ax, stroke_lw=3)          # no color
    assert all(not t.get_path_effects() for t in _overlay_labels(ax))


def test_overlay_per_axis_label_color():
    fig = plt.figure(figsize=(8, 4))
    ax = make_wcs_frame(111, projection="AIT", center=(0, 0), fig=fig)
    sph.add_overlay_ticks(ax, lon_label_kwargs={"color": "red"},
                          lat_label_kwargs={"color": "blue"})
    lon = _overlay_labels(ax, "lon")
    lat = _overlay_labels(ax, "lat")
    assert lon and lat
    assert {t.get_color() for t in lon} == {"red"}
    assert {t.get_color() for t in lat} == {"blue"}


def test_overlay_per_axis_preserves_all_labels():
    """The two-pass per-axis render keeps every label the shared path would."""
    fig = plt.figure(figsize=(8, 4))
    ax = make_wcs_frame(111, projection="AIT", center=(0, 0), fig=fig)
    ov_shared = sph.add_overlay_ticks(ax, label_kwargs={"mode": "complete"})
    n_shared = len(_overlay_labels(ax))

    fig2 = plt.figure(figsize=(8, 4))
    ax2 = make_wcs_frame(111, projection="AIT", center=(0, 0), fig=fig2)
    sph.add_overlay_ticks(ax2, lon_label_kwargs={"mode": "complete"},
                          lat_label_kwargs={"mode": "complete"})
    n_split = len(_overlay_labels(ax2))
    assert n_split == n_shared
    assert ov_shared is not None


# ---------------------------------------------------------------------------
# Interpretive tangent-relative rotation
# ---------------------------------------------------------------------------

def _rotations(ax, kind="lon"):
    return [round(t.get_rotation(), 3) for t in _overlay_labels(ax, kind)]


def test_tangent_perp_equals_tangent_plus_90():
    fig = plt.figure(figsize=(8, 4))
    ax = make_wcs_frame(111, projection="AIT", center=(0, 0), fig=fig)
    sph.add_overlay_ticks(ax, lon_at="lat=-30", lat_at=None,
                          suppress_default="lon",
                          label_kwargs={"rotate": "tangent_perp"})
    perp = _rotations(ax)

    fig2 = plt.figure(figsize=(8, 4))
    ax2 = make_wcs_frame(111, projection="AIT", center=(0, 0), fig=fig2)
    sph.add_overlay_ticks(ax2, lon_at="lat=-30", lat_at=None,
                          suppress_default="lon",
                          label_kwargs={"rotate": "tangent+90"})
    plus90 = _rotations(ax2)
    assert perp == plus90 and perp


def test_tangent_offset_parses_signs_and_spaces():
    """'tangent-30' and 'tangent + 45' parse (no ValueError from float())."""
    for spec in ("tangent-30", "tangent + 45", "tangent+0"):
        fig = plt.figure(figsize=(8, 4))
        ax = make_wcs_frame(111, projection="AIT", center=(0, 0), fig=fig)
        sph.add_overlay_ticks(ax, lon_at="lat=-30", lat_at=None,
                              suppress_default="lon",
                              label_kwargs={"rotate": spec})
        assert _overlay_labels(ax), f"{spec}: no labels rendered"
        plt.close(fig)


def test_tangent_perp_differs_from_tangent_upright():
    """The +90 offset actually rotates relative to the plain tangent."""
    fig = plt.figure(figsize=(8, 4))
    ax = make_wcs_frame(111, projection="AIT", center=(0, 0), fig=fig)
    sph.add_overlay_ticks(ax, lon_at="lat=-30", lat_at=None,
                          suppress_default="lon",
                          label_kwargs={"rotate": "tangent_upright"})
    upright = set(_rotations(ax))

    fig2 = plt.figure(figsize=(8, 4))
    ax2 = make_wcs_frame(111, projection="AIT", center=(0, 0), fig=fig2)
    sph.add_overlay_ticks(ax2, lon_at="lat=-30", lat_at=None,
                          suppress_default="lon",
                          label_kwargs={"rotate": "tangent_perp"})
    perp = set(_rotations(ax2))
    assert upright != perp
