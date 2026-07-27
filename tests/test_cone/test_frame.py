"""Tests for skyplothelper.cone.frame."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from skyplothelper.cone.frame import make_bowtie_frame, make_cone_frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_make_cone_frame_basic():
    fig = plt.figure()
    ax = make_cone_frame(
        111, angle_center=180, angle_half_width=30,
        r_min=0, r_max=0.15, fig=fig,
    )
    assert ax is not None
    fig.canvas.draw()


def test_make_cone_frame_with_angle_label():
    fig = plt.figure()
    ax = make_cone_frame(
        111, angle_center=90, angle_half_width=40,
        r_min=0, r_max=0.1,
        angle_label="Galactic longitude $\\ell$",
        fig=fig,
    )
    assert ax is not None
    fig.canvas.draw()


def test_make_bowtie_frame_returns_pair():
    fig = plt.figure()
    pair = make_bowtie_frame(
        angle_center=0, angle_half_width=45,
        r_min=0, r_max=0.2, fig=fig,
    )
    # Should return a 2-tuple of axes (top, bot or left, right)
    assert len(pair) == 2
    fig.canvas.draw()


# ============================================================
# Per-half overrides
# ============================================================

def test_make_bowtie_frame_per_half_override_vertical():
    """``top_kwargs`` / ``bot_kwargs`` apply per-half overrides
    to the underlying ``make_cone_frame`` calls."""
    fig = plt.figure()
    top, bot = make_bowtie_frame(
        angle_center=0, angle_half_width=45,
        r_min=0, r_max=0.2, fig=fig,
        gridcolor="0.7",  # default for both
        top_kwargs={"gridcolor": "steelblue"},
        bot_kwargs={"gridcolor": "crimson"},
    )
    assert top._cone_bowtie_role == "top"
    assert bot._cone_bowtie_role == "bot"
    # Each half should have a polar grid styled with its own color.
    # The exact gridline-color introspection depends on matplotlib
    # internals; just confirm both halves rendered without error.
    fig.canvas.draw()


def test_make_bowtie_frame_per_half_override_horizontal():
    fig = plt.figure()
    left, right = make_bowtie_frame(
        orientation="horizontal",
        angle_center=180, angle_half_width=40,
        r_min=0, r_max=0.15, fig=fig,
        left_kwargs={"angle_label": "NGP"},
        right_kwargs={"angle_label": "SGP"},
    )
    assert left._cone_bowtie_role == "left"
    assert right._cone_bowtie_role == "right"
    fig.canvas.draw()


def test_make_bowtie_frame_top_kwargs_rejected_for_horizontal():
    """Mixing orientation-incompatible per-half kwargs raises
    rather than silently ignoring them."""
    fig = plt.figure()
    with pytest.raises(TypeError, match="orientation='vertical'"):
        make_bowtie_frame(
            orientation="horizontal",
            angle_center=0, angle_half_width=45,
            r_min=0, r_max=0.2, fig=fig,
            top_kwargs={"gridcolor": "blue"},
        )


def test_make_bowtie_frame_left_kwargs_rejected_for_vertical():
    fig = plt.figure()
    with pytest.raises(TypeError, match="orientation='horizontal'"):
        make_bowtie_frame(
            orientation="vertical",
            angle_center=0, angle_half_width=45,
            r_min=0, r_max=0.2, fig=fig,
            left_kwargs={"gridcolor": "red"},
        )


# --- grid knobs and the wrapper's default forwarding ------------------------

def _render(fn, **kw):
    """Pixels, not artists: polar grid lines are not axes children, so
    inspecting the artist tree finds nothing and a working knob looks broken.
    """
    fig = plt.figure(figsize=(4, 4), dpi=100)
    fn(fig=fig, **kw)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return buf


def _diff(a, b):
    return int((np.abs(a.astype(int) - b.astype(int)).sum(axis=2) > 8).sum())


_CONE = dict(angle_center=180, angle_half_width=30, r_max=1.0)


@pytest.mark.parametrize("kw", [
    {"gridlw": 3.0}, {"gridls": "--"},
    {"gridcolor": "red"}, {"gridalpha": 1.0},
])
def test_cone_grid_knobs_reach_the_render(kw):
    ref = _render(make_cone_frame, **_CONE)
    assert _diff(ref, _render(make_cone_frame, **_CONE, **kw)) > 50


def test_cone_gridlw_default_is_the_historical_value():
    """None must keep 0.5, not inherit rcParams — inheriting would move
    every existing cone render."""
    a = _render(make_cone_frame, **_CONE)
    b = _render(make_cone_frame, **_CONE, gridlw=0.5)
    assert _diff(a, b) == 0


def test_bowtie_defaults_match_the_frame_defaults():
    """make_bowtie_frame must not shadow make_cone_frame's defaults.

    It used to restate them and forward them unconditionally, so changing a
    default in one place and not the other silently did nothing here.
    """
    ref = _render(make_bowtie_frame, **_CONE)
    explicit = _render(make_bowtie_frame, **_CONE, gridcolor="0.8",
                       gridalpha=0.5, label_fontsize=11, tick_fontsize=9)
    assert _diff(ref, explicit) == 0


@pytest.mark.parametrize("kw", [
    {"gridlw": 3.0}, {"gridls": "--"}, {"gridcolor": "red"},
    {"tick_fontsize": 16}, {"label_fontsize": 20},
])
def test_bowtie_style_knobs_reach_both_halves(kw):
    ref = _render(make_bowtie_frame, **_CONE)
    assert _diff(ref, _render(make_bowtie_frame, **_CONE, **kw)) > 50


def test_bowtie_per_half_override_still_wins():
    a = _render(make_bowtie_frame, **_CONE, gridcolor="red")
    b = _render(make_bowtie_frame, **_CONE, gridcolor="red",
                top_kwargs={"gridcolor": "blue"})
    assert _diff(a, b) > 50
