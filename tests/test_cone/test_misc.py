"""Smoke tests for cosmology, ticks, labels, twin."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from skyplothelper.cone.cosmology import _R_VARIABLES, redshift_to_r
from skyplothelper.cone.frame import make_cone_frame
from skyplothelper.cone.labels import flip_label, get_label_pad, set_label_pad
from skyplothelper.cone.ticks import add_minor_rticks, log_r
from skyplothelper.cone.twin import make_twinr


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---- cosmology ----

def test_r_variables_known():
    assert "redshift" in _R_VARIABLES
    assert "comoving_distance" in _R_VARIABLES
    assert "lookback_time" in _R_VARIABLES


def test_redshift_to_r_identity():
    """r_variable='redshift' returns the input unchanged."""
    z = np.array([0.01, 0.05, 0.1])
    r = redshift_to_r(z, r_variable="redshift")
    np.testing.assert_allclose(np.asarray(r), z)


def test_redshift_to_r_comoving_distance():
    """Comoving distance requires astropy + cosmology; skip otherwise."""
    pytest.importorskip("astropy.cosmology")
    from astropy.cosmology import Planck18
    z = np.array([0.01, 0.05])
    r = redshift_to_r(z, r_variable="comoving_distance",
                      cosmology=Planck18, r_unit="Mpc")
    # Just verify monotonic increase and positive
    arr = np.asarray(r)
    assert arr[1] > arr[0] > 0


# ---- ticks ----

def test_add_minor_rticks_smoke():
    fig = plt.figure()
    ax = make_cone_frame(
        111, angle_center=180, angle_half_width=30,
        r_min=0, r_max=0.15, fig=fig,
    )
    add_minor_rticks(ax, step=0.025)
    fig.canvas.draw()


def test_log_r_smoke():
    """log_r requires r_min > 0."""
    fig = plt.figure()
    ax = make_cone_frame(
        111, angle_center=180, angle_half_width=30,
        r_min=0.01, r_max=0.5, fig=fig,
    )
    log_r(ax)
    fig.canvas.draw()


# ---- twin ----

def test_make_twinr_smoke():
    fig = plt.figure()
    ax = make_cone_frame(
        111, angle_center=180, angle_half_width=30,
        r_min=0, r_max=0.15, fig=fig,
    )
    # Identity conversion as a smoke test
    twin = make_twinr(ax, convert=lambda r: r * 1.0, inverse=lambda r: r * 1.0)
    assert twin is not None
    fig.canvas.draw()


# ---- labels ----

def test_set_get_label_pad_no_crash():
    """set_label_pad / get_label_pad must run without exception on a plain Text.

    These helpers are designed for the pad-recomputer-cached Text artists
    placed by ``_place_axis_labels``; on an arbitrary Text they may return
    None / no-op, but they must not raise.
    """
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "test")
    set_label_pad(txt, 10)
    get_label_pad(txt)  # may return None — just don't crash


def test_flip_label_no_op_on_plain_text():
    """flip_label should not crash on a non-cone Text artist."""
    fig, ax = plt.subplots()
    txt = ax.text(0.5, 0.5, "test")
    # Should be a graceful no-op or at most rotate
    flip_label(txt)
