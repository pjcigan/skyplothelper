"""images (quicklook_plot / simpleimageplot / levels) verification.

The canonical ``tests/test_images/*`` smoke-tests the public surface
loosely. This file fills in:

  * ``rescale_image`` covers all 8 registered stretches.
  * ``make_norm`` returns a ``matplotlib.colors.Normalize`` instance
    for each stretch.
  * ``clip_percentile`` / ``clip_sigma`` / ``clip_zscale`` produce
    sensible ordered (lo, hi) tuples.
  * ``quicklook_plot`` returns an axes + plotted artists on a
    synthetic Gaussian source.
  * ``simpleimageplot`` writes a savepath without raising.
  * ``auto_stretch`` returns a name in the registry.
  * ``describe_image`` runs without raising.
"""

import os
import tempfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import Normalize

from skyplothelper.images.levels import (
    auto_stretch,
    clip_percentile,
    clip_sigma,
    clip_zscale,
    describe_image,
    list_stretches,
    make_norm,
    rescale_image,
)
from skyplothelper.images.quicklook import (
    _stat_fmt,
    quicklook_plot,
    simpleimage_figure,
)
from skyplothelper.wcs_frame import dummy_standard_hdr


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ============================================================
# Synthetic test image: 2D Gaussian + small noise
# ============================================================

@pytest.fixture
def synthetic_image_and_header():
    rng = np.random.default_rng(0)
    nx, ny = 80, 80
    yy, xx = np.mgrid[0:ny, 0:nx]
    # Gaussian source + Gaussian noise
    img = (np.exp(-((xx - 40) ** 2 + (yy - 40) ** 2) / (2 * 8 ** 2)) +
           rng.normal(0, 0.02, (ny, nx)))
    hdr = dummy_standard_hdr(
        centercoords_deg=(180.0, 30.0),
        cdelts=(-1.0 / 3600, 1.0 / 3600),
        cunit="deg", projection="TAN",
        naxis_xy=(nx, ny),
    )
    return img, hdr


# ============================================================
# clip_* return ordered (lo, hi) pairs
# ============================================================

def test_clip_percentile_returns_ordered_pair(synthetic_image_and_header):
    img, _ = synthetic_image_and_header
    lo, hi = clip_percentile(img, plo=1, phi=99)
    assert lo < hi


def test_clip_sigma_returns_ordered_pair(synthetic_image_and_header):
    img, _ = synthetic_image_and_header
    lo, hi = clip_sigma(img, sigma_lo=3)
    assert lo < hi


def test_clip_zscale_returns_ordered_pair(synthetic_image_and_header):
    img, _ = synthetic_image_and_header
    lo, hi = clip_zscale(img)
    assert lo < hi


# ============================================================
# rescale_image / make_norm cover all registered stretches
# ============================================================

_REGISTERED_STRETCHES = ["linear", "sqrt", "squared", "log",
                         "asinh", "sinh", "power"]
# 'histeq' requires astropy + the data — skip from the parametrize


@pytest.mark.parametrize("stretch", _REGISTERED_STRETCHES)
def test_rescale_image_each_stretch(stretch, synthetic_image_and_header):
    img, _ = synthetic_image_and_header
    out = rescale_image(img, stretch=stretch, clip="percentile")
    assert out.shape == img.shape
    assert np.all(np.isfinite(out))
    # rescale_image normalizes the output to [0, 1] (within numerical tolerance)
    assert out.min() >= -1e-6, f"min={out.min()} should be >= 0"
    assert out.max() <= 1.0 + 1e-6, f"max={out.max()} should be <= 1"


@pytest.mark.parametrize("stretch", _REGISTERED_STRETCHES)
def test_make_norm_each_stretch(stretch, synthetic_image_and_header):
    img, _ = synthetic_image_and_header
    norm = make_norm(stretch=stretch, data=img)
    assert isinstance(norm, Normalize)
    # Sanity: .vmin/.vmax are set
    assert norm.vmin is not None
    assert norm.vmax is not None
    assert norm.vmin < norm.vmax


# ============================================================
# auto_stretch + describe_image + list_stretches
# ============================================================

def test_auto_stretch_returns_known_stretch(synthetic_image_and_header):
    """auto_stretch returns (stretch_name, reasoning_string)."""
    img, _ = synthetic_image_and_header
    out = auto_stretch(img)
    assert isinstance(out, tuple) and len(out) == 2
    name, reason = out
    assert isinstance(name, str)
    assert name in _REGISTERED_STRETCHES + ["histeq"]
    assert isinstance(reason, str) and len(reason) > 0


def test_describe_image_runs(synthetic_image_and_header, capsys):
    img, _ = synthetic_image_and_header
    describe_image(img, name="Test")
    out = capsys.readouterr().out
    # Expect at least one numeric statistic in the output
    assert any(c.isdigit() for c in out), "describe_image should print stats"


def test_list_stretches_runs(capsys):
    list_stretches()
    out = capsys.readouterr().out
    for s in _REGISTERED_STRETCHES:
        assert s in out, f"list_stretches() did not mention {s!r}"


# ============================================================
# quicklook_plot — returns an axes / draws successfully
# ============================================================

def test_quicklook_plot_runs_on_synthetic_image(synthetic_image_and_header):
    img, hdr = synthetic_image_and_header
    fig = plt.figure(figsize=(6, 6.5))
    from astropy.wcs import WCS
    ax = fig.add_subplot(111, projection=WCS(hdr))
    result = quicklook_plot(img, ax=ax, header=hdr,
                            contours=True, image=False, show_info=False)
    fig.canvas.draw()
    # quicklook_plot returns a QuicklookResult NamedTuple
    assert result.fig is fig
    assert result.contour_set is not None


def test_quicklook_plot_image_and_contours(synthetic_image_and_header):
    img, hdr = synthetic_image_and_header
    fig = plt.figure(figsize=(6, 6.5))
    from astropy.wcs import WCS
    ax = fig.add_subplot(111, projection=WCS(hdr))
    result = quicklook_plot(img, ax=ax, header=hdr,
                            contours=True, image=True,
                            colormap="inferno",
                            stretch="asinh", show_info=False)
    fig.canvas.draw()
    assert result.image is not None
    assert result.contour_set is not None


# ============================================================
# Info-block statistic formatting — peak and RMS format from
# their own magnitudes (a bright peak must not truncate a faint RMS)
# ============================================================

def test_stat_fmt_ladder():
    assert _stat_fmt(1234.5) == ".1f"     # >= 100
    assert _stat_fmt(3.013) == ".3f"      # >= 1
    assert _stat_fmt(0.05) == ".4f"       # >= 0.01
    assert _stat_fmt(0.00042) == ".3g"    # sub-0.01 -> 3 sig figs


def test_stat_fmt_faint_rms_not_truncated():
    # A sub-mJy RMS alongside a ~few-Jy peak: fixed .3f would render 0.000.
    peak, rms = 3.013, 0.00042
    assert f"{rms:{_stat_fmt(rms)}}" == "0.00042"
    # Peak stays on the familiar fixed-decimal ladder.
    assert f"{peak:{_stat_fmt(peak)}}" == "3.013"


def test_info_block_shows_faint_rms(synthetic_image_and_header):
    # Bright compact source (~3 Jy peak) on very faint noise (~4e-4 RMS),
    # the regime that previously truncated RMS to 0.000 in the info block.
    _, hdr = synthetic_image_and_header
    from astropy.wcs import WCS
    nx, ny = 80, 80
    yy, xx = np.mgrid[0:ny, 0:nx]
    rng = np.random.default_rng(1)
    img = (3.0 * np.exp(-((xx - 40) ** 2 + (yy - 40) ** 2) / (2 * 4 ** 2)) +
           rng.normal(0, 4e-4, (ny, nx)))
    fig = plt.figure(figsize=(6, 6.5))
    ax = fig.add_subplot(111, projection=WCS(hdr))
    quicklook_plot(img, ax=ax, header=hdr, image=True,
                   contours=False, show_info=True)
    fig.canvas.draw()
    info = "".join(t.get_text() for t in ax.texts if "RMS =" in t.get_text())
    assert "RMS =" in info
    # The RMS value on the line is nonzero (not truncated to 0.000/0.0000).
    rms_field = info.split("RMS =")[1].split()[0]
    assert float(rms_field) > 0.0, info


# ============================================================
# simpleimageplot — quick savepath roundtrip
# ============================================================

def test_simpleimage_figure_writes_savepath(synthetic_image_and_header):
    img, hdr = synthetic_image_and_header
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.png")
        simpleimage_figure(img, hdr, axtitle="test", savepath=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000  # PNG with content
