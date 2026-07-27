"""Integration: build a multi-feature figure and verify it draws.

Combines wcs_frame + overlays + healpix + images + figures into one test
to catch interactions that single-module smoke tests miss.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.io import fits

from skyplothelper.figures import allsky_figure
from skyplothelper.healpix import _HAS_HEALPY, healpix_allsky_figure
from skyplothelper.images.quicklook import quicklook_figure
from skyplothelper.overlays.constellations import add_constellation_boundaries
from skyplothelper.overlays.planes import add_plane_overlay
from skyplothelper.overlays.surveys import add_survey_footprint


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_phase3_integration_allsky_with_overlays():
    """All-sky AIT with constellations + ecliptic + survey footprint."""
    fig, ax = allsky_figure(projection="AIT", center=180)
    add_plane_overlay(ax, plane="ecliptic", color="orange")
    add_constellation_boundaries(ax)
    add_survey_footprint(ax, survey="sdss")
    fig.canvas.draw()
    assert fig is not None


@pytest.mark.skipif(not _HAS_HEALPY, reason="healpy not installed")
def test_phase3_integration_healpix_allsky():
    """A HEALPix array rendered all-sky."""
    import healpy as hp
    arr = np.arange(hp.nside2npix(8), dtype=float)
    # healpix_allsky_figure returns HealpixResult(fig, ax, mappable, colorbar).
    result = healpix_allsky_figure(arr, projection="AIT")
    result.fig.canvas.draw()


def test_phase3_integration_quicklook_synthetic_fits():
    """A synthetic FITS image rendered via quicklook_plot."""
    rng = np.random.default_rng(7)
    data = rng.standard_normal((64, 64)).astype(float)
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = 64
    hdr["NAXIS2"] = 64
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRPIX1"] = 32.5
    hdr["CRPIX2"] = 32.5
    hdr["CRVAL1"] = 180.0
    hdr["CRVAL2"] = 0.0
    hdr["CDELT1"] = -1.0 / 3600
    hdr["CDELT2"] = 1.0 / 3600
    result = quicklook_figure(data, header=hdr)
    result.fig.canvas.draw()
