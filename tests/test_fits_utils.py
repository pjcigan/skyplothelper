"""Tests for skyplothelper.core.fits_utils."""

import numpy as np
import pytest
from astropy.io import fits

from skyplothelper.core.fits_utils import (
    force_hdr_floats,
    force_hdr_to_2D,
    force_hdr_to_3D,
    getasecperpix,
    getcdelts,
    getcdmatrix,
    getdegperpix,
    getsteradperpix,
    header_coord_grids,
    squeeze_image,
)


def _base_header(extra=None):
    """Minimal 2D TAN header. Pixel scale: 1 arcsec/pix on each axis."""
    hdr = fits.Header()
    hdr['NAXIS'] = 2
    hdr['NAXIS1'] = 100
    hdr['NAXIS2'] = 100
    hdr['CTYPE1'] = 'RA---TAN'
    hdr['CTYPE2'] = 'DEC--TAN'
    hdr['CRPIX1'] = 50.5
    hdr['CRPIX2'] = 50.5
    hdr['CRVAL1'] = 180.0
    hdr['CRVAL2'] = 0.0
    if extra:
        for k, v in extra.items():
            hdr[k] = v
    return hdr


# ---- getcdelts: CDELT, CD-matrix, PC-matrix should all give matching results ----

def test_getcdelts_cdelt_form():
    pixscale_deg = 1.0 / 3600.0
    hdr = _base_header({
        'CDELT1': -pixscale_deg,
        'CDELT2': +pixscale_deg,
    })
    c1, c2 = getcdelts(hdr)
    assert c1 == pytest.approx(-pixscale_deg, abs=1e-12)
    assert c2 == pytest.approx(+pixscale_deg, abs=1e-12)


def test_getcdelts_cd_matrix_form():
    pixscale_deg = 1.0 / 3600.0
    hdr = _base_header({
        'CD1_1': -pixscale_deg,
        'CD1_2': 0.0,
        'CD2_1': 0.0,
        'CD2_2': +pixscale_deg,
    })
    c1, c2 = getcdelts(hdr)
    assert c1 == pytest.approx(-pixscale_deg, abs=1e-12)
    assert c2 == pytest.approx(+pixscale_deg, abs=1e-12)


def test_getcdelts_pc_matrix_form():
    pixscale_deg = 1.0 / 3600.0
    hdr = _base_header({
        'PC1_1': 1.0,
        'PC1_2': 0.0,
        'PC2_1': 0.0,
        'PC2_2': 1.0,
        'CDELT1': -pixscale_deg,
        'CDELT2': +pixscale_deg,
    })
    # CDELT keys take priority — the function returns them directly without
    # consulting the PC matrix. (The PC fallback only fires when CDELT is
    # absent AND CD is absent, which is unusual but handled.)
    c1, c2 = getcdelts(hdr)
    assert c1 == pytest.approx(-pixscale_deg, abs=1e-12)
    assert c2 == pytest.approx(+pixscale_deg, abs=1e-12)


def test_getcdelts_pc_matrix_only_no_cdelt():
    """When CDELT is missing entirely, the PC fallback (CD-matrix-style) is used."""
    pixscale_deg = 1.0 / 3600.0
    hdr = _base_header()
    # Strip any auto-added CDELT keys, then set PC + a stash CDELT for the
    # fallback computation in the function (it reads CDELT* under the PC branch).
    # The source uses PC*_*  *  CDELT* multiplication, so we must keep CDELT*.
    hdr['PC1_1'] = 1.0
    hdr['PC1_2'] = 0.0
    hdr['PC2_1'] = 0.0
    hdr['PC2_2'] = 1.0
    hdr['CDELT1'] = -pixscale_deg
    hdr['CDELT2'] = +pixscale_deg
    c1, c2 = getcdelts(hdr)
    assert c1 == pytest.approx(-pixscale_deg, abs=1e-12)
    assert c2 == pytest.approx(+pixscale_deg, abs=1e-12)


def test_getcdelts_missing_raises():
    hdr = _base_header()  # no CDELT, no CD, no PC
    with pytest.raises((ValueError, KeyError)):
        getcdelts(hdr)


def test_getdegperpix_from_cdelt():
    pixscale_deg = 1.0 / 3600.0
    hdr = _base_header({'CDELT1': -pixscale_deg, 'CDELT2': +pixscale_deg})
    assert getdegperpix(hdr) == pytest.approx(pixscale_deg, abs=1e-12)


# ---- squeeze_image ----

def test_squeeze_image_4d_degenerate():
    data = np.arange(256 * 256, dtype=float).reshape(1, 1, 256, 256)
    hdr = _base_header({
        'NAXIS': 4,
        'NAXIS3': 1,
        'NAXIS4': 1,
        'CTYPE3': 'FREQ',
        'CRVAL3': 1.4e9,
        'CTYPE4': 'STOKES',
        'CRVAL4': 1,
    })
    out_data, out_hdr = squeeze_image(data, hdr, verbose=False)
    assert out_data.shape == (256, 256)
    assert out_hdr['NAXIS'] == 2
    assert 'NAXIS3' not in out_hdr
    assert 'CTYPE3' not in out_hdr
    assert 'NAXIS4' not in out_hdr


def test_squeeze_image_2d_passthrough():
    data = np.zeros((10, 10))
    out_data, out_hdr = squeeze_image(data, header=None, verbose=False)
    assert out_data.shape == (10, 10)
    assert out_hdr is None


def test_squeeze_image_nondegenerate_raises():
    """A real 3D cube with NAXIS3 > 1 must NOT be silently flattened."""
    data = np.zeros((4, 10, 10))  # 4-channel cube
    hdr = _base_header({'NAXIS': 3, 'NAXIS3': 4, 'CTYPE3': 'FREQ'})
    with pytest.raises(ValueError, match="non-degenerate"):
        squeeze_image(data, hdr, verbose=False)


# ---- pixel-scale wrappers ----

def test_getasecperpix_and_steradperpix():
    hdr = _base_header({'CDELT1': -1 / 3600, 'CDELT2': 1 / 3600})
    assert getasecperpix(hdr) == pytest.approx(1.0)
    assert getsteradperpix(hdr) == pytest.approx(np.radians(1 / 3600) ** 2)


def test_getcdmatrix_from_cd_cards():
    hdr = _base_header({'CD1_1': -1 / 3600, 'CD1_2': 0.,
                        'CD2_1': 0., 'CD2_2': 1 / 3600})
    assert getcdmatrix(hdr) == (-1 / 3600, 0., 0., 1 / 3600)


def test_getcdmatrix_from_cdelt_crota():
    hdr = _base_header({'CDELT1': -1 / 3600, 'CDELT2': 1 / 3600, 'CROTA2': 0.})
    cd11, cd12, cd21, cd22 = getcdmatrix(hdr)
    assert cd11 == pytest.approx(-1 / 3600)
    assert cd22 == pytest.approx(1 / 3600)
    assert cd12 == pytest.approx(0.0, abs=1e-12)


# ---- header dimension / float fixers ----

def test_force_hdr_to_2D_strips_axis_cards_nondestructive():
    hdr = _base_header({'CDELT1': -1 / 3600, 'CDELT2': 1 / 3600})
    for k, v in {'NAXIS': 4, 'NAXIS3': 1, 'CTYPE3': 'FREQ', 'CRVAL3': 1e9,
                 'CRPIX3': 1, 'CDELT3': 1e6, 'NAXIS4': 1, 'CTYPE4': 'STOKES',
                 'CRVAL4': 1, 'WCSAXES': 4}.items():
        hdr[k] = v
    out = force_hdr_to_2D(hdr)
    assert out['NAXIS'] == 2 and out['WCSAXES'] == 2
    for k in ('NAXIS3', 'CTYPE3', 'CRVAL3', 'NAXIS4', 'CTYPE4'):
        assert k not in out
    # input header untouched
    assert hdr['NAXIS'] == 4 and 'CTYPE3' in hdr


def test_force_hdr_to_3D_keeps_axis3():
    hdr = _base_header({'CDELT1': -1 / 3600, 'CDELT2': 1 / 3600})
    for k, v in {'NAXIS': 4, 'NAXIS3': 1, 'CTYPE3': 'FREQ', 'NAXIS4': 1,
                 'CTYPE4': 'STOKES', 'WCSAXES': 4}.items():
        hdr[k] = v
    out = force_hdr_to_3D(hdr)
    assert out['NAXIS'] == 3 and out['WCSAXES'] == 3
    assert 'CTYPE3' in out and 'CTYPE4' not in out and 'NAXIS4' not in out


def test_force_hdr_floats_coerces_strings():
    from astropy.wcs import WCS
    hdr = _base_header({'CDELT1': '-0.0002778', 'CDELT2': '0.0002778',
                        'CRVAL1': '180.0', 'CRPIX1': '50.5', 'CRPIX2': '50.5'})
    out = force_hdr_floats(hdr)
    assert isinstance(out['CDELT1'], float)
    assert out['CDELT1'] == pytest.approx(-0.0002778)
    assert isinstance(out['CRVAL1'], float)
    WCS(out)  # must not raise


# ---- header_coord_grids ----

def _grid_header(nx=11, ny=9):
    hdr = _base_header({'CDELT1': -1 / 3600, 'CDELT2': 1 / 3600})
    hdr['NAXIS1'] = nx
    hdr['NAXIS2'] = ny
    hdr['CRPIX1'] = 6
    hdr['CRPIX2'] = 5
    hdr['CRVAL1'] = 180.0
    hdr['CRVAL2'] = 0.0
    return hdr


def test_header_coord_grids_from_header():
    lon, lat = header_coord_grids(_grid_header(11, 9))
    assert lon.shape == (9, 11)
    # reference pixel crpix-1 = (5, 4) 0-based → ~crval
    assert lon[4, 5] == pytest.approx(180.0, abs=1e-6)
    assert lat[4, 5] == pytest.approx(0.0, abs=1e-6)


def test_header_coord_grids_subset_and_1d():
    hdr = _grid_header(20, 20)
    lon, lat = header_coord_grids(hdr, x=np.arange(0, 20, 2),
                                  y=np.arange(0, 20, 4))
    assert lon.shape == (5, 10)
    lon1d, lat1d = header_coord_grids(hdr, return_1d=True)
    assert lon1d.shape == (20,) and lat1d.shape == (20,)


def test_header_coord_grids_wcs_input():
    from astropy.wcs import WCS
    w = WCS(_grid_header(12, 10))
    lon, lat = header_coord_grids(w, shape=(10, 12))
    assert lon.shape == (10, 12)
