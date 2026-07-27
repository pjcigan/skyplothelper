"""Smoke tests for skyplothelper.images.reprojection."""

import pytest

from skyplothelper.images.reprojection import (
    _HAS_REPROJECT,
    load_sky_image,
    reproject_background,
    reproject_rgb_map,
)


def test_module_imports():
    """Even without reproject installed, the module must import cleanly."""
    assert callable(load_sky_image)
    assert callable(reproject_background)
    assert callable(reproject_rgb_map)


@pytest.mark.skipif(not _HAS_REPROJECT, reason="reproject not installed")
def test_reproject_smoke():
    """Real reproject test deferred to integration; just ensure callable."""
    pass


@pytest.mark.skipif(not _HAS_REPROJECT, reason="reproject not installed")
def test_reproject_background_matches_per_channel_reproject():
    """The shared-coordinate path must reproduce a per-channel reproject_interp
    (the old implementation) to interpolation tolerance."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    from skyplothelper.images.reprojection import reproject_background, reproject_interp

    sy, sx = 400, 800
    rgb = np.random.RandomState(0).random((sy, sx, 3)).astype(np.float32)
    shdr = fits.Header({"NAXIS": 2, "NAXIS1": sx, "NAXIS2": sy,
                        "CTYPE1": "RA---CAR", "CRVAL1": 180., "CRPIX1": sx / 2,
                        "CDELT1": -360. / sx, "CUNIT1": "deg",
                        "CTYPE2": "DEC--CAR", "CRVAL2": 0., "CRPIX2": sy / 2,
                        "CDELT2": 180. / sy, "CUNIT2": "deg"})
    tx, ty = 600, 300
    thdr = fits.Header({"NAXIS": 2, "NAXIS1": tx, "NAXIS2": ty,
                        "CTYPE1": "RA---AIT", "CRVAL1": 180., "CRPIX1": tx / 2,
                        "CDELT1": -360. / tx, "CUNIT1": "deg",
                        "CTYPE2": "DEC--AIT", "CRVAL2": 0., "CRPIX2": ty / 2,
                        "CDELT2": 180. / ty, "CUNIT2": "deg"})
    swcs = WCS(shdr)
    old = np.clip(np.nan_to_num(np.stack([
        reproject_interp((rgb[:, :, i], swcs), thdr, shape_out=(ty, tx),
                         order="bilinear")[0] for i in range(3)], -1), nan=0.),
        0, 1)
    new = reproject_background(rgb, shdr, thdr)
    assert new.shape == (ty, tx, 3)
    inside = old.sum(-1) > 0.01
    assert np.abs(old - new)[inside].mean() < 2e-3


@pytest.mark.skipif(not _HAS_REPROJECT, reason="reproject not installed")
def test_reproject_rgb_map_uint8_texture():
    """A uint8 RGB texture (e.g. a JPG panorama wrapped by pseudofits_from_image)
    must reproject without a dtype crash. map_coordinates preserves the input
    dtype, so the off-footprint NaN fill fails on an integer plane unless we
    interpolate in float. Regression for the uint8 draping crash."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    from skyplothelper.images.reprojection import reproject_rgb_map

    # Mirror pseudofits_from_image: data is (ny, nx, 3) channel-last uint8, the
    # header is NAXIS=3 with the sky WCS on image axes 2/3 (axis 1 = channels).
    sy, sx = 180, 360
    rgb = (np.random.RandomState(2).random((sy, sx, 3)) * 255).astype(np.uint8)
    shdr = fits.Header(dict(
        NAXIS=3, NAXIS1=3, NAXIS2=sx, NAXIS3=sy,
        CRPIX2=sx / 2, CRPIX3=sy / 2, CRVAL2=180., CRVAL3=0.,
        CDELT2=360. / sx, CDELT3=-180. / sy,
        CTYPE2="RA---CAR", CTYPE3="DEC--CAR",
        CUNIT2="deg", CUNIT3="deg", RADESYSa="ICRS",
    ).items())
    hdu = fits.ImageHDU(rgb, shdr)

    tx, ty = 300, 150
    thdr = fits.Header({"NAXIS": 2, "NAXIS1": tx, "NAXIS2": ty,
                        "CTYPE1": "RA---AIT", "CRVAL1": 180., "CRPIX1": tx / 2,
                        "CDELT1": -360. / tx, "CUNIT1": "deg",
                        "CTYPE2": "DEC--AIT", "CRVAL2": 0., "CRPIX2": ty / 2,
                        "CDELT2": 180. / ty, "CUNIT2": "deg"})
    out = reproject_rgb_map(hdu, WCS(thdr), shape_out=(ty, tx))
    assert out.shape == (ty, tx, 3)
    assert out.dtype == np.uint8            # preserves the input dtype contract


@pytest.mark.skipif(not _HAS_REPROJECT, reason="reproject not installed")
def test_reproject_background_downscale_full_frame():
    """downscale>1 still returns the full-frame shape (so ax.imshow lands)."""
    import numpy as np
    from astropy.io import fits

    from skyplothelper.images.reprojection import reproject_background

    sy, sx = 200, 400
    rgb = np.random.RandomState(1).random((sy, sx, 3)).astype(np.float32)
    shdr = fits.Header({"NAXIS": 2, "NAXIS1": sx, "NAXIS2": sy,
                        "CTYPE1": "RA---CAR", "CRVAL1": 180., "CRPIX1": sx / 2,
                        "CDELT1": -360. / sx, "CUNIT1": "deg",
                        "CTYPE2": "DEC--CAR", "CRVAL2": 0., "CRPIX2": sy / 2,
                        "CDELT2": 180. / sy, "CUNIT2": "deg"})
    tx, ty = 500, 260
    thdr = fits.Header({"NAXIS": 2, "NAXIS1": tx, "NAXIS2": ty,
                        "CTYPE1": "RA---AIT", "CRVAL1": 180., "CRPIX1": tx / 2,
                        "CDELT1": -360. / tx, "CUNIT1": "deg",
                        "CTYPE2": "DEC--AIT", "CRVAL2": 0., "CRPIX2": ty / 2,
                        "CDELT2": 180. / ty, "CUNIT2": "deg"})
    full = reproject_background(rgb, shdr, thdr)
    draft = reproject_background(rgb, shdr, thdr, downscale=2)
    assert draft.shape == full.shape == (ty, tx, 3)
    assert 0.0 <= draft.min() and draft.max() <= 1.0


@pytest.mark.skipif(not _HAS_REPROJECT, reason="reproject not installed")
def test_reproject_background_full_frame_on_shifted_limits():
    """On an axes whose xlim/ylim don't start near 0 (e.g. a limb-framed SIN
    globe inset), reproject_background builds the output for the FULL frame (the
    WCS pixel_shape), not the view span — so it aligns pixel 0 with data 0 and
    ax.imshow(result) isn't mis-placed / clipped. Regression for the inset bug."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.io import fits

    from skyplothelper.globe.frame import make_globe_frame

    ny, nx = 90, 180
    img = np.zeros((ny, nx), dtype=float)
    img[40:50, :] = 1.0                     # equatorial band
    hdr = fits.Header()
    hdr["NAXIS"], hdr["NAXIS1"], hdr["NAXIS2"] = 2, nx, ny
    hdr["CTYPE1"], hdr["CTYPE2"] = "RA---CAR", "DEC--CAR"
    hdr["CRPIX1"], hdr["CRPIX2"] = nx / 2 + 0.5, ny / 2 + 0.5
    hdr["CRVAL1"], hdr["CRVAL2"] = 0.0, 0.0
    hdr["CDELT1"], hdr["CDELT2"] = -2.0, 2.0

    fig = plt.figure()
    ax = make_globe_frame(111, center_LONdeg=0, center_LATdeg=0)
    fig.canvas.draw()
    full = ax.wcs.pixel_shape                # (360, 360)
    ax.set_xlim(120, 300)
    ax.set_ylim(90, 320)                     # shifted, non-zero origin
    bg = reproject_background(img, hdr, ax)
    assert bg.shape == (full[1], full[0])    # full frame, not the ~180x230 span
    # the equatorial band lands near the disk-center row (~180), not off-frame
    rows = np.where(bg.max(axis=1) > 0.3)[0]
    assert 150 < (rows.min() + rows.max()) / 2 < 210
    plt.close(fig)


@pytest.mark.skipif(not _HAS_REPROJECT, reason="reproject not installed")
def test_reproject_preserves_float32_precision():
    """Only INTEGER planes get promoted for interpolation.

    Forcing an already-float plane up to float64 perturbs every interpolated
    pixel by ~3e-8 — numerically harmless, but enough to re-render (and churn
    the diff of) every figure that drapes a float32 texture. Guards against
    re-introducing the blanket cast that the uint8 fix originally used.
    """
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    from skyplothelper.images.reprojection import _reproject_shared

    def _hdr(nx, ny, proj):
        return fits.Header({
            "NAXIS": 2, "NAXIS1": nx, "NAXIS2": ny,
            "CTYPE1": f"RA---{proj}", "CRVAL1": 180., "CRPIX1": nx / 2,
            "CDELT1": -360. / nx, "CUNIT1": "deg",
            "CTYPE2": f"DEC--{proj}", "CRVAL2": 0., "CRPIX2": ny / 2,
            "CDELT2": 180. / ny, "CUNIT2": "deg"})

    src, tgt = WCS(_hdr(240, 120, "CAR")), WCS(_hdr(200, 100, "AIT"))
    rgb32 = np.random.default_rng(0).random((120, 240, 3)).astype(np.float32)
    from_f32 = _reproject_shared(rgb32, src, tgt, (100, 200))
    from_f64 = _reproject_shared(rgb32.astype(np.float64), src, tgt, (100, 200))
    finite = np.isfinite(from_f32) & np.isfinite(from_f64)
    # If float32 were silently promoted these would be bit-identical.
    assert (from_f32[finite] != from_f64[finite]).any()


# --- cross-frame reprojection actually transforms ---------------------------

def _galactic_stripe_source(n=120):
    """A source whose only feature is a bright stripe along galactic b=0."""
    import astropy.io.fits as pyfits
    import numpy as np
    img = np.zeros((n, 2 * n, 3), dtype=float)
    img[n // 2 - 2:n // 2 + 3, :, :] = 1.0
    h = pyfits.Header()
    h["NAXIS"], h["NAXIS1"], h["NAXIS2"] = 2, 2 * n, n
    h["CTYPE1"], h["CTYPE2"] = "GLON-CAR", "GLAT-CAR"
    h["CRPIX1"], h["CRPIX2"] = n + 0.5, n / 2 + 0.5
    h["CRVAL1"], h["CRVAL2"] = 0.0, 0.0
    h["CDELT1"], h["CDELT2"] = -180.0 / n, 180.0 / n
    return img, h


def test_cross_frame_reprojection_is_not_a_no_op():
    """Reprojecting one galactic source onto galactic vs ICRS targets must
    give different pixels.

    Regression: `_reproject_shared` paired `pixel_to_world_values` with
    `world_to_pixel_values`. Those APIs are frame-agnostic — they exchange
    bare numbers in each WCS's own frame — so target RA/Dec were read as
    GLON/GLAT with no conversion and no error, and the cross-frame drape
    silently became a no-op. Same-frame drapes stayed correct, which is
    exactly why nothing caught it.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    import skyplothelper as sph
    img, hdr = _galactic_stripe_source()
    outs = []
    for frame in ("galactic", "ICRS"):
        fig, ax = sph.allsky_figure(projection="AIT", center=0, frame=frame,
                                    npix=(400, 200))
        outs.append(np.asarray(sph.reproject_background(img, hdr, ax),
                               dtype=float))
        plt.close("all")
    both = np.isfinite(outs[0]) & np.isfinite(outs[1])
    assert np.abs(outs[0] - outs[1])[both].mean() > 0.01


def test_reprojected_flux_follows_the_galactic_plane_not_the_equator():
    """The decisive check: on an ICRS target, the stripe must land on b=0.

    Asserting "the renders differ" alone would pass for a transform that is
    merely wrong in some other way, so this pins where the flux actually goes.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.coordinates import SkyCoord

    import skyplothelper as sph
    img, hdr = _galactic_stripe_source()
    fig, ax = sph.allsky_figure(projection="AIT", center=0, frame="ICRS",
                                npix=(400, 200))
    out = np.asarray(sph.reproject_background(img, hdr, ax), dtype=float)
    lum = np.nanmean(out, axis=2)

    ny, nx = lum.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    bright = lum > 0.5
    assert bright.sum() > 50, "the stripe did not survive the reprojection"

    world = ax.wcs.pixel_to_world(xx[bright], yy[bright])
    gal_b = SkyCoord(world).galactic.b.deg
    # The stripe is ~5 source-pixels tall (~7.5 deg); allow generously for
    # AIT resampling at this test's coarse grid.
    assert np.nanmedian(np.abs(gal_b)) < 12.0

    dec = SkyCoord(world).icrs.dec.deg
    # And it must NOT be lying along the celestial equator, which is what the
    # untransformed no-op produced.
    assert np.nanmedian(np.abs(dec)) > np.nanmedian(np.abs(gal_b))
    plt.close("all")


def _periodic_lon_source(ny=180, nx=360):
    """A source that is smooth and EXACTLY periodic in longitude.

    Deliberately featureless: on a real star field a one-pixel seam is
    indistinguishable from ordinary structure, so the first attempt at this
    measurement could not tell the two apart. Here any discontinuity can only
    be the interpolation.
    """
    import astropy.io.fits as pyfits
    import numpy as np
    h = pyfits.Header()
    h["NAXIS"], h["NAXIS1"], h["NAXIS2"] = 2, nx, ny
    h["CTYPE1"], h["CTYPE2"] = "GLON-CAR", "GLAT-CAR"
    h["CRPIX1"], h["CRPIX2"] = nx / 2 + 0.5, ny / 2 + 0.5
    h["CRVAL1"], h["CRVAL2"] = 0.0, 0.0
    h["CDELT1"], h["CDELT2"] = -360.0 / nx, 180.0 / ny
    glon = (h["CRVAL1"]
            + (np.arange(nx) + 1 - h["CRPIX1"]) * h["CDELT1"]) % 360.0
    plane = np.repeat(_lon_signal(glon)[None, :], ny, axis=0)
    return np.dstack([plane, plane, plane]), h


def _lon_signal(lon_deg):
    import numpy as np
    return 0.5 + 0.4 * np.cos(np.radians(lon_deg))


def test_periodic_source_has_no_seam_after_cross_frame_drape():
    """A full-sky cylindrical source wraps: its first and last columns are
    neighbors on the sky.

    Regression: map_coordinates does not know that, so a target pixel landing
    between them interpolated off the array edge and took the fill value —
    a one-pixel dark chord along the source's seam. Same-frame drapes hid it
    (there the seam sits on the map edge); fixing the frame transform moved it
    mid-map and exposed it.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.coordinates import SkyCoord

    import skyplothelper as sph
    img, hdr = _periodic_lon_source()
    fig, ax = sph.allsky_figure(projection="AIT", center=180, frame="ICRS",
                                npix=(600, 300))
    got = np.nanmean(np.asarray(sph.reproject_background(img, hdr, ax),
                                dtype=float), axis=2)
    ny, nx = got.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    sky = SkyCoord(ax.wcs.pixel_to_world(xx, yy)).galactic
    want = _lon_signal(sky.l.deg)
    plt.close("all")

    # Longitude is degenerate at the poles — one output pixel there spans every
    # longitude, so the per-pixel expectation is meaningless. Excluded because
    # the oracle fails there, not the code.
    ok = (np.isfinite(got) & np.isfinite(want)
          & (np.abs(sky.b.deg) < 89.0))
    err = np.abs(got - want)[ok]
    assert err.mean() < 1e-3
    assert int((err > 0.1).sum()) == 0


# --- geo / celestial drapes are announced -----------------------------------

def _wcs_for(lon_ctype, lat_ctype):
    import astropy.io.fits as pyfits
    from astropy.wcs import WCS
    h = pyfits.Header()
    h["NAXIS"], h["NAXIS1"], h["NAXIS2"] = 2, 180, 90
    h["CTYPE1"], h["CTYPE2"] = lon_ctype, lat_ctype
    h["CRPIX1"], h["CRPIX2"] = 90.5, 45.5
    h["CRVAL1"], h["CRVAL2"] = 0.0, 0.0
    h["CDELT1"], h["CDELT2"] = -2.0, 2.0
    return WCS(h)


def _mismatch_warnings(src, tgt):
    import warnings

    from skyplothelper.images.reprojection import (
        _warn_if_geo_celestial_mismatch,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_if_geo_celestial_mismatch(_wcs_for(*src), _wcs_for(*tgt))
        return [str(w.message) for w in caught if "body-fixed" in str(w.message)]


@pytest.mark.parametrize("src,tgt", [
    (("TLON-CAR", "TLAT-CAR"), ("RA---CAR", "DEC--CAR")),
    (("RA---CAR", "DEC--CAR"), ("TLON-CAR", "TLAT-CAR")),
])
def test_geo_celestial_drape_warns(src, tgt):
    """ITRS<->ICRS is an epoch-dependent Earth rotation, and a texture drape is
    not an observation at an instant — so the result is rotated by an
    essentially arbitrary amount. The conversion is correct and documented;
    what was missing was any runtime signal that the frames weren't matched."""
    msgs = _mismatch_warnings(src, tgt)
    assert msgs
    assert "obstime" in msgs[0]


@pytest.mark.parametrize("src,tgt", [
    (("TLON-CAR", "TLAT-CAR"), ("TLON-CAR", "TLAT-CAR")),
    (("RA---CAR", "DEC--CAR"), ("RA---CAR", "DEC--CAR")),
    # The discrimination that matters: galactic->celestial is a FIXED rotation
    # with no epoch dependence, so it is a legitimate cross-frame drape and
    # must stay silent. Warning here would train users to ignore the warning.
    (("GLON-CAR", "GLAT-CAR"), ("RA---CAR", "DEC--CAR")),
])
def test_legitimate_drapes_stay_quiet(src, tgt):
    assert _mismatch_warnings(src, tgt) == []
