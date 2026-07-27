"""Tests for skyplothelper.globe.insets."""

import matplotlib

matplotlib.use("Agg")

def test_inset_center_accepts_skycoord_in_parent_frame():
    """An inset's center resolves in the PARENT axes' frame, not blindly ICRS:
    it stays l/b on a galactic parent and converts on an ICRS one."""
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.coordinates import SkyCoord

    import skyplothelper as sph
    from skyplothelper.globe.insets import reproject_inset_axes

    gal = SkyCoord(120.0, 30.0, unit="deg", frame="galactic")

    fig, pax = sph.allsky_figure(projection="AIT", center=180, frame="galactic")
    iax = reproject_inset_axes(pax, [0.6, 0.6, 0.3, 0.3], center=gal, size=10.0)
    assert np.allclose(iax.wcs.wcs.crval, [120.0, 30.0], atol=1e-3)
    plt.close(fig)

    fig, pax = sph.allsky_figure(projection="AIT", center=180)
    iax = reproject_inset_axes(pax, [0.6, 0.6, 0.3, 0.3], center=gal, size=10.0)
    assert np.allclose(iax.wcs.wcs.crval,
                       [gal.icrs.ra.deg, gal.icrs.dec.deg], atol=1e-3)
    plt.close(fig)


def test_inset_center_tuple_unchanged():
    import matplotlib.pyplot as plt
    import numpy as np

    import skyplothelper as sph
    from skyplothelper.globe.insets import reproject_inset_axes
    fig, pax = sph.allsky_figure(projection="AIT", center=180)
    iax = reproject_inset_axes(pax, [0.6, 0.6, 0.3, 0.3],
                               center=(83.6, 22.0), size=5.0)
    assert np.allclose(iax.wcs.wcs.crval, [83.6, 22.0], atol=1e-3)
    plt.close(fig)
