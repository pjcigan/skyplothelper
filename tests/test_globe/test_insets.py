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


def test_clean_inset_hides_axis_labels_keeps_ticklabels():
    import matplotlib.pyplot as plt

    import skyplothelper as sph

    fig, pax = sph.allsky_figure(projection="AIT", center=180)
    iax = sph.reproject_inset_axes(pax, [0.6, 0.6, 0.3, 0.3],
                                   center=(180, 0), size=8.0)
    assert iax.coords[0].axislabels.get_visible()      # verbose labels on by default
    sph.clean_inset(iax)
    assert not iax.coords[0].axislabels.get_visible()  # ...gone after clean_inset
    assert not iax.coords[1].axislabels.get_visible()
    plt.close(fig)


def test_reproject_inset_clean_knob_opt_in():
    import matplotlib.pyplot as plt

    import skyplothelper as sph

    fig, pax = sph.allsky_figure(projection="AIT", center=180)
    # default (clean=False): axis labels present
    plain = sph.reproject_inset_axes(pax, [0.05, 0.6, 0.3, 0.3],
                                     center=(180, 0), size=8.0)
    assert plain.coords[0].axislabels.get_visible()
    # clean=True: applied on creation
    tidy = sph.reproject_inset_axes(pax, [0.6, 0.6, 0.3, 0.3],
                                    center=(180, 0), size=8.0, clean=True)
    assert not tidy.coords[0].axislabels.get_visible()
    plt.close(fig)
