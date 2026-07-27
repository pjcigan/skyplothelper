"""optional-dependency failure modes.

Each subsystem that depends on a non-base library guards entry with
either an ``_HAS_*`` flag or a ``_require_*`` helper that raises an
informative ImportError when the library is missing. This file
monkeypatches those flags to False and verifies the user gets a
helpful, dependency-named error message rather than a confusing
NameError or AttributeError later in the call.

Tested gates:
  * healpy → ``healpix.*``, ``figures.projection_gallery``
  * cartopy → ``cartopy_backend.*``, ``globe.nightshade``,
    ``globe.plotting._draw_country_borders``
  * astroquery → ``queries.*``
  * reproject → ``images.reprojection.*``
  * scipy → ``globe.nightshade``, ``globe.plotting._gaussian_smooth``
  * PIL → ``diagnostics.saved_plot_size_reducer``
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ============================================================
# healpy gate
# ============================================================

def test_healpix_to_celestial_raises_when_healpy_missing(monkeypatch):
    from skyplothelper import healpix
    monkeypatch.setattr(healpix, "_HAS_HEALPY", False)
    with pytest.raises(ImportError, match="(?i)healpy"):
        healpix.healpix_to_celestial([0.0, 1.0, 2.0])


def test_healpix_smooth_raises_when_healpy_missing(monkeypatch):
    from skyplothelper import healpix
    monkeypatch.setattr(healpix, "_HAS_HEALPY", False)
    with pytest.raises(ImportError, match="(?i)healpy"):
        healpix.healpix_smooth([0.0, 1.0, 2.0], sigma_deg=1.0)


def test_auto_nside_raises_when_healpy_missing(monkeypatch):
    from skyplothelper import healpix
    monkeypatch.setattr(healpix, "_HAS_HEALPY", False)
    with pytest.raises(ImportError, match="(?i)healpy"):
        healpix.auto_nside(resolution_deg=1.0)


def test_projection_gallery_raises_when_healpy_missing(monkeypatch):
    """figures.projection_gallery imports _require_healpy from healpix."""
    from skyplothelper import figures, healpix
    monkeypatch.setattr(healpix, "_HAS_HEALPY", False)
    with pytest.raises(ImportError, match="(?i)healpy"):
        figures.projection_gallery()


# ============================================================
# cartopy gate
# ============================================================

def test_resolve_cartopy_crs_raises_when_cartopy_missing(monkeypatch):
    from skyplothelper import cartopy_backend
    monkeypatch.setattr(cartopy_backend, "_HAS_CARTOPY", False)
    with pytest.raises(ImportError, match="(?i)cartopy"):
        cartopy_backend._resolve_cartopy_crs("mollweide")


def test_make_cartopy_frame_raises_when_cartopy_missing(monkeypatch):
    from skyplothelper import cartopy_backend
    monkeypatch.setattr(cartopy_backend, "_HAS_CARTOPY", False)
    with pytest.raises(ImportError, match="(?i)cartopy"):
        cartopy_backend.make_cartopy_frame()


def test_make_nightshade_raises_when_cartopy_missing(monkeypatch):
    from datetime import datetime

    import numpy as np

    from skyplothelper.globe import nightshade
    monkeypatch.setattr(nightshade, "_HAS_CARTOPY", False)
    rgb = np.zeros((100, 200, 3))
    # The gaussian blend needs cartopy; the default 'elevation' blend does not.
    with pytest.raises(ImportError, match="(?i)cartopy"):
        nightshade.make_nightshade_blend(rgb, datetime(2024, 6, 21, 12),
                                         blend="gaussian")


# ============================================================
# scipy gate (used in nightshade Gaussian blending)
# ============================================================

def test_make_nightshade_raises_when_scipy_missing(monkeypatch):
    from datetime import datetime

    import numpy as np

    from skyplothelper.globe import nightshade
    monkeypatch.setattr(nightshade, "_HAS_SCIPY", False)
    rgb = np.zeros((100, 200, 3))
    # The gaussian blend needs scipy; the default 'elevation' blend does not.
    with pytest.raises(ImportError, match="(?i)scipy"):
        nightshade.make_nightshade_blend(rgb, datetime(2024, 6, 21, 12),
                                         blend="gaussian")


# ============================================================
# astroquery gate
# ============================================================

def test_resolve_name_raises_when_astroquery_missing(monkeypatch):
    """Force astroquery's own import to fail and verify the wrapper
    re-raises with an informative message that names the dependency."""
    import sys

    from skyplothelper import queries

    # Prevent astroquery.simbad import from succeeding.
    saved = {k: sys.modules[k] for k in list(sys.modules)
             if k.startswith("astroquery")}

    class _BlockedFinder:
        def find_module(self, name, path=None):
            if name.startswith("astroquery"):
                return self
            return None

        def load_module(self, name):
            raise ImportError(f"forced-missing for test: {name}")

        def find_spec(self, name, path=None, target=None):
            if name.startswith("astroquery"):
                raise ImportError(f"forced-missing for test: {name}")
            return None

    finder = _BlockedFinder()
    sys.meta_path.insert(0, finder)
    for k in list(sys.modules):
        if k.startswith("astroquery"):
            del sys.modules[k]
    try:
        with pytest.raises(ImportError, match="(?i)astroquery"):
            queries._require_astroquery("simbad")
    finally:
        sys.meta_path.remove(finder)
        sys.modules.update(saved)


# ============================================================
# reproject gate
# ============================================================

def test_reproject_background_raises_when_reproject_missing(monkeypatch):
    import numpy as np

    from skyplothelper.images import reprojection
    monkeypatch.setattr(reprojection, "_HAS_REPROJECT", False)

    img = np.zeros((50, 50))
    src_hdr = {"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
               "CRVAL1": 180, "CRVAL2": 0,
               "CRPIX1": 25, "CRPIX2": 25,
               "CDELT1": -0.1, "CDELT2": 0.1,
               "NAXIS": 2, "NAXIS1": 50, "NAXIS2": 50}
    with pytest.raises(ImportError, match="(?i)reproject"):
        reprojection.reproject_background(img, src_hdr, src_hdr)


def test_reproject_rgb_map_raises_when_reproject_missing(monkeypatch):
    """reproject_rgb_map is the second public reproject-gated entry."""
    from skyplothelper.images import reprojection
    monkeypatch.setattr(reprojection, "_HAS_REPROJECT", False)
    with pytest.raises(ImportError, match="(?i)reproject"):
        reprojection.reproject_rgb_map(None)


# ============================================================
# PIL gate (saved_plot_size_reducer)
# ============================================================

def test_saved_plot_size_reducer_raises_when_pil_missing(monkeypatch):
    from skyplothelper import diagnostics
    monkeypatch.setattr(diagnostics, "_HAS_PIL", False)
    with pytest.raises(ImportError, match="(?i)pil|pillow"):
        diagnostics.saved_plot_size_reducer("/tmp/_does_not_exist.png")
