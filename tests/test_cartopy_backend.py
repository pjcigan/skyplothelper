"""Smoke tests for skyplothelper.cartopy_backend (skips without cartopy)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from skyplothelper.cartopy_backend import (
    _HAS_CARTOPY,
    list_cartopy_projections,
    make_cartopy_frame,
)

pytestmark = pytest.mark.skipif(not _HAS_CARTOPY, reason="cartopy not installed")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_make_cartopy_frame_smoke():
    fig = plt.figure()
    ax = make_cartopy_frame(111, projection="mollweide", center=180, fig=fig)
    assert ax is not None


def test_list_cartopy_projections_runs(capsys):
    list_cartopy_projections()
    out = capsys.readouterr().out
    assert "Mollweide" in out


def test_cartopy_frame_center_accepts_skycoord():
    """make_cartopy_frame mirrors make_wcs_frame: a SkyCoord center converts
    into the frame being built, not blindly to ICRS."""
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from astropy.coordinates import SkyCoord

    from skyplothelper.cartopy_backend import make_cartopy_frame
    gal = SkyCoord(120.0, 30.0, unit="deg", frame="galactic")
    ax = make_cartopy_frame(111, projection="mollweide", center=gal,
                            frame="galactic")
    assert ax is not None
    plt.close("all")


# ---- label color: theme-aware default + an actual knob (audit B+C) ----

import skyplothelper as sph  # noqa: E402


def _gridliners(ax):
    return [a for a in ax.get_children() if type(a).__name__ == "Gridliner"]


def test_cartopy_label_color_is_muted_not_full_ink():
    """These labels were a deliberate '0.3'. Resolving them to a tick rcParam
    would darken every existing light-theme figure; only dark was broken."""
    import matplotlib
    matplotlib.rcdefaults()
    sph.set_style(base="standard")
    ax = sph.make_cartopy_frame(projection="Mollweide")
    gls = _gridliners(ax)
    assert gls
    assert all(g.xlabel_style["color"] == "0.3" for g in gls)
    plt.close(ax.figure)
    matplotlib.rcdefaults()


def test_cartopy_label_color_lightens_on_a_dark_theme():
    import matplotlib
    matplotlib.rcdefaults()
    sph.set_style(base="standard", theme="dark_sky")
    ax = sph.make_cartopy_frame(projection="Mollweide")
    gls = _gridliners(ax)
    assert gls
    assert all(g.xlabel_style["color"] == "0.70" for g in gls)
    plt.close(ax.figure)
    matplotlib.rcdefaults()
    sph.set_style(base="standard")


def test_cartopy_label_color_override():
    ax = sph.make_cartopy_frame(projection="Mollweide", label_color="#ff00ff")
    gls = _gridliners(ax)
    assert gls
    assert all(g.xlabel_style["color"] == "#ff00ff" for g in gls)
    plt.close(ax.figure)


def test_cartopy_unlabeled_fallback_uses_the_same_color():
    """The ax.text fallback runs only on cartopy versions that cannot label a
    projection, so force it rather than leave the branch unverified."""
    import cartopy.mpl.geoaxes as cga
    import matplotlib
    original = cga.GeoAxes.gridlines

    def picky(self, *a, **kw):
        if kw.get("draw_labels"):
            raise TypeError("simulated: cannot label this projection")
        return original(self, *a, **kw)

    cga.GeoAxes.gridlines = picky
    try:
        matplotlib.rcdefaults()
        sph.set_style(base="standard")
        ax = sph.make_cartopy_frame(projection="Mollweide")
        assert ax.texts and all(t.get_color() == "0.3" for t in ax.texts)
        plt.close(ax.figure)

        ax = sph.make_cartopy_frame(projection="Mollweide",
                                    label_color="#ff00ff")
        assert ax.texts and all(t.get_color() == "#ff00ff" for t in ax.texts)
        plt.close(ax.figure)
    finally:
        cga.GeoAxes.gridlines = original
        matplotlib.rcdefaults()
        sph.set_style(base="standard")
