"""data_plots verification.

Canonical ``tests/test_data_plots.py`` smoke-tests the proper-motion
and displacement helpers on tiny inputs but doesn't cover
``plot_catalog``. This file fills in:

  * Each helper's return-type contract.
  * ``plot_sky_vectors`` color-array / units / cos_dec branches.
  * ``plot_catalog`` with dict, tuple, and astropy Table inputs;
    ``colorby`` / ``sizeby`` / ``label_col`` paths.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.table import Table
from matplotlib.collections import PathCollection
from matplotlib.colors import LogNorm
from matplotlib.quiver import Quiver
from matplotlib.text import Annotation

from skyplothelper.data_plots import (
    CatalogPlot,
    plot_catalog,
    plot_displacement,
    plot_sky_vectors,
)
from skyplothelper.wcs_frame import make_wcs_frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def allsky_axes():
    fig = plt.figure(figsize=(10, 5))
    ax = make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    fig.canvas.draw()
    return fig, ax


# ============================================================
# plot_sky_vectors — return type, units, color array
# ============================================================

def test_plot_sky_vectors_returns_named_tuple(allsky_axes):
    """Returns SkyVectorResult NamedTuple (.quiver + .colorbar)."""
    from skyplothelper.data_plots import SkyVectorResult
    fig, ax = allsky_axes
    rng = np.random.default_rng(0)
    n = 20
    lon = rng.uniform(60, 300, n)
    lat = rng.uniform(-60, 60, n)
    pm_lon = rng.normal(0, 5, n)
    pm_lat = rng.normal(0, 5, n)
    result = plot_sky_vectors(ax, lon, lat, pm_lon, pm_lat,
                                 units="mas", scale=0.1)
    assert isinstance(result, SkyVectorResult)
    assert isinstance(result.quiver, Quiver)
    assert result.colorbar is None  # default add_colorbar=False
    # 4-field NamedTuple: quiver, colorbar, scale, deg_per_pix (the last two
    # feed sky_quiverkey). Use attribute access, not a 2-name unpack.
    assert result._fields == ("quiver", "colorbar", "scale", "deg_per_pix")
    assert result.scale == 0.1
    assert result.deg_per_pix > 0


@pytest.mark.parametrize("units", ["arcsec", "arcmin", "mas", "deg", "uas"])
def test_plot_sky_vectors_each_unit(allsky_axes, units):
    fig, ax = allsky_axes
    result = plot_sky_vectors(ax, [180.0], [30.0], [1.0], [1.0],
                                 units=units, scale=1.0)
    assert isinstance(result.quiver, Quiver)


def test_plot_sky_vectors_invalid_unit_raises(allsky_axes):
    fig, ax = allsky_axes
    with pytest.raises(ValueError, match="(?i)unit"):
        plot_sky_vectors(ax, [0], [0], [0], [0], units="furlongs")


def test_plot_sky_vectors_color_array_propagates(allsky_axes):
    fig, ax = allsky_axes
    rng = np.random.default_rng(1)
    n = 15
    lon = rng.uniform(60, 300, n)
    lat = rng.uniform(-60, 60, n)
    pm_lon = rng.normal(0, 5, n)
    pm_lat = rng.normal(0, 5, n)
    speed = np.sqrt(pm_lon ** 2 + pm_lat ** 2)
    result = plot_sky_vectors(ax, lon, lat, pm_lon, pm_lat,
                                 units="mas", scale=0.1, color=speed)
    assert isinstance(result.quiver, Quiver)


def test_plot_sky_vectors_auto_scale(allsky_axes):
    fig, ax = allsky_axes
    n = 20
    rng = np.random.default_rng(2)
    pm_lon = rng.normal(0, 5, n)
    pm_lat = rng.normal(0, 5, n)
    result = plot_sky_vectors(ax, rng.uniform(60, 300, n),
                                 rng.uniform(-60, 60, n),
                                 pm_lon, pm_lat,
                                 units="mas", scale="auto",
                                 auto_target_deg=2.0)
    assert isinstance(result.quiver, Quiver)


def test_plot_sky_vectors_color_by_magnitude_attaches_colorbar(allsky_axes):
    """``color_by_magnitude=True, add_colorbar=True`` is the canonical
    recipe for the magnitude-coded arrows + auto colorbar pattern."""
    from matplotlib.colorbar import Colorbar
    fig, ax = allsky_axes
    rng = np.random.default_rng(3)
    n = 25
    lon = rng.uniform(60, 300, n)
    lat = rng.uniform(-60, 60, n)
    pm_lon = rng.normal(0, 8, n)
    pm_lat = rng.normal(0, 8, n)
    result = plot_sky_vectors(ax, lon, lat, pm_lon, pm_lat,
                                 units="mas", scale="auto",
                                 color_by_magnitude=True,
                                 add_colorbar=True)
    assert isinstance(result.colorbar, Colorbar)
    # Colorbar should have the auto-generated label "magnitude (<units>)".
    assert "mas" in result.colorbar.ax.get_ylabel() or \
           "mas" in result.colorbar.ax.get_xlabel()


def test_plot_sky_vectors_pivot_default_is_middle(allsky_axes):
    """Default ``pivot='middle'`` matches the cleaner plot_RDEM
    look (arrows centered on data point, not tail-anchored)."""
    fig, ax = allsky_axes
    result = plot_sky_vectors(ax, [180.0], [30.0], [1.0], [1.0],
                                 units="mas", scale=1.0)
    assert result.quiver.pivot == "middle"


# ============================================================
# plot_displacement — returns artist list
# ============================================================

def test_plot_displacement_returns_annotations(allsky_axes):
    """plot_displacement (geodesic, default) returns a list with a shaft
    Line2D plus an arrowhead Annotation per (lon1,lat1)→(lon2,lat2) pair."""
    from matplotlib.lines import Line2D
    fig, ax = allsky_axes
    out = plot_displacement(ax,
                            lon1=[180, 190, 200],
                            lat1=[30, 35, 40],
                            lon2=[182, 191, 198],
                            lat2=[32, 36, 39],
                            color="C3", lw=1.5)
    assert isinstance(out, list)
    # One shaft + one head per source (none on the wrap seam here).
    assert sum(isinstance(a, Line2D) for a in out) == 3
    assert sum(isinstance(a, Annotation) for a in out) == 3

    # The legacy opt-out still returns one Annotation per source.
    legacy = plot_displacement(ax, lon1=[180, 190], lat1=[30, 35],
                               lon2=[182, 191], lat2=[32, 36],
                               geodesic=False)
    assert len(legacy) == 2
    assert all(isinstance(a, Annotation) for a in legacy)


# ============================================================
# plot_catalog — dict, tuple, Table inputs
# ============================================================

def test_plot_catalog_tuple_input_returns_pathcollection(allsky_axes):
    fig, ax = allsky_axes
    rng = np.random.default_rng(3)
    n = 50
    ra = rng.uniform(0, 360, n)
    dec = rng.uniform(-60, 60, n)
    sc = plot_catalog(ax, (ra, dec), color="C0", s=15, alpha=0.7)
    # No colorbar → bare PathCollection (back-compatible default return).
    assert isinstance(sc, PathCollection)


def test_plot_catalog_dict_input_with_named_columns(allsky_axes):
    fig, ax = allsky_axes
    rng = np.random.default_rng(4)
    n = 30
    cat = {"ra_deg": rng.uniform(0, 360, n),
           "dec_deg": rng.uniform(-60, 60, n)}
    sc = plot_catalog(ax, cat, ra_col="ra_deg", dec_col="dec_deg",
                      color="C1", s=15)
    assert isinstance(sc, PathCollection)


def test_plot_catalog_astropy_table_input(allsky_axes):
    fig, ax = allsky_axes
    rng = np.random.default_rng(5)
    n = 25
    tbl = Table({
        "ra": rng.uniform(0, 360, n),
        "dec": rng.uniform(-60, 60, n),
        "magnitude": rng.uniform(10, 18, n),
    })
    sc = plot_catalog(ax, tbl, ra_col="ra", dec_col="dec",
                      color="C2", s=15)
    assert isinstance(sc, PathCollection)


def test_plot_catalog_colorby_uses_colormap(allsky_axes):
    """When colorby= is set, ``color=`` is ignored — the scatter's
    array carries the values."""
    fig, ax = allsky_axes
    rng = np.random.default_rng(6)
    n = 40
    cat = {"ra": rng.uniform(0, 360, n),
           "dec": rng.uniform(-60, 60, n),
           "z": rng.uniform(0, 3, n)}
    sc = plot_catalog(ax, cat, colorby="z", cmap="plasma",
                      vmin=0, vmax=3, s=15)
    arr = sc.get_array()
    assert arr is not None
    assert len(arr) == n


def test_plot_catalog_sizeby_scales_to_smin_smax(allsky_axes):
    fig, ax = allsky_axes
    rng = np.random.default_rng(7)
    n = 30
    cat = {"ra": rng.uniform(0, 360, n),
           "dec": rng.uniform(-60, 60, n),
           "mag": rng.uniform(10, 18, n)}
    sc = plot_catalog(ax, cat, sizeby="mag", smin=20, smax=200)
    sizes = sc.get_sizes()
    assert sizes.min() >= 20 - 1e-6
    assert sizes.max() <= 200 + 1e-6


def test_plot_catalog_size_vlim_shares_scale_and_stashes_info(allsky_axes):
    # Fixed raw bounds make equal values render equal across separate calls
    # (the multi-marker case), and the scaling is stashed for a size legend.
    fig, ax = allsky_axes
    kw = dict(sizeby="n", size_vlim=(1, 100), smin=10, smax=200)
    a = plot_catalog(ax, {"ra": [10.0], "dec": [0.0], "n": [50.0]}, **kw)
    b = plot_catalog(ax, {"ra": [20.0], "dec": [0.0], "n": [50.0]},
                     marker="s", **kw)
    assert np.isclose(a.get_sizes()[0], b.get_sizes()[0])
    # A value at the upper bound maps to smax; the stash records the scaling.
    c = plot_catalog(ax, {"ra": [0.0], "dec": [0.0], "n": [100.0]}, **kw)
    assert np.isclose(c.get_sizes()[0], 200.0)
    assert c._sph_size_info.smin == 10 and c._sph_size_info.smax == 200


def test_plot_catalog_label_col_adds_text_artists(allsky_axes):
    fig, ax = allsky_axes
    n_before = len(ax.texts)
    cat = {"ra": [60, 120, 180, 240, 300],
           "dec": [30, -10, 0, 40, -30],
           "name": ["A", "B", "C", "D", "E"]}
    plot_catalog(ax, cat, label_col="name", s=20)
    n_after = len(ax.texts)
    assert (n_after - n_before) == 5


# ---------------------------------------------------------------------------
# plot_catalog — scaling / colormap / colorbar convenience knobs
# ---------------------------------------------------------------------------

def test_plot_catalog_size_scale_sqrt_compresses_high_end(allsky_axes):
    """A sqrt scale stays within [smin, smax] but reshapes the spacing:
    the gap between the two largest inputs shrinks relative to linear."""
    fig, ax = allsky_axes
    cat = {"ra": [0, 60, 120, 180], "dec": [0, 0, 0, 0],
           "v": [1.0, 4.0, 9.0, 16.0]}
    lin = plot_catalog(ax, cat, sizeby="v", smin=10, smax=200,
                       size_scale="linear").get_sizes()
    sq = plot_catalog(ax, cat, sizeby="v", smin=10, smax=200,
                      size_scale="sqrt").get_sizes()
    for arr in (lin, sq):
        assert arr.min() >= 10 - 1e-6
        assert arr.max() <= 200 + 1e-6
        # monotonic in the (monotonic) input
        assert np.all(np.diff(arr) > 0)
    # sqrt evenly spaces sqrt(v)=1,2,3,4 → equal steps; linear is bottom-heavy.
    assert np.ptp(np.diff(sq)) < np.ptp(np.diff(lin))


def test_plot_catalog_size_scale_callable_matches_manual(allsky_axes):
    fig, ax = allsky_axes
    raw = np.array([1.0, 10.0, 100.0, 1000.0])
    cat = {"ra": [0, 60, 120, 180], "dec": [0, 0, 0, 0], "v": raw}
    sizes = plot_catalog(ax, cat, sizeby="v", smin=0, smax=100,
                         size_scale=np.log10).get_sizes()
    # log10 of 1..1000 is 0,1,2,3 → evenly mapped onto [0, 100]
    np.testing.assert_allclose(sizes, [0, 100 / 3, 200 / 3, 100], atol=1e-6)


def test_plot_catalog_size_scale_log_guards_nonpositive(allsky_axes):
    fig, ax = allsky_axes
    cat = {"ra": [0, 60, 120], "dec": [0, 0, 0], "v": [0.0, 10.0, 100.0]}
    with pytest.warns(UserWarning, match="non-positive"):
        sizes = plot_catalog(ax, cat, sizeby="v", smin=10, smax=100,
                             size_scale="log").get_sizes()
    assert np.all(np.isfinite(sizes))
    assert sizes.min() >= 10 - 1e-6


def test_plot_catalog_color_scale_log_sets_lognorm(allsky_axes):
    fig, ax = allsky_axes
    cat = {"ra": [0, 60, 120, 180], "dec": [0, 0, 0, 0],
           "err": [0.01, 0.1, 1.0, 10.0]}
    sc = plot_catalog(ax, cat, colorby="err", color_scale="log",
                      vmin=0.01, vmax=10.0)
    assert isinstance(sc.norm, LogNorm)
    assert sc.norm.vmin == 0.01
    assert sc.norm.vmax == 10.0


def test_plot_catalog_color_scale_sqrt_sets_powernorm(allsky_axes):
    from matplotlib.colors import PowerNorm
    fig, ax = allsky_axes
    cat = {"ra": [0, 60, 120, 180], "dec": [0, 0, 0, 0],
           "v": [0.0, 1.0, 4.0, 9.0]}
    sc = plot_catalog(ax, cat, colorby="v", color_scale="sqrt",
                      vmin=0.0, vmax=9.0)
    assert isinstance(sc.norm, PowerNorm)
    assert sc.norm.gamma == 0.5
    assert sc.norm.vmin == 0.0
    assert sc.norm.vmax == 9.0


def test_plot_catalog_color_scale_normalize_used_as_is(allsky_axes):
    fig, ax = allsky_axes
    norm = LogNorm(vmin=0.5, vmax=5.0)
    cat = {"ra": [0, 60, 120], "dec": [0, 0, 0], "err": [1.0, 2.0, 3.0]}
    sc = plot_catalog(ax, cat, colorby="err", color_scale=norm)
    assert sc.norm is norm


def test_plot_catalog_cmap_range_truncates_colormap(allsky_axes):
    fig, ax = allsky_axes
    cat = {"ra": [0, 60, 120], "dec": [0, 0, 0], "z": [0.0, 0.5, 1.0]}
    full = plot_catalog(ax, cat, colorby="z", cmap="viridis")
    trunc = plot_catalog(ax, cat, colorby="z", cmap="viridis",
                         cmap_range=(0.3, 0.7))
    # The truncated map's value at 0.0 equals the full map's value at 0.3.
    np.testing.assert_allclose(trunc.cmap(0.0), full.cmap(0.3), atol=1e-6)
    np.testing.assert_allclose(trunc.cmap(1.0), full.cmap(0.7), atol=1e-6)


def test_plot_catalog_returns_colorbar_when_requested(allsky_axes):
    fig, ax = allsky_axes
    cat = {"ra": [0, 60, 120], "dec": [0, 0, 0], "z": [0.1, 0.5, 0.9]}
    res = plot_catalog(ax, cat, colorby="z", cbar=True, cbar_label="redshift")
    assert isinstance(res, CatalogPlot)
    assert res.colorbar is not None
    assert res.colorbar.ax.get_ylabel() == "redshift"


def test_plot_catalog_cbar_format_and_ticks_applied(allsky_axes):
    from matplotlib.ticker import StrMethodFormatter
    fig, ax = allsky_axes
    cat = {"ra": [0, 60, 120], "dec": [0, 0, 0], "z": [0.1, 0.5, 0.9]}
    res = plot_catalog(ax, cat, colorby="z", cbar=True,
                       cbar_ticks=[0.2, 0.4, 0.6, 0.8],
                       cbar_format="{x:.3f}")
    cb = res.colorbar
    assert isinstance(cb.formatter, StrMethodFormatter)
    np.testing.assert_allclose(cb.get_ticks(), [0.2, 0.4, 0.6, 0.8])


def test_plot_catalog_size_legend_adds_representative_markers(allsky_axes):
    fig, ax = allsky_axes
    cat = {"ra": np.linspace(0, 350, 20), "dec": np.zeros(20),
           "n": np.linspace(1.0, 100.0, 20)}
    plot_catalog(ax, cat, sizeby="n", smin=10, smax=200,
                 size_legend=True, size_legend_num=5)
    legs = [a for a in ax.get_children()
            if a.__class__.__name__ == "Legend"]
    assert legs, "no size legend attached"
    leg = legs[-1]
    assert len(leg.get_texts()) == 5
    assert leg.get_title().get_text() == "n"
    # Legend marker sizes increase with the encoded value.
    sizes = [h.get_markersize() for h in leg.legend_handles]
    assert all(np.diff(sizes) > 0)


def test_plot_catalog_size_legend_warns_without_sizeby(allsky_axes):
    fig, ax = allsky_axes
    cat = {"ra": [0, 60, 120], "dec": [0, 0, 0]}
    with pytest.warns(UserWarning, match="no valid sizeby"):
        plot_catalog(ax, cat, size_legend=True)


# ---------------------------------------------------------------------------
# plot_catalog — coordinate frame + lon/lat column generality
# ---------------------------------------------------------------------------

def test_plot_catalog_frame_converts_galactic_to_equatorial(allsky_axes):
    """frame='galactic' must convert l/b onto the (ICRS) plot — a galactic
    point and its ICRS-equivalent must land at the same display position."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    fig, ax = allsky_axes
    # Galactic center.
    gc = SkyCoord(l=0 * u.deg, b=0 * u.deg, frame="galactic")
    eq = gc.icrs

    sc_gal = plot_catalog(ax, {"l": [0.0], "b": [0.0]},
                          lon_col="l", lat_col="b", frame="galactic")
    sc_eq = plot_catalog(ax, {"ra": [eq.ra.deg], "dec": [eq.dec.deg]},
                         frame=None)
    fig.canvas.draw()
    d_gal = sc_gal.get_offset_transform().transform(sc_gal.get_offsets())
    d_eq = sc_eq.get_offset_transform().transform(sc_eq.get_offsets())
    np.testing.assert_allclose(d_gal, d_eq, atol=1.0)  # within 1 px

    # Sanity: NOT supplying frame (treating l/b as native ICRS) lands elsewhere.
    sc_naive = plot_catalog(ax, {"ra": [0.0], "dec": [0.0]}, frame=None)
    fig.canvas.draw()
    d_naive = sc_naive.get_offset_transform().transform(sc_naive.get_offsets())
    assert not np.allclose(d_gal, d_naive, atol=5.0)


def test_plot_catalog_lon_lat_aliases(allsky_axes):
    fig, ax = allsky_axes
    cat = {"glon": [10.0, 200.0], "glat": [5.0, -30.0]}
    sc = plot_catalog(ax, cat, lon_col="glon", lat_col="glat",
                      frame="galactic")
    assert isinstance(sc, PathCollection)
    assert len(sc.get_offsets()) == 2


def test_plot_catalog_autodetects_galactic_columns(allsky_axes):
    """An l/b table resolves without an explicit lon/lat col."""
    fig, ax = allsky_axes
    cat = {"l": [10.0, 200.0, 350.0], "b": [5.0, -30.0, 60.0]}
    sc = plot_catalog(ax, cat, frame="galactic")
    assert isinstance(sc, PathCollection)
    assert len(sc.get_offsets()) == 3
