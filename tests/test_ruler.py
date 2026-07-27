"""Tests for the :class:`skyplothelper.Ruler` class.

Coverage: construction in data + world coords, tick generation (auto +
explicit + n_ticks), label unit auto-selection, straight vs geodesic
modes, component setters, axes wiring, and public API surface.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402

import skyplothelper as sph  # noqa: E402
from skyplothelper._text_layout import (  # noqa: E402
    _normalize_readable_angle,
    _resolve_rotation_deg,
    _resolve_text_anchor,
)
from skyplothelper.overlays.ruler import (  # noqa: E402
    Ruler,
    _format_angle_label,
    _format_converted_label,
    _format_numeric,
    _make_distance_converter,
    _make_redshift_converter,
    _nice_interval,
    _normalize_convert,
    _resolve_cosmology,
    _resolve_label_side_sign,
    _resolve_title_side_sign,
)

# ---- helpers ----------------------------------------------------------------

def _plain_axes():
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    fig.canvas.draw()
    return fig, ax


def _wcs_axes(cdelt_asec=2.0, npix=100):
    """A TAN-projected WCSAxes with a configurable pixel scale."""
    from skyplothelper.wcs_frame import make_wcs_frame
    cdelt_deg = cdelt_asec / 3600.0
    fig = plt.figure(figsize=(5, 5))
    ax = make_wcs_frame(111, projection="TAN", center=(180.0, 0.0),
                        cdelt=cdelt_deg, npix=(npix, npix), fig=fig)
    fig.canvas.draw()
    return fig, ax


# ---- _nice_interval --------------------------------------------------------

@pytest.mark.parametrize("span,target,expected", [
    # raw=span/target_n, then largest of (1,2,5)*10^n that is <= raw.
    (100.0, 4, 20.0),    # raw=25 → 20 → 5 intervals
    (47.0, 4, 10.0),     # raw=11.75 → 10 → ~4–5 intervals
    (10.0, 4, 2.0),      # raw=2.5  → 2  → 5 intervals
    (4.0, 4, 1.0),       # raw=1    → 1  → 4 intervals
    (1.0, 4, 0.2),       # raw=0.25 → 0.2 → 5 intervals
])
def test_nice_interval_picks_one_two_five_step(span, target, expected):
    """Nice interval lands on 1/2/5 × 10^n with at least ``target_n``
    intervals across the line (algorithm picks the floor, not the
    ceil)."""
    got = _nice_interval(span, target_n=target)
    assert got == pytest.approx(expected)


def test_nice_interval_handles_zero_and_negative_span():
    """Pathological spans collapse to a sentinel — no division-by-zero,
    no infinite intervals."""
    assert _nice_interval(0.0) == 1.0
    assert _nice_interval(-5.0) == 1.0
    assert _nice_interval(float("nan")) == 1.0


# ---- _format_angle_label ---------------------------------------------------

def test_format_angle_label_auto_picks_arcsec_below_60():
    assert _format_angle_label(30.0, unit="auto") == "30″"


def test_format_angle_label_auto_picks_arcmin_below_3600():
    assert _format_angle_label(120.0, unit="auto") == "2′"


def test_format_angle_label_auto_picks_deg_above_3600():
    assert _format_angle_label(7200.0, unit="auto") == "2°"


def test_format_angle_label_explicit_unit_overrides_auto():
    assert _format_angle_label(60.0, unit="arcsec") == "60″"
    assert _format_angle_label(60.0, unit="arcmin") == "1′"


def test_format_angle_label_pix_unit():
    assert _format_angle_label(42.0, unit="pix") == "42 px"


def test_format_angle_label_rejects_unknown_unit():
    with pytest.raises(ValueError, match="label_unit must be"):
        _format_angle_label(30.0, unit="parsec")


def test_format_angle_label_auto_promotes_to_mas_below_arcsec():
    # ~10 mas (0.0098") used to label as 0.0098"; now reads in mas
    assert _format_angle_label(0.0098, unit="auto") == "9.8 mas"
    assert _format_angle_label(0.5, unit="auto") == "500 mas"


def test_format_angle_label_auto_promotes_to_uas_then_nas():
    assert _format_angle_label(5e-4, unit="auto") == "500 μas"
    assert _format_angle_label(5e-7, unit="auto") == "500 nas"


def test_format_angle_label_explicit_mas_uas_nas():
    assert _format_angle_label(0.0098, unit="mas") == "9.8 mas"
    assert _format_angle_label(5e-4, unit="uas") == "500 μas"
    assert _format_angle_label(5e-7, unit="nas") == "500 nas"
    # 'μas' accepted as an alias for 'uas'
    assert _format_angle_label(5e-4, unit="μas") == "500 μas"


def test_ruler_accepts_mas_uas_nas_units():
    # constructor + update() accept the new units (incl. the 'μas' alias)
    for u in ("mas", "uas", "μas", "nas"):
        Ruler((0, 0), (10, 0), label_unit=u)


def test_ruler_sub_arcsec_labels_in_mas():
    """A VLBI-scale (sub-arcsec) ruler's ticks all fall in the mas regime, so
    auto labels read in mas instead of a tiny fraction of an arcsec — the
    reported case (~10 mas labeling as 0.0098″)."""
    fig = plt.figure()
    ax = sph.make_wcs_frame(111, "TAN", center=(83.6, 22.0),
                            cdelt=4e-4 / 3600.0, npix=200, fig=fig)
    fig.canvas.draw()
    r = Ruler((40, 80), (160, 80), ax=ax, label_unit="auto",
              coord_type="pixel")
    r.add_to(ax)
    labels = [t.get_text() for t in r._label_artists if t.get_text()]
    assert labels, "no labels rendered"
    assert all(lbl.endswith("mas") for lbl in labels), labels
    plt.close(fig)


# ---- basic construction -----------------------------------------------------

def test_construct_stores_endpoints():
    r = Ruler((10, 20), (30, 50))
    assert r.xy1 == (10.0, 20.0)
    assert r.xy2 == (30.0, 50.0)


def test_default_geodesic_is_false():
    r = Ruler((10, 20), (30, 50))
    assert r.geodesic is False


def test_geodesic_kwarg_propagates():
    r = Ruler((10, 20), (30, 50), geodesic=True)
    assert r.geodesic is True


def test_pixscale_kwarg_overrides_axes_scale():
    """An explicit ``pixscale_asec=`` always wins over what the axes'
    WCS would report (useful for anisotropic WCSes)."""
    fig, ax = _wcs_axes(cdelt_asec=2.0)
    r = Ruler((10, 10), (80, 10), ax=ax, pixscale_asec=99.0)
    assert r.pixscale_asec == 99.0
    plt.close(fig)


def test_pixscale_inferred_from_wcsaxes():
    fig, ax = _wcs_axes(cdelt_asec=2.0)
    r = Ruler((10, 10), (80, 10), ax=ax)
    assert r.pixscale_asec == pytest.approx(2.0)
    plt.close(fig)


def test_pixscale_unknown_for_plain_axes():
    fig, ax = _plain_axes()
    r = Ruler((10, 10), (80, 10), ax=ax)
    assert r.pixscale_asec is None
    plt.close(fig)


def test_invalid_tick_side_raises():
    with pytest.raises(ValueError, match="tick_side must be"):
        Ruler((0, 0), (10, 0), tick_side="diagonal")


def test_invalid_label_unit_raises():
    with pytest.raises(ValueError, match="label_unit must be"):
        Ruler((0, 0), (10, 0), label_unit="furlong")


def test_n_ticks_must_be_at_least_two():
    with pytest.raises(ValueError, match="n_ticks must be"):
        Ruler((0, 0), (10, 0), n_ticks=1)


# ---- distance ---------------------------------------------------------------

def test_angular_distance_uses_pixscale_for_straight_line():
    """Straight-line distance = ``hypot(dx, dy) * pixscale``."""
    r = Ruler((10, 10), (40, 50), pixscale_asec=0.5)
    expected = float(np.hypot(30, 40)) * 0.5   # 50 px * 0.5 = 25 arcsec
    assert r.angular_distance_asec() == pytest.approx(expected)


def test_angular_distance_returns_none_without_scale():
    r = Ruler((10, 10), (40, 50))
    assert r.angular_distance_asec() is None


def test_angular_distance_geodesic_uses_great_circle():
    """For geodesic mode on a WCSAxes, the distance should match the
    great-circle separation between the endpoints' world coordinates."""
    fig, ax = _wcs_axes(cdelt_asec=2.0, npix=100)
    # Pick two pixel positions 30 px apart at the field center
    x1, y1 = 35, 50
    x2, y2 = 65, 50
    r = Ruler((x1, y1), (x2, y2), ax=ax, geodesic=True,
              coord_type="pixel")
    # Expected: ~30 px * 2 asec/px = ~60 arcsec at the tangent point
    got = r.angular_distance_asec()
    assert got == pytest.approx(60.0, rel=0.02)
    plt.close(fig)


# ---- add_to / remove --------------------------------------------------------

def test_add_to_attaches_main_line():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0)
    r.add_to(ax)
    assert r.line_artist is not None
    assert r.line_artist in ax.lines
    plt.close(fig)


def test_add_to_returns_self_for_chaining():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50))
    assert r.add_to(ax) is r
    plt.close(fig)


def test_add_to_attaches_tick_and_label_artists():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0)
    r.add_to(ax)
    assert len(r.tick_artists) > 0
    assert len(r.label_artists) > 0
    for tick in r.tick_artists:
        assert isinstance(tick, Line2D)
        assert tick in ax.lines
    for lab in r.label_artists:
        assert isinstance(lab, Text)
        assert lab in ax.texts
    plt.close(fig)


def test_remove_strips_all_artists():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0)
    r.add_to(ax)
    line = r.line_artist
    ticks = r.tick_artists
    labels = r.label_artists
    r.remove()
    assert line not in ax.lines
    for t in ticks:
        assert t not in ax.lines
    for lab in labels:
        assert lab not in ax.texts
    assert r.line_artist is None
    assert r.tick_artists == []
    assert r.label_artists == []
    plt.close(fig)


def test_remove_is_safe_if_never_added():
    """remove() before add_to() should be a no-op, not raise."""
    r = Ruler((10, 50), (90, 50))
    r.remove()  # no-op


# ---- tick generation --------------------------------------------------------

def test_both_sided_ticks_produces_two_lines_per_position():
    """``tick_side='both'`` should produce 2 tick segments per position."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0, tick_side="both")
    r.add_to(ax)
    # Line spans 80 px → positions at 0, 20, 40, 60, 80 = 5 positions
    # × 2 sides = 10 tick lines
    assert len(r.tick_artists) == 10
    assert len(r.label_artists) == 5
    plt.close(fig)


def test_one_sided_ticks_produces_one_line_per_position():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0, tick_side="left")
    r.add_to(ax)
    assert len(r.tick_artists) == 5
    plt.close(fig)


def test_tick_side_none_suppresses_ticks_but_keeps_labels():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0, tick_side="none")
    r.add_to(ax)
    assert len(r.tick_artists) == 0
    # Line + labels still present
    assert r.line_artist is not None
    assert len(r.label_artists) == 5
    plt.close(fig)


def test_labels_false_suppresses_labels_but_keeps_ticks():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0, labels=False)
    r.add_to(ax)
    assert len(r.label_artists) == 0
    assert len(r.tick_artists) > 0
    plt.close(fig)


def test_explicit_n_ticks_uses_that_many_positions():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              n_ticks=5, tick_side="both")
    r.add_to(ax)
    # 5 positions × 2 sides = 10
    assert len(r.tick_artists) == 10
    plt.close(fig)


def test_auto_tick_interval_falls_back_to_pixels_without_scale():
    """Without a pixel scale, auto-nice intervals are still applied —
    just expressed in pixels rather than arcsec."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50))  # no pixscale
    r.add_to(ax)
    assert len(r.tick_artists) > 0
    # First label should be in pixel units when no scale available
    assert " px" in r.label_artists[0].get_text()
    plt.close(fig)


def test_zero_length_ruler_produces_no_ticks():
    fig, ax = _plain_axes()
    r = Ruler((50, 50), (50, 50), pixscale_asec=1.0)
    r.add_to(ax)
    # Degenerate — no ticks or labels emitted, but no crash
    assert r.tick_artists == []
    assert r.label_artists == []
    plt.close(fig)


# ---- label unit selection ---------------------------------------------------

def test_label_unit_auto_picks_arcsec_for_short_ruler():
    """A 25 arcsec ruler should auto-label in arcsec."""
    fig, ax = _plain_axes()
    r = Ruler((0, 50), (50, 50), pixscale_asec=0.5,
              tick_interval=5.0)
    r.add_to(ax)
    # All labels under 60 arcsec → arcsec units
    for lab in r.label_artists:
        assert "″" in lab.get_text()
    plt.close(fig)


def test_label_unit_explicit_pins_units():
    fig, ax = _plain_axes()
    r = Ruler((0, 50), (50, 50), pixscale_asec=0.5,
              tick_interval=5.0, label_unit="arcmin")
    r.add_to(ax)
    for lab in r.label_artists:
        assert "′" in lab.get_text()
    plt.close(fig)


def test_label_fmt_callable_overrides_default():
    """A custom formatter receives ``(value, unit)`` and its return
    string becomes the label."""
    fig, ax = _plain_axes()

    def fmt(value, unit):
        return f"D={value:.1f}"

    r = Ruler((0, 50), (50, 50), pixscale_asec=0.5,
              tick_interval=10.0, label_fmt=fmt)
    r.add_to(ax)
    for lab in r.label_artists:
        assert lab.get_text().startswith("D=")
    plt.close(fig)


# ---- geodesic mode ----------------------------------------------------------

def test_geodesic_samples_along_great_circle_arc():
    """Geodesic mode should produce a polyline with > 2 vertices
    (sampled along the arc), not just the two endpoints."""
    fig, ax = _wcs_axes(cdelt_asec=2.0, npix=100)
    r = Ruler((30, 50), (70, 50), ax=ax, geodesic=True,
              n_geodesic_pts=32, coord_type="pixel")
    r.add_to(ax)
    xs, ys = r.line_artist.get_data()
    assert len(xs) == 32
    plt.close(fig)


def test_straight_line_has_two_vertices():
    fig, ax = _plain_axes()
    r = Ruler((30, 50), (70, 50))
    r.add_to(ax)
    xs, ys = r.line_artist.get_data()
    assert len(xs) == 2
    plt.close(fig)


# ---- from_world factory -----------------------------------------------------

def test_from_world_accepts_skycoord_pair():
    from astropy.coordinates import SkyCoord

    fig, ax = _wcs_axes(cdelt_asec=2.0, npix=100)
    c1 = SkyCoord(180.0, -0.01, unit="deg")
    c2 = SkyCoord(180.0, +0.01, unit="deg")
    r = Ruler.from_world(c1, c2, ax=ax)
    assert isinstance(r, Ruler)
    # Endpoints should land near pixel center (~50, 50) ± a few px
    assert 30 < r.xy1[0] < 70
    assert 30 < r.xy2[0] < 70
    plt.close(fig)


def test_from_world_accepts_lon_lat_tuples():
    """Tuples are interpreted as degrees in the WCS frame."""
    fig, ax = _wcs_axes(cdelt_asec=2.0, npix=100)
    r = Ruler.from_world((180.0, -0.01), (180.0, +0.01), ax=ax)
    assert isinstance(r, Ruler)
    plt.close(fig)


def test_from_world_defaults_to_geodesic_true():
    fig, ax = _wcs_axes(cdelt_asec=2.0)
    r = Ruler.from_world((180.0, -0.01), (180.0, +0.01), ax=ax)
    assert r.geodesic is True
    plt.close(fig)


def test_from_world_geodesic_false_override_works():
    fig, ax = _wcs_axes(cdelt_asec=2.0)
    r = Ruler.from_world((180.0, -0.01), (180.0, +0.01), ax=ax,
                         geodesic=False)
    assert r.geodesic is False
    plt.close(fig)


def test_from_world_requires_wcs_axes():
    fig, ax = _plain_axes()
    with pytest.raises(ValueError, match="requires an axes with a WCS"):
        Ruler.from_world((180.0, 0.0), (180.1, 0.0), ax=ax)
    plt.close(fig)


def test_from_world_rejects_bad_endpoint_input():
    fig, ax = _wcs_axes(cdelt_asec=2.0)
    with pytest.raises(TypeError, match="expected SkyCoord"):
        Ruler.from_world(180.0, (180.1, 0.0), ax=ax)
    plt.close(fig)


def test_from_world_forwards_kwargs():
    fig, ax = _wcs_axes(cdelt_asec=2.0)
    r = Ruler.from_world((180.0, 0.0), (180.005, 0.0), ax=ax,
                          color="C2", lw=2.0, tick_side="left")
    assert r._color == "C2"
    assert r._lw == 2.0
    assert r._tick_side == "left"
    plt.close(fig)


# ---- component setters ------------------------------------------------------

def test_set_line_updates_attached_artist_in_place():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), color="k", lw=1.0)
    r.add_to(ax)
    r.set_line(color="C0", lw=2.5, ls="--")
    assert r.line_artist.get_linewidth() == 2.5
    assert r.line_artist.get_linestyle() == "--"
    plt.close(fig)


def test_set_line_returns_self_for_chaining():
    r = Ruler((10, 50), (90, 50))
    assert r.set_line(color="red") is r


def test_set_ticks_recolors_existing_tick_artists():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0, color="k")
    r.add_to(ax)
    r.set_ticks(color="C3", lw=2.0)
    for tick in r.tick_artists:
        c = tick.get_color()
        # C3 is some red-ish — at minimum, not pure black
        assert c == "C3" or c[0] > 0.5  # tolerant
        assert tick.get_linewidth() == 2.0
    plt.close(fig)


def test_set_ticks_validates_side_kwarg():
    r = Ruler((0, 0), (10, 0))
    with pytest.raises(ValueError, match="tick_side must be"):
        r.set_ticks(side="oblique")


def test_set_labels_updates_attached_text_in_place():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0)
    r.add_to(ax)
    r.set_labels(color="C2", fontsize=14)
    for lab in r.label_artists:
        assert lab.get_fontsize() == 14
    plt.close(fig)


def test_set_labels_validates_unit_kwarg():
    r = Ruler((0, 0), (10, 0))
    with pytest.raises(ValueError, match="label_unit must be"):
        r.set_labels(unit="lightyear")


# ---- public API surface -----------------------------------------------------

def test_ruler_exported_at_package_top_level():
    assert sph.Ruler is Ruler


# ---- convert= helpers -------------------------------------------------------

def test_resolve_cosmology_accepts_name_string():
    from astropy import cosmology
    assert _resolve_cosmology("Planck18") is cosmology.Planck18


def test_resolve_cosmology_accepts_instance():
    from astropy import cosmology
    assert _resolve_cosmology(cosmology.WMAP9) is cosmology.WMAP9


def test_resolve_cosmology_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown cosmology"):
        _resolve_cosmology("BogusCosmo7")


def test_resolve_cosmology_rejects_non_string_non_cosmology():
    with pytest.raises(TypeError, match="Cosmology instance"):
        _resolve_cosmology(42)


def test_make_redshift_converter_returns_fn_and_unit():
    fn, unit = _make_redshift_converter(redshift=0.5, unit="kpc")
    assert callable(fn)
    assert unit == "kpc"
    # ~6.3 kpc/asec at z=0.5 (Planck18)
    assert 5.0 < fn(1.0) < 7.5


def test_make_redshift_converter_unit_conversion():
    """kpc → Mpc should scale by 1/1000."""
    fn_kpc, _ = _make_redshift_converter(redshift=0.5, unit="kpc")
    fn_mpc, _ = _make_redshift_converter(redshift=0.5, unit="Mpc")
    assert fn_mpc(1.0) == pytest.approx(fn_kpc(1.0) / 1000.0, rel=1e-9)


def test_make_redshift_converter_rejects_non_length_unit():
    with pytest.raises(ValueError, match="length-like astropy unit"):
        _make_redshift_converter(redshift=0.5, unit="kg")


def test_make_distance_converter_small_angle():
    """1 arcsec at 1 pc = 1 AU (definition of parsec)."""
    fn, unit = _make_distance_converter(
        distance=1.0, distance_unit="pc", unit="au")
    assert fn(1.0) == pytest.approx(1.0, rel=1e-5)
    assert unit == "au"


def test_make_distance_converter_scales_linearly_with_distance():
    fn_1, _ = _make_distance_converter(
        distance=1.0, distance_unit="pc", unit="au")
    fn_10, _ = _make_distance_converter(
        distance=10.0, distance_unit="pc", unit="au")
    assert fn_10(1.0) == pytest.approx(10 * fn_1(1.0))


def test_make_distance_converter_rejects_non_length_unit():
    with pytest.raises(ValueError, match="length-like astropy units"):
        _make_distance_converter(
            distance=1.0, distance_unit="pc", unit="kg")


def test_normalize_convert_none_returns_none_tuple():
    fn, unit = _normalize_convert(None)
    assert fn is None
    assert unit is None


def test_normalize_convert_callable_passes_through():
    f = lambda x: x * 2  # noqa: E731
    fn, unit = _normalize_convert(f, convert_unit="kpc")
    assert fn is f
    assert unit == "kpc"


def test_normalize_convert_callable_unit_optional():
    """A bare callable without convert_unit produces a None unit
    string — labels then show just the numeric value."""
    fn, unit = _normalize_convert(lambda x: x)
    assert callable(fn)
    assert unit is None


def test_normalize_convert_dict_redshift_dispatches():
    fn, unit = _normalize_convert(dict(redshift=0.5, unit="kpc"))
    assert callable(fn)
    assert unit == "kpc"


def test_normalize_convert_dict_distance_dispatches():
    fn, unit = _normalize_convert(
        dict(distance=10, distance_unit="pc", unit="au"))
    assert callable(fn)
    assert unit == "au"


def test_normalize_convert_dict_missing_keys_raises():
    with pytest.raises(ValueError, match="must contain 'redshift' or"):
        _normalize_convert(dict(unit="kpc"))


def test_normalize_convert_rejects_non_dict_non_callable():
    with pytest.raises(TypeError, match="None, callable, or dict"):
        _normalize_convert(42)


def test_format_converted_label_with_unit():
    assert _format_converted_label(12.345, "kpc") == "12.35 kpc"


def test_format_converted_label_without_unit():
    assert _format_converted_label(12.345, None) == "12.35"
    assert _format_converted_label(12.345, "") == "12.35"


# ---- Ruler convert= integration --------------------------------------------

def test_convert_callable_renders_converted_labels():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0,
              convert=lambda asec: asec * 0.5, convert_unit="kpc")
    r.add_to(ax)
    # Ticks at 0, 20, 40, 60, 80 arcsec → 0, 10, 20, 30, 40 kpc
    texts = [lab.get_text() for lab in r.label_artists]
    assert texts == ["0 kpc", "10 kpc", "20 kpc", "30 kpc", "40 kpc"]
    plt.close(fig)


def test_convert_redshift_dict_renders_kpc_labels():
    """At z=0.5 (Planck18), every 1 arcsec ≈ 6.3 kpc projected. The
    test asserts the labels carry the kpc suffix and that the numeric
    values increase monotonically along the ruler."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=0.5,
              tick_interval=10.0,
              convert=dict(redshift=0.5, unit="kpc"))
    r.add_to(ax)
    texts = [lab.get_text() for lab in r.label_artists]
    assert all("kpc" in t for t in texts)
    # The numeric prefix should grow tick-to-tick.
    nums = [float(t.split()[0]) for t in texts]
    assert nums == sorted(nums)
    assert nums[0] == 0.0
    assert nums[-1] > 200.0    # 40" * ~6.3 kpc/asec = ~250 kpc
    plt.close(fig)


def test_convert_distance_dict_renders_au_labels():
    """1 arcsec at 100 pc = 100 AU (definitionally, since 1 pc = 206265 AU)."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=0.05,
              tick_interval=1.0,
              convert=dict(distance=100, distance_unit="pc",
                           unit="au"))
    r.add_to(ax)
    texts = [lab.get_text() for lab in r.label_artists]
    assert texts == ["0 au", "100 au", "200 au", "300 au", "400 au"]
    plt.close(fig)


def test_convert_without_pixscale_raises_at_add_to():
    """convert= operates on arcsec values, so a known pixel scale is
    required — without it, the conversion would silently treat pixels
    as arcsec. We raise early at add_to() time instead."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50),       # no pixscale
              convert=lambda asec: asec)
    with pytest.raises(ValueError, match="convert= requires"):
        r.add_to(ax)
    plt.close(fig)


def test_convert_takes_precedence_over_label_unit():
    """When ``convert=`` is set, ``label_unit=`` is ignored for the
    tick labels (the converted unit wins)."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0,
              label_unit="arcmin",        # ignored
              convert=lambda asec: asec, convert_unit="kpc")
    r.add_to(ax)
    for lab in r.label_artists:
        assert "kpc" in lab.get_text()
        assert "′" not in lab.get_text()
    plt.close(fig)


def test_label_fmt_takes_precedence_over_convert():
    """A custom label_fmt should win over the configured convert=."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0,
              label_fmt=lambda v, u: f"X={v:.0f}",
              convert=lambda asec: asec * 0.5, convert_unit="kpc")
    r.add_to(ax)
    for lab in r.label_artists:
        text = lab.get_text()
        assert text.startswith("X=")
        assert "kpc" not in text
    plt.close(fig)


def test_set_labels_can_swap_convert_after_construction():
    """The convert= kwarg can be replaced via set_labels() — the new
    value takes effect on the next add_to()."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0)
    r.add_to(ax)
    # Initially no convert — an angular label (arcsec/arcmin), not kpc.
    assert r.label_artists[0].get_text().endswith(("″", "′", "°"))
    r.remove()
    r.set_labels(convert=lambda asec: asec * 0.5, convert_unit="kpc")
    r.add_to(ax)
    assert "kpc" in r.label_artists[0].get_text()
    plt.close(fig)


# ---- v2 helpers: format / rotation / side ----------------------------------

def test_format_numeric_default_is_four_sig_fig():
    assert _format_numeric(1.33333) == "1.333"
    assert _format_numeric(5.0) == "5"


def test_format_numeric_honors_printf_fmt():
    assert _format_numeric(1.33333, fmt="%.2f") == "1.33"
    assert _format_numeric(42.0, fmt="%3d") == " 42"


def test_normalize_readable_angle_folds_upside_down():
    """Angles in (90, 270] flip by 180° so text is right-side-up."""
    assert _normalize_readable_angle(0) == 0.0
    assert _normalize_readable_angle(45) == 45.0
    assert _normalize_readable_angle(90) == 90.0
    assert _normalize_readable_angle(135) == -45.0   # flipped from 135
    assert _normalize_readable_angle(180) == 0.0     # flipped from 180
    assert _normalize_readable_angle(225) == 45.0    # flipped from 225
    assert _normalize_readable_angle(-45) == -45.0


def test_resolve_rotation_deg_modes():
    """The four rotation modes (auto, horizontal, perpendicular,
    numeric literal) map to the expected angles given a 30° tangent."""
    assert _resolve_rotation_deg("horizontal", 30.0) == 0.0
    assert _resolve_rotation_deg("auto", 30.0) == 30.0
    assert _resolve_rotation_deg("perpendicular", 30.0) == pytest.approx(-60.0)
    assert _resolve_rotation_deg(45.0, 30.0) == 45.0  # literal


def test_resolve_rotation_deg_invalid_mode_raises():
    with pytest.raises(ValueError, match="rotation mode"):
        _resolve_rotation_deg("oblique", 30.0)


def test_resolve_label_side_sign_auto_matches_tick_side():
    """label_side='auto' tracks the tick side (right ticks → right
    labels; everything else → +1 side)."""
    assert _resolve_label_side_sign("auto", "both") == +1
    assert _resolve_label_side_sign("auto", "left") == +1
    assert _resolve_label_side_sign("auto", "right") == -1
    assert _resolve_label_side_sign("auto", "none") == +1


def test_resolve_label_side_sign_explicit_overrides_tick_side():
    assert _resolve_label_side_sign("left", "right") == +1
    assert _resolve_label_side_sign("right", "left") == -1


def test_resolve_label_side_sign_invalid_raises():
    with pytest.raises(ValueError, match="label_side"):
        _resolve_label_side_sign("middle", "both")


def test_resolve_title_side_sign_auto_is_opposite_label():
    assert _resolve_title_side_sign("auto", +1) == -1
    assert _resolve_title_side_sign("auto", -1) == +1


def test_resolve_title_side_sign_explicit_overrides():
    assert _resolve_title_side_sign("left", +1) == +1
    assert _resolve_title_side_sign("right", +1) == -1


# ---- _resolve_text_anchor (padding helper) ---------------------------------

def test_resolve_text_anchor_zero_rotation_above_line():
    """At rotation=0, the outward direction in local coords equals
    the outward direction in display coords. Label above (perp=+y) →
    va='bottom' so text grows upward from the anchor."""
    assert _resolve_text_anchor(0.0, +1, 0.0, 1.0) == ("center", "bottom")
    assert _resolve_text_anchor(0.0, -1, 0.0, 1.0) == ("center", "top")


def test_resolve_text_anchor_zero_rotation_horizontal_perp():
    """At rotation=0, vertical ruler perp=+x → outward stays +x in
    local → ha='left' so text grows rightward (away from the line)."""
    assert _resolve_text_anchor(0.0, +1, 1.0, 0.0) == ("left", "center")
    assert _resolve_text_anchor(0.0, -1, 1.0, 0.0) == ("right", "center")


def test_resolve_text_anchor_perpendicular_label_above_horizontal_ruler():
    """At rotation=90°, the text rotates 90° CCW around the anchor.
    Outward = +y in display maps to +x in local (since R(-90) ·
    (0,1) = (1,0)) → ha='left'. After rotation, local +x points to
    display +y, so text grows upward from the anchor as intended."""
    assert _resolve_text_anchor(90.0, +1, 0.0, 1.0) == ("left", "center")
    assert _resolve_text_anchor(90.0, -1, 0.0, 1.0) == ("right", "center")


def test_resolve_text_anchor_auto_equivalent_at_tangent_angle():
    """For 'auto' rotation, the text angle matches the tangent. The
    perpendicular outward, expressed in the text's local frame,
    always points to local +up — so the anchor is va='bottom' (+1
    side) regardless of which way the ruler is pointing."""
    # Horizontal ruler: tangent angle = 0, outward perp = (0, ±1).
    assert _resolve_text_anchor(0.0, +1, 0.0, 1.0) == ("center", "bottom")
    # 45° diagonal: tangent angle = 45°, outward perp = (-√2/2, √2/2).
    perp_45 = (-np.sqrt(0.5), np.sqrt(0.5))
    assert _resolve_text_anchor(45.0, +1, *perp_45) == ("center", "bottom")
    # Vertical ruler: tangent angle = 90°, outward perp = (-1, 0).
    assert _resolve_text_anchor(90.0, +1, -1.0, 0.0) == ("center", "bottom")


def test_resolve_text_anchor_negative_90_rotation():
    """At rotation=-90°, local +x maps to display -y. Outward = +y
    (label above line) → we need to anchor at local -x (ha='right')
    so text grows in local -x = display +y direction."""
    assert _resolve_text_anchor(-90.0, +1, 0.0, 1.0) == ("right", "center")


def test_label_anchor_perpendicular_rotation_uses_ha_left():
    """Concrete render check: a horizontal ruler with
    label_rotation='perpendicular' and labels above the line should
    use ha='left' (not va='bottom'), so the text grows upward from
    the anchor instead of leftward."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
               tick_interval=20.0, label_rotation="perpendicular")
    r.add_to(ax)
    for lab in r.label_artists:
        assert lab.get_horizontalalignment() == "left"
        assert lab.get_verticalalignment() == "center"
    plt.close(fig)


def test_label_anchor_auto_rotation_consistent_for_horizontal_ruler():
    """For 'auto' rotation on a horizontal ruler, labels above the
    line stay at va='bottom' (the existing well-behaved case)."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
               tick_interval=20.0, label_rotation="auto")
    r.add_to(ax)
    for lab in r.label_artists:
        assert lab.get_horizontalalignment() == "center"
        assert lab.get_verticalalignment() == "bottom"
    plt.close(fig)


def test_default_tick_length_is_compact():
    """The v2 default tick_length is ``4`` pt — a compact look that
    doesn't crowd the labels. (If this default changes, callers who
    relied on the old 6 pt will see a subtle visual shift; update the
    test along with the default.)"""
    r = Ruler((10, 50), (90, 50))
    assert r._tick_length == 4.0


# ---- title ------------------------------------------------------------------

def test_title_kwarg_renders_text_artist():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0, title="Distance bar")
    r.add_to(ax)
    assert r.title_artist is not None
    assert r.title_artist.get_text() == "Distance bar"
    assert r.title_artist in ax.texts
    plt.close(fig)


def test_title_none_produces_no_artist():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0, tick_interval=20.0)
    r.add_to(ax)
    assert r.title_artist is None
    plt.close(fig)


def test_title_property_returns_current_text():
    r = Ruler((0, 0), (10, 0), title="foo")
    assert r.title == "foo"


def test_title_color_defaults_to_main_line_color():
    """Title inherits ``color`` when ``title_color`` is unset."""
    import matplotlib.colors as mcolors
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0, color="C2", title="x")
    r.add_to(ax)
    expected = mcolors.to_rgba("C2")
    got = mcolors.to_rgba(r.title_artist.get_color())
    assert got == pytest.approx(expected)
    plt.close(fig)


def test_title_side_defaults_to_opposite_label_side():
    """With label_side='auto' on a both-sided ruler, labels go to +1
    and title to -1. Confirm by checking the title y-position lies
    on the opposite side of the line from the first label."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0, title="below")
    r.add_to(ax)
    lab_y = r.label_artists[0].get_position()[1]
    title_y = r.title_artist.get_position()[1]
    # Line sits at y=50; label and title bracket it.
    assert (lab_y - 50) * (title_y - 50) < 0
    plt.close(fig)


def test_set_title_updates_text_in_place():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0, title="original")
    r.add_to(ax)
    r.set_title("updated")
    assert r.title_artist.get_text() == "updated"
    assert r.title == "updated"
    plt.close(fig)


def test_set_title_returns_self_for_chaining():
    r = Ruler((10, 50), (90, 50), title="x")
    assert r.set_title("y", color="C3") is r


def test_set_title_validates_side_kwarg():
    r = Ruler((10, 50), (90, 50), title="x")
    with pytest.raises(ValueError, match="title_side"):
        r.set_title(side="diagonal")


def test_remove_strips_title_too():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0, title="x")
    r.add_to(ax)
    title = r.title_artist
    r.remove()
    assert title not in ax.texts
    assert r.title_artist is None
    plt.close(fig)


def test_invalid_title_side_at_construction_raises():
    with pytest.raises(ValueError, match="title_side"):
        Ruler((0, 0), (10, 0), title="x", title_side="diagonal")


# ---- fmt (printf-style format) ---------------------------------------------

def test_fmt_string_applies_to_label_numeric():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0, fmt="%.1f")
    r.add_to(ax)
    # 0, 20, 40, 60, 80 arcsec. The auto unit is resolved ONCE from the
    # largest tick (80″ → arcmin), so all ticks render uniformly in arcmin.
    texts = [lab.get_text() for lab in r.label_artists]
    assert texts == ["0.0′", "0.3′", "0.7′", "1.0′", "1.3′"]
    plt.close(fig)


def test_fmt_string_works_with_convert():
    """fmt= applies to the converted numeric value."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0, fmt="%.0f",
              convert=lambda asec: asec * 0.5, convert_unit="kpc")
    r.add_to(ax)
    texts = [lab.get_text() for lab in r.label_artists]
    assert texts == ["0 kpc", "10 kpc", "20 kpc", "30 kpc", "40 kpc"]
    plt.close(fig)


def test_label_fmt_callable_still_wins_over_fmt():
    """label_fmt callable bypasses both fmt= and convert=."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0,
              fmt="%.5f",     # ignored
              label_fmt=lambda v, u: "X")
    r.add_to(ax)
    for lab in r.label_artists:
        assert lab.get_text() == "X"
    plt.close(fig)


# ---- label_side ------------------------------------------------------------

def test_label_side_left_puts_labels_above_horizontal_ruler():
    """For a horizontal ruler xy1→xy2 pointing right, the +1
    perpendicular direction is 'up' (the 'left' side as the user
    travels along the ruler). label_side='left' places labels above."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0, label_side="left")
    r.add_to(ax)
    for lab in r.label_artists:
        assert lab.get_position()[1] > 50
    plt.close(fig)


def test_label_side_right_puts_labels_below_horizontal_ruler():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0, label_side="right")
    r.add_to(ax)
    for lab in r.label_artists:
        assert lab.get_position()[1] < 50
    plt.close(fig)


def test_invalid_label_side_raises():
    with pytest.raises(ValueError, match="label_side"):
        Ruler((0, 0), (10, 0), label_side="middle")


# ---- label rotation --------------------------------------------------------

def test_label_rotation_horizontal_is_zero():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0, label_rotation="horizontal")
    r.add_to(ax)
    for lab in r.label_artists:
        assert lab.get_rotation() == pytest.approx(0.0)
    plt.close(fig)


def test_label_rotation_auto_follows_tangent():
    """For a 45° diagonal in data coords on an equal-aspect axes, the
    display-coord tangent angle should also be close to 45° (modulo
    figure aspect — the test uses an equal-aspect axes)."""
    fig, ax = _plain_axes()
    r = Ruler((10, 10), (90, 90), pixscale_asec=1.0,
              tick_interval=20.0, label_rotation="auto")
    r.add_to(ax)
    # All labels should have the same rotation (constant tangent
    # for a straight line).
    rotations = [lab.get_rotation() for lab in r.label_artists]
    assert len(set(np.round(rotations, 5))) == 1
    # And the rotation should be non-zero for a diagonal ruler.
    assert abs(rotations[0]) > 10.0
    plt.close(fig)


def test_label_rotation_perpendicular_is_orthogonal_to_tangent():
    fig, ax = _plain_axes()
    # Horizontal ruler → tangent angle = 0 → perpendicular = 90 (or
    # readability-folded equivalent).
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0, label_rotation="perpendicular")
    r.add_to(ax)
    for lab in r.label_artists:
        assert abs(lab.get_rotation()) == pytest.approx(90.0)
    plt.close(fig)


def test_label_rotation_numeric_literal_is_used_directly():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0, label_rotation=45.0)
    r.add_to(ax)
    for lab in r.label_artists:
        assert lab.get_rotation() == pytest.approx(45.0)
    plt.close(fig)


def test_label_rotation_add_offsets_base_rotation():
    """rotation_add=90 turns 'horizontal' into vertical, and 'auto'
    into perpendicular-to-tangent."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0,
              label_rotation="horizontal", label_rotation_add=90.0)
    r.add_to(ax)
    for lab in r.label_artists:
        # 90° rotation (or its readability fold to 90°)
        assert abs(lab.get_rotation()) == pytest.approx(90.0)
    plt.close(fig)


def test_label_rotation_invalid_mode_raises_at_add_to():
    """Invalid rotation modes raise at render time, not construction
    time (since we don't validate until we resolve)."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0, label_rotation="oblique")
    with pytest.raises(ValueError, match="rotation mode"):
        r.add_to(ax)
    plt.close(fig)


# ---- tick_positions --------------------------------------------------------

def test_tick_positions_overrides_interval_and_n_ticks():
    fig, ax = _plain_axes()
    r = Ruler((0, 50), (100, 50), pixscale_asec=1.0,
              tick_positions=[0, 25, 50, 75, 100])
    r.add_to(ax)
    # Largest tick 100″ → arcmin; uniform across all ticks.
    texts = [lab.get_text() for lab in r.label_artists]
    assert texts == ["0′", "0.4167′", "0.8333′", "1.25′", "1.667′"]
    plt.close(fig)


def test_tick_positions_skips_out_of_range():
    fig, ax = _plain_axes()
    # 100 px ruler, positions including a negative and an over-range:
    r = Ruler((0, 50), (100, 50), pixscale_asec=1.0,
              tick_positions=[-10, 20, 50, 80, 200])
    r.add_to(ax)
    # In-range 20, 50, 80″; largest (80″) → arcmin, uniform.
    texts = [lab.get_text() for lab in r.label_artists]
    assert texts == ["0.3333′", "0.8333′", "1.333′"]
    plt.close(fig)


def test_tick_positions_explicit_overrides_explicit_interval():
    """tick_positions= wins when both are passed."""
    fig, ax = _plain_axes()
    r = Ruler((0, 50), (100, 50), pixscale_asec=1.0,
              tick_interval=20.0,
              tick_positions=[0, 50, 100])
    r.add_to(ax)
    assert len(r.label_artists) == 3
    plt.close(fig)


# ---- endpoint collision fix ------------------------------------------------

def test_endpoint_well_separated_is_kept():
    """A 91 px ruler at interval 20 (5px ticks + endpoint gap=11 > 10
    threshold) keeps the endpoint."""
    fig, ax = _plain_axes()
    r = Ruler((0, 50), (91, 50), pixscale_asec=1.0)
    r.add_to(ax)
    n_before_endpoint = 5    # 0, 20, 40, 60, 80
    assert len(r.label_artists) == n_before_endpoint + 1
    plt.close(fig)


def test_endpoint_close_to_last_tick_is_dropped():
    """An 85 px ruler at interval 20 (endpoint gap=5 < 10 threshold)
    skips the endpoint label to avoid visual clutter."""
    fig, ax = _plain_axes()
    r = Ruler((0, 50), (85, 50), pixscale_asec=1.0)
    r.add_to(ax)
    assert len(r.label_artists) == 5    # 0, 20, 40, 60, 80
    plt.close(fig)


def test_endpoint_exact_match_no_duplicate():
    """An 80 px ruler at interval 20 has the endpoint coincide with
    the last regular tick — single label, not duplicated."""
    fig, ax = _plain_axes()
    r = Ruler((0, 50), (80, 50), pixscale_asec=1.0,
              tick_interval=20.0)
    r.add_to(ax)
    assert len(r.label_artists) == 5
    plt.close(fig)


# ---- from_polar factory ----------------------------------------------------

def test_from_polar_plot_convention_angle_zero():
    """angle=0 in 'plot' convention → second endpoint along +x."""
    fig, ax = _plain_axes()
    r = Ruler.from_polar((50, 50), length=30.0, angle=0.0,
                          pixscale_asec=1.0,
                          angle_convention="plot")
    assert r.xy2[0] == pytest.approx(80.0)
    assert r.xy2[1] == pytest.approx(50.0)
    plt.close(fig)


def test_from_polar_plot_convention_angle_ninety():
    """angle=90 in 'plot' convention → second endpoint along +y."""
    r = Ruler.from_polar((50, 50), length=30.0, angle=90.0,
                          pixscale_asec=1.0,
                          angle_convention="plot")
    assert r.xy2[0] == pytest.approx(50.0)
    assert r.xy2[1] == pytest.approx(80.0)


def test_from_polar_fits_convention_pa_zero_no_wcs():
    """Without a WCS, the FITS convention defaults to the E-left
    convention: PA=0 → north (+y direction)."""
    r = Ruler.from_polar((50, 50), length=30.0, angle=0.0,
                          pixscale_asec=1.0,
                          angle_convention="fits")
    # PA=0 (north of N-up image) → +y, second endpoint at (50, 80)
    assert r.xy2[0] == pytest.approx(50.0)
    assert r.xy2[1] == pytest.approx(80.0)


def test_from_polar_fits_convention_pa_ninety_no_wcs():
    """PA=90 (east on N-up E-left image) → -x direction."""
    r = Ruler.from_polar((50, 50), length=30.0, angle=90.0,
                          pixscale_asec=1.0,
                          angle_convention="fits")
    assert r.xy2[0] == pytest.approx(20.0)
    assert r.xy2[1] == pytest.approx(50.0)


def test_from_polar_arcsec_unit_converts_via_pixscale():
    """length=10 arcsec at 0.5 asec/pix → 20 px segment."""
    r = Ruler.from_polar((50, 50), length=10.0, angle=0.0,
                          pixscale_asec=0.5,
                          length_unit="arcsec",
                          angle_convention="plot")
    assert r.xy2[0] - r.xy1[0] == pytest.approx(20.0)


def test_from_polar_arcmin_unit_converts():
    """length=2 arcmin = 120 arcsec → 60 px at 2 asec/pix."""
    r = Ruler.from_polar((50, 50), length=2.0, angle=0.0,
                          pixscale_asec=2.0,
                          length_unit="arcmin",
                          angle_convention="plot")
    assert r.xy2[0] - r.xy1[0] == pytest.approx(60.0)


def test_from_polar_pix_unit_no_scale_needed():
    """length_unit='pix' works without a pixel scale."""
    r = Ruler.from_polar((50, 50), length=30.0, angle=0.0,
                          length_unit="pix",
                          angle_convention="plot")
    assert r.xy2[0] - r.xy1[0] == pytest.approx(30.0)


def test_from_polar_non_pix_unit_without_scale_raises():
    with pytest.raises(ValueError, match="requires a known pixel scale"):
        Ruler.from_polar((50, 50), length=30.0, angle=0.0,
                          length_unit="arcsec")


def test_from_polar_invalid_length_unit_raises():
    with pytest.raises(ValueError, match="length_unit"):
        Ruler.from_polar((50, 50), length=30.0, angle=0.0,
                          pixscale_asec=1.0, length_unit="parsec")


def test_from_polar_invalid_angle_convention_raises():
    with pytest.raises(ValueError, match="angle_convention"):
        Ruler.from_polar((50, 50), length=30.0, angle=0.0,
                          pixscale_asec=1.0, angle_convention="rad")


def test_from_polar_forwards_styling_kwargs():
    r = Ruler.from_polar((50, 50), length=30.0, angle=0.0,
                          pixscale_asec=1.0,
                          angle_convention="plot",
                          color="C3", lw=2.0, title="bar")
    assert r._color == "C3"
    assert r._lw == 2.0
    assert r.title == "bar"


# ---- set_labels: new v2 setter kwargs --------------------------------------

def test_set_labels_can_swap_label_fmt_callable():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0, tick_interval=20.0)
    r.add_to(ax)
    r.remove()
    r.set_labels(label_fmt=lambda v, u: "X")
    r.add_to(ax)
    assert all(lab.get_text() == "X" for lab in r.label_artists)
    plt.close(fig)


def test_set_labels_can_swap_printf_fmt():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0, tick_interval=20.0)
    r.add_to(ax)
    r.remove()
    r.set_labels(fmt="%.2f")
    r.add_to(ax)
    # Largest tick (80″) → arcmin, resolved once → all ticks in arcmin.
    assert r.label_artists[0].get_text() == "0.00′"
    plt.close(fig)


def test_set_labels_can_swap_side_and_rotation():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0, tick_interval=20.0)
    r.add_to(ax)
    r.remove()
    r.set_labels(side="right", rotation="perpendicular",
                  rotation_add=15.0)
    r.add_to(ax)
    # Labels now below (side='right' on horizontal ruler)
    for lab in r.label_artists:
        assert lab.get_position()[1] < 50
    plt.close(fig)


def test_set_labels_validates_side():
    r = Ruler((0, 0), (10, 0))
    with pytest.raises(ValueError, match="label_side"):
        r.set_labels(side="oblique")


# ---- endcaps ---------------------------------------------------------------

def test_endcap_style_default_is_none():
    """Default behavior is unchanged: no endcaps, endpoint label
    follows the collision rule."""
    r = Ruler((10, 50), (90, 50))
    assert r._endcap_style == "none"
    assert r._endcaps == "both"
    assert r._endcap_label == "auto"


def test_invalid_endcap_style_raises():
    with pytest.raises(ValueError, match="endcap_style must be"):
        Ruler((0, 0), (10, 0), endcap_style="serif")


def test_invalid_endcaps_kwarg_raises():
    with pytest.raises(ValueError, match="endcaps must be"):
        Ruler((0, 0), (10, 0), endcaps="middle")


def test_invalid_endcap_label_raises():
    with pytest.raises(ValueError, match="endcap_label must be"):
        Ruler((0, 0), (10, 0), endcap_label="sometimes")


def test_endcap_artists_property_starts_empty():
    r = Ruler((10, 50), (90, 50), endcap_style="arrow")
    assert r.endcap_artists == []


def test_endcap_tick_renders_two_line2d_per_end():
    """``endcap_style='tick'`` with ``endcaps='both'`` and ``tick_side=
    'both'`` produces 4 endcap artists (2 ends × 2 sides) as Line2Ds.
    The regular endpoint ticks are suppressed (the endcap replaces
    them), so total tick count drops by 2 sides × 2 ends = 4."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0, endcap_style="tick")
    r.add_to(ax)
    assert len(r.endcap_artists) == 4
    for cap in r.endcap_artists:
        assert isinstance(cap, Line2D)
        assert cap in ax.lines
    # 5 positions × 2 sides = 10 regular ticks if no endcaps; with
    # endcaps replacing the start + end ticks → 3 inner positions × 2
    # sides = 6 regular ticks.
    assert len(r.tick_artists) == 6
    plt.close(fig)


def test_endcap_tick_length_scale_makes_caps_longer_than_regular():
    """Endpoint ticks (endcap_style='tick') should be longer than
    the regular ticks by ``endcap_length_scale``."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0,
              endcap_style="tick", endcap_length_scale=2.0)
    r.add_to(ax)
    cap = r.endcap_artists[0]
    reg = r.tick_artists[0]
    cap_xs, cap_ys = cap.get_data()
    reg_xs, reg_ys = reg.get_data()
    cap_len = abs(cap_ys[1] - cap_ys[0])
    reg_len = abs(reg_ys[1] - reg_ys[0])
    assert cap_len == pytest.approx(2.0 * reg_len, rel=1e-3)
    plt.close(fig)


def test_endcap_tick_one_side_only_produces_two_caps():
    """With ``endcaps='both'`` and ``tick_side='left'`` (one-sided
    ticks), endpoints get one cap each → 2 endcap artists."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0,
              endcap_style="tick", tick_side="left")
    r.add_to(ax)
    assert len(r.endcap_artists) == 2
    plt.close(fig)


def test_endcap_arrow_renders_one_fancy_arrow_per_end():
    """``endcap_style='arrow'`` always produces one
    :class:`FancyArrowPatch` per end, regardless of tick_side."""
    from matplotlib.patches import FancyArrowPatch

    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0, endcap_style="arrow")
    r.add_to(ax)
    assert len(r.endcap_artists) == 2
    for cap in r.endcap_artists:
        assert isinstance(cap, FancyArrowPatch)
        assert cap in ax.patches
    plt.close(fig)


def test_endcap_arrow_replaces_endpoint_ticks():
    """``endcap_style='arrow'`` with ``tick_side='both'`` replaces
    the two endpoint ticks (per side) at each end with one
    arrowhead — so regular tick count drops by 4 (2 sides × 2 ends)."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0, endcap_style="arrow")
    r.add_to(ax)
    # Without endcaps: 5 positions × 2 sides = 10 ticks.
    # With arrow endcaps replacing both endpoints: 3 inner × 2 = 6.
    assert len(r.tick_artists) == 6
    plt.close(fig)


def test_endcaps_start_only_produces_one_arrow():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0,
              endcap_style="arrow", endcaps="start")
    r.add_to(ax)
    assert len(r.endcap_artists) == 1
    plt.close(fig)


def test_endcaps_end_only_produces_one_arrow():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0,
              endcap_style="arrow", endcaps="end")
    r.add_to(ax)
    assert len(r.endcap_artists) == 1
    plt.close(fig)


def test_endcaps_none_suppresses_caps_even_when_style_set():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0,
              endcap_style="arrow", endcaps="none")
    r.add_to(ax)
    assert r.endcap_artists == []
    plt.close(fig)


def test_endcap_style_none_means_no_endcap_artists():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0,
              endcap_style="none", endcaps="both")
    r.add_to(ax)
    assert r.endcap_artists == []
    plt.close(fig)


def test_endcap_label_true_forces_endpoint_label_under_collision():
    """An 85 px ruler at interval 20 has endpoint gap=5 (< 10
    threshold) — normally the endpoint is dropped. With
    ``endcap_label=True``, it must be included regardless.
    (85 arcsec auto-promotes to arcmin → '1.417′'.)"""
    fig, ax = _plain_axes()
    r = Ruler((0, 50), (85, 50), pixscale_asec=1.0,
              endcap_label=True)
    r.add_to(ax)
    texts = [lab.get_text() for lab in r.label_artists]
    assert "1.417′" in texts
    plt.close(fig)


def test_endcap_label_false_suppresses_endpoint_label_even_with_cap():
    """With ``endcap_style='arrow'`` the endpoint would normally
    get a label (the cap disambiguates). ``endcap_label=False``
    suppresses it explicitly."""
    fig, ax = _plain_axes()
    r = Ruler((0, 50), (91, 50), pixscale_asec=1.0,
              endcap_style="arrow", endcaps="end",
              endcap_label=False)
    r.add_to(ax)
    texts = [lab.get_text() for lab in r.label_artists]
    # 0, 20, 40, 60, 80 — but NOT the 91 endpoint
    assert "91″" not in texts
    assert "1.517′" not in texts
    plt.close(fig)


def test_endcap_label_auto_includes_endpoint_when_capped():
    """An 85 px ruler (collision-suppressed by default) gets its
    endpoint labeled when an endcap is drawn there, because the
    cap visually disambiguates the endpoint from the previous tick.
    (85 arcsec auto-promotes to arcmin → '1.417′'.)"""
    fig, ax = _plain_axes()
    r = Ruler((0, 50), (85, 50), pixscale_asec=1.0,
              endcap_style="arrow", endcaps="end")
    r.add_to(ax)
    texts = [lab.get_text() for lab in r.label_artists]
    assert "1.417′" in texts
    plt.close(fig)


def test_endcap_label_auto_no_cap_uses_collision_rule():
    """Without endcaps, ``endcap_label='auto'`` falls back to the
    collision rule — endpoint dropped when too close."""
    fig, ax = _plain_axes()
    r = Ruler((0, 50), (85, 50), pixscale_asec=1.0)
    r.add_to(ax)
    texts = [lab.get_text() for lab in r.label_artists]
    assert "1.417′" not in texts
    plt.close(fig)


def test_endcap_color_defaults_to_tick_color():
    """An unset endcap_color resolves to tick_color (which in turn
    resolves to the main line color)."""
    import matplotlib.colors as mcolors

    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0,
              endcap_style="tick", color="C2", tick_color="C5")
    r.add_to(ax)
    cap = r.endcap_artists[0]
    assert mcolors.to_rgba(cap.get_color()) == pytest.approx(
        mcolors.to_rgba("C5"))
    plt.close(fig)


def test_endcap_color_explicit_wins():
    import matplotlib.colors as mcolors

    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0,
              endcap_style="tick", tick_color="C5",
              endcap_color="C3")
    r.add_to(ax)
    cap = r.endcap_artists[0]
    assert mcolors.to_rgba(cap.get_color()) == pytest.approx(
        mcolors.to_rgba("C3"))
    plt.close(fig)


def test_remove_strips_endcap_artists():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0, endcap_style="arrow")
    r.add_to(ax)
    caps = r.endcap_artists
    assert caps
    r.remove()
    for c in caps:
        assert c not in ax.patches
        assert c not in ax.lines
    assert r.endcap_artists == []
    plt.close(fig)


def test_from_polar_defaults_to_arrow_endcap_at_end():
    """``Ruler.from_polar`` defaults to ``endcap_style='arrow'`` and
    ``endcaps='end'`` — the "60″ at PA=45°" use case naturally
    reads as an outward arrow at the destination, not at the
    anchor."""
    r = Ruler.from_polar((50, 50), length=30.0, angle=0.0,
                          pixscale_asec=1.0,
                          angle_convention="plot")
    assert r._endcap_style == "arrow"
    assert r._endcaps == "end"


def test_from_polar_user_override_wins():
    """Explicit kwargs in from_polar override the arrow default."""
    r = Ruler.from_polar((50, 50), length=30.0, angle=0.0,
                          pixscale_asec=1.0,
                          angle_convention="plot",
                          endcap_style="none")
    assert r._endcap_style == "none"

    r = Ruler.from_polar((50, 50), length=30.0, angle=0.0,
                          pixscale_asec=1.0,
                          angle_convention="plot",
                          endcap_style="tick", endcaps="both")
    assert r._endcap_style == "tick"
    assert r._endcaps == "both"


# ---- lambda0 kwarg --------------------------------------------------------

def test_lambda0_default_is_zero():
    r = Ruler((10, 50), (90, 50))
    assert r._lambda0 == 0.0


def test_lambda0_zero_matches_v1_label_set():
    """``lambda0=0`` (default) keeps the v1 behavior: tick labels run
    from 0 at xy1 to total at xy2, all non-negative."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0, tick_interval=20.0)
    r.add_to(ax)
    texts = [lab.get_text() for lab in r.label_artists]
    # 0, 20, 40, 60, 80 arcsec — the auto unit resolves once from the largest
    # tick (80″ → arcmin), so all ticks render uniformly in arcmin.
    assert texts == ["0′", "0.3333′", "0.6667′", "1′", "1.333′"]
    plt.close(fig)


def test_lambda0_half_produces_symmetric_signed_labels():
    """``lambda0=0.5`` puts the value-0 tick at the midpoint and
    produces ±-valued labels around it."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0, lambda0=0.5)
    r.add_to(ax)
    texts = [lab.get_text() for lab in r.label_artists]
    assert texts == ["-40″", "-20″", "0″", "20″", "40″"]
    plt.close(fig)


def test_lambda0_asymmetric_value_with_collision_endpoint():
    """``lambda0=0.3`` on an 80-px / interval=20 ruler: regular ticks
    at {-20, 0, 20, 40} (k from -1 to 2); endpoints at d=-24 (gap=4
    < half-interval → drop) and d=+56 (gap=16 > 10 → include)."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0, lambda0=0.3)
    r.add_to(ax)
    texts = [lab.get_text() for lab in r.label_artists]
    assert texts == ["-20″", "0″", "20″", "40″", "56″"]
    plt.close(fig)


def test_lambda0_tick_positions_use_signed_coordinates():
    """When ``lambda0 > 0``, the user-supplied ``tick_positions=`` is
    interpreted in *signed* d_label units (the same units the
    labels show), not absolute distance from xy1."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              lambda0=0.5,
              tick_positions=[-40, -20, 0, 20, 40])
    r.add_to(ax)
    texts = [lab.get_text() for lab in r.label_artists]
    assert texts == ["-40″", "-20″", "0″", "20″", "40″"]
    plt.close(fig)


def test_lambda0_out_of_range_raises():
    with pytest.raises(ValueError, match="lambda0 must be in"):
        Ruler((0, 0), (10, 0), lambda0=1.5)
    with pytest.raises(ValueError, match="lambda0 must be in"):
        Ruler((0, 0), (10, 0), lambda0=-0.1)


def test_lambda0_endpoint_collision_rule_at_start():
    """The collision rule applies at the START endpoint too when
    ``lambda0 > 0``. Construct a ruler whose start endpoint is
    well-separated from the nearest regular tick — it should be
    included."""
    fig, ax = _plain_axes()
    # 100-pix ruler, lambda0=0.4 → zero at 40, d_min=-40, d_max=+60.
    # interval=30 → regular ticks at -30, 0, 30, 60. Start endpoint
    # at -40: gap to -30 is 10 < 15 (half interval) → DROP.
    r = Ruler((0, 50), (100, 50), pixscale_asec=1.0,
              tick_interval=30.0, lambda0=0.4)
    r.add_to(ax)
    texts = [lab.get_text() for lab in r.label_artists]
    # Largest tick 60″ → arcmin, resolved once → uniform arcmin.
    assert "-0.6667′" not in texts      # start (−40″) collision-dropped
    assert texts == ["-0.5′", "0′", "0.5′", "1′"]
    plt.close(fig)


def test_lambda0_endcap_label_true_forces_both_endpoints():
    """``endcap_label=True`` includes both start and end labels
    regardless of the collision rule."""
    fig, ax = _plain_axes()
    r = Ruler((0, 50), (100, 50), pixscale_asec=1.0,
              tick_interval=30.0, lambda0=0.4,
              endcap_label=True)
    r.add_to(ax)
    texts = [lab.get_text() for lab in r.label_artists]
    # Largest tick 60″ → arcmin; the forced start endpoint (−40″ = −0.6667′).
    assert "-0.6667′" in texts
    plt.close(fig)


# ---- Ruler.from_zero factory ----------------------------------------------

def test_from_zero_symmetric_default():
    """``extent_back=None`` defaults to symmetric: the ruler extends
    ``extent`` in both directions, zero at *xy*."""
    r = Ruler.from_zero((50, 50), extent=20.0, angle=0.0,
                          pixscale_asec=1.0,
                          angle_convention="plot")
    assert r.xy1 == (30.0, 50.0)
    assert r.xy2 == (70.0, 50.0)
    assert r._lambda0 == 0.5


def test_from_zero_asymmetric_extents():
    """Different ``extent`` vs ``extent_back`` shifts the zero point
    proportionally."""
    r = Ruler.from_zero((50, 50), extent=30.0, extent_back=10.0,
                         angle=0.0, pixscale_asec=1.0,
                         angle_convention="plot")
    assert r.xy1 == (40.0, 50.0)        # 10 back
    assert r.xy2 == (80.0, 50.0)        # 30 forward
    # lambda0 = 10 / (10 + 30) = 0.25
    assert r._lambda0 == pytest.approx(0.25)


def test_from_zero_arcsec_unit_converts_via_pixscale():
    """``extent=10 arcsec`` at 0.5 asec/pix → 20-pix half-length."""
    r = Ruler.from_zero((50, 50), extent=10.0, angle=0.0,
                         pixscale_asec=0.5,
                         length_unit="arcsec",
                         angle_convention="plot")
    assert (r.xy2[0] - r.xy1[0]) == pytest.approx(40.0)  # 20 each side


def test_from_zero_pix_unit_no_scale_needed():
    r = Ruler.from_zero((50, 50), extent=15.0, angle=0.0,
                         length_unit="pix",
                         angle_convention="plot")
    assert (r.xy2[0] - r.xy1[0]) == pytest.approx(30.0)


def test_from_zero_default_endcaps_are_arrows_both():
    """``from_zero`` defaults to bidirectional arrows — the natural
    visual for a ruler extending from a zero coordinate."""
    r = Ruler.from_zero((50, 50), extent=20.0, angle=0.0,
                         pixscale_asec=1.0,
                         angle_convention="plot")
    assert r._endcap_style == "arrow"
    assert r._endcaps == "both"


def test_from_zero_endcap_override_wins():
    r = Ruler.from_zero((50, 50), extent=20.0, angle=0.0,
                         pixscale_asec=1.0,
                         angle_convention="plot",
                         endcap_style="none")
    assert r._endcap_style == "none"


def test_from_zero_zero_tick_lands_at_anchor():
    """With ``extent=extent_back``, the lambda0=0.5 tick (value 0)
    should sit at the anchor coordinate."""
    fig, ax = _plain_axes()
    r = Ruler.from_zero((50, 50), extent=20.0, angle=0.0,
                         pixscale_asec=1.0,
                         angle_convention="plot",
                         tick_interval=10.0)
    r.add_to(ax)
    # The "0″" label's anchor should sit at x=50 (the zero point).
    # find the label with text "0″" and verify its x-position
    zero_labs = [lab for lab in r.label_artists
                 if lab.get_text() == "0″"]
    assert len(zero_labs) == 1
    lab_x = zero_labs[0].get_position()[0]
    assert lab_x == pytest.approx(50.0, abs=0.5)
    plt.close(fig)


def test_from_zero_fits_convention_default():
    """``angle_convention='fits'`` is the default — same as
    ``from_polar``."""
    # angle=0 (PA=0, north on E-left WCS-less default) → +y direction
    r = Ruler.from_zero((50, 50), extent=20.0, angle=0.0,
                         pixscale_asec=1.0)   # no ax → no WCS, defaults to E-left
    # extent=extent_back=20, fits angle=0 → angle_plot=90 → +y direction
    assert r.xy1 == pytest.approx((50.0, 30.0))    # 20 back along -y
    assert r.xy2 == pytest.approx((50.0, 70.0))    # 20 forward along +y


def test_from_zero_negative_total_extent_raises():
    with pytest.raises(ValueError, match="extent \\+ extent_back must be"):
        Ruler.from_zero((0, 0), extent=-5.0, extent_back=0.0,
                          angle=0.0, pixscale_asec=1.0)


def test_from_zero_invalid_length_unit_raises():
    with pytest.raises(ValueError, match="length_unit"):
        Ruler.from_zero((0, 0), extent=10.0, angle=0.0,
                          pixscale_asec=1.0, length_unit="parsec")


def test_from_zero_invalid_angle_convention_raises():
    with pytest.raises(ValueError, match="angle_convention"):
        Ruler.from_zero((0, 0), extent=10.0, angle=0.0,
                          pixscale_asec=1.0, angle_convention="rad")


def test_from_zero_no_pixscale_for_arcsec_raises():
    with pytest.raises(ValueError, match="requires a known pixel scale"):
        Ruler.from_zero((0, 0), extent=10.0, angle=0.0,
                          length_unit="arcsec")


# ---- live reflow on resize / pan / zoom ------------------------------------

def test_add_to_connects_reflow_callbacks():
    """``add_to`` registers three matplotlib lifecycle callbacks
    (canvas resize + xlim/ylim changes) so the ruler stays
    visually consistent during pan / zoom / window-resize."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0, tick_interval=20.0)
    r.add_to(ax)
    assert r._reflow_cid_resize is not None
    assert r._reflow_cid_xlim is not None
    assert r._reflow_cid_ylim is not None
    assert r._reflow_fig is fig
    plt.close(fig)


def test_remove_disconnects_reflow_callbacks():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0, tick_interval=20.0)
    r.add_to(ax)
    r.remove()
    assert r._reflow_cid_resize is None
    assert r._reflow_cid_xlim is None
    assert r._reflow_cid_ylim is None
    assert r._reflow_fig is None
    assert r._host_axes is None
    plt.close(fig)


def test_on_layout_change_rebuilds_tick_length_in_data_coords():
    """Doubling the figure's vertical extent (with aspect free)
    doubles the display-coord ratio, so the data-coord tick length
    perpendicular to a horizontal ruler should HALVE to preserve
    the display-points visual length."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    fig.canvas.draw()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0)
    r.add_to(ax)
    fig.canvas.draw()
    xs0, ys0 = r.tick_artists[0].get_data()
    initial_len = abs(ys0[1] - ys0[0])

    # Resize: height doubled (display-pixels-per-data-unit doubled
    # in y), so the data-coord tick length should halve.
    fig.set_size_inches(6, 12)
    r._on_layout_change()        # manually fire (Agg doesn't auto-fire)
    xs1, ys1 = r.tick_artists[0].get_data()
    new_len = abs(ys1[1] - ys1[0])
    assert new_len == pytest.approx(initial_len / 2.0, rel=0.05)
    plt.close(fig)


def test_on_layout_change_is_idempotent_in_steady_state():
    """If the layout hasn't changed, calling _on_layout_change()
    should leave the artists at their existing positions (no
    drift, no crash)."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0, tick_interval=20.0)
    r.add_to(ax)
    xs0, ys0 = r.tick_artists[0].get_data()
    for _ in range(3):
        r._on_layout_change()
    xs1, ys1 = r.tick_artists[0].get_data()
    assert xs0 == pytest.approx(xs1)
    assert ys0 == pytest.approx(ys1)
    plt.close(fig)


def test_reflow_guard_prevents_reentry():
    """The ``_relayout_in_progress`` flag must prevent a callback
    fired during the rebuild from re-entering ``_on_layout_change``
    in a loop."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0, tick_interval=20.0)
    r.add_to(ax)
    # Simulate reentry: set the guard, then call the callback —
    # it should bail without doing anything.
    r._relayout_in_progress = True
    pre_tick_count = len(r.tick_artists)
    r._on_layout_change()
    post_tick_count = len(r.tick_artists)
    assert post_tick_count == pre_tick_count
    r._relayout_in_progress = False     # cleanup
    plt.close(fig)


def test_reflow_rebuilds_labels_and_endcaps_too():
    """Layout change rebuilds every artist family, not just the
    tick lines: labels and endcaps must also reflow."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0,
              endcap_style="arrow",
              title="reflow me")
    r.add_to(ax)
    pre_label_text = r.label_artists[0].get_text()
    pre_endcap_count = len(r.endcap_artists)
    pre_title_text = r.title_artist.get_text()

    r._on_layout_change()

    # All artists rebuilt — same logical content, just new instances
    assert r.label_artists[0].get_text() == pre_label_text
    assert len(r.endcap_artists) == pre_endcap_count
    assert r.title_artist.get_text() == pre_title_text
    plt.close(fig)


def test_reflow_survives_after_remove():
    """Calling remove() and then triggering a resize event must not
    crash — the disconnected callbacks should no longer fire on
    this ruler instance."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0, tick_interval=20.0)
    r.add_to(ax)
    r.remove()
    # Manually firing the (now-no-op) callback should also be safe.
    r._on_layout_change()      # host_axes is None → returns early
    plt.close(fig)


# ---- SkyCoord auto-resolve on the canonical / polar / zero entry points ----

def test_canonical_constructor_accepts_skycoord_pair():
    """Passing SkyCoord objects to the canonical constructor
    auto-projects to pixel via ``ax.wcs``."""
    from astropy.coordinates import SkyCoord

    fig, ax = _wcs_axes(cdelt_asec=2.0)
    c1 = SkyCoord(180.005, -0.01, unit="deg")
    c2 = SkyCoord(180.005, +0.01, unit="deg")
    r = Ruler(c1, c2, ax=ax)
    # On a 100×100 TAN at center (180, 0) with cdelt=2 asec/pix,
    # ±0.01° = ±36 arcsec = ±18 px from CRPIX2=50.5; expected
    # y-coords roughly 32.5 and 68.5 (within numeric tolerance).
    assert r.xy1[1] == pytest.approx(31.5, abs=1.0)
    assert r.xy2[1] == pytest.approx(67.5, abs=1.0)
    plt.close(fig)


def test_canonical_constructor_mixed_skycoord_and_tuple():
    """SkyCoord + numeric tuple is OK — each endpoint is
    resolved independently."""
    from astropy.coordinates import SkyCoord

    fig, ax = _wcs_axes(cdelt_asec=2.0)
    c1 = SkyCoord(180.0, 0.0, unit="deg")    # roughly the axes center
    # A SkyCoord always means world and ignores coord_type; the numeric pair
    # is governed by it, so pin 'pixel' to mix the two meanings explicitly.
    r = Ruler(c1, (75, 50), ax=ax, coord_type="pixel")
    assert r.xy2 == (75.0, 50.0)
    plt.close(fig)


def test_canonical_constructor_skycoord_without_ax_raises():
    from astropy.coordinates import SkyCoord
    c = SkyCoord(180.0, 0.0, unit="deg")
    with pytest.raises(ValueError, match="requires ax= with a WCS"):
        Ruler(c, c)


def test_canonical_constructor_vector_skycoord_raises():
    from astropy.coordinates import SkyCoord

    fig, ax = _wcs_axes(cdelt_asec=2.0)
    cc = SkyCoord([180.0, 180.005], [0.0, 0.01], unit="deg")
    with pytest.raises(ValueError, match="scalar"):
        Ruler(cc, (50, 50), ax=ax)
    plt.close(fig)


def test_canonical_constructor_bad_xy_type_raises():
    fig, ax = _wcs_axes(cdelt_asec=2.0)
    with pytest.raises(TypeError, match="SkyCoord or"):
        Ruler("not a coord", (50, 50), ax=ax)
    plt.close(fig)


def test_from_polar_accepts_skycoord_anchor():
    from astropy.coordinates import SkyCoord

    fig, ax = _wcs_axes(cdelt_asec=2.0)
    c = SkyCoord(180.0, 0.0, unit="deg")
    r = Ruler.from_polar(c, length=20.0, angle=0.0,
                          ax=ax, length_unit="arcsec",
                          angle_convention="plot")
    # Anchor at ~(50.5, 50.5) → +20'' east at plot=0 → (50.5+10, 50.5)
    assert r.xy1[0] == pytest.approx(50.5, abs=1.0)
    assert r.xy2[0] - r.xy1[0] == pytest.approx(10.0, rel=0.01)
    plt.close(fig)


def test_from_zero_accepts_skycoord_anchor():
    from astropy.coordinates import SkyCoord

    fig, ax = _wcs_axes(cdelt_asec=2.0)
    c = SkyCoord(180.0, 0.0, unit="deg")
    r = Ruler.from_zero(c, extent=10.0, angle=0.0,
                         ax=ax, length_unit="arcsec",
                         angle_convention="plot")
    # Symmetric around c → lambda0 = 0.5
    assert r._lambda0 == pytest.approx(0.5)
    plt.close(fig)


# ---- Ruler.from_axes_fraction factory + clip_on -----------------------------

def test_from_axes_fraction_basic_placement():
    """``from_axes_fraction((0, 0), (1, 0))`` spans the bottom edge
    of the axes data range."""
    fig, ax = _plain_axes()
    # _plain_axes uses xlim/ylim 0-100
    r = Ruler.from_axes_fraction((0.0, 0.0), (1.0, 0.0), ax=ax)
    assert r._uses_axes_frac is True
    assert r._xy1_axfrac == (0.0, 0.0)
    assert r._xy2_axfrac == (1.0, 0.0)
    assert r.xy1 == pytest.approx((0.0, 0.0))
    assert r.xy2 == pytest.approx((100.0, 0.0))
    plt.close(fig)


def test_from_axes_fraction_outside_frame_for_twin_axis():
    """Negative y_frac places the spine *below* the bottom edge —
    the twin-axis layout. Resulting data-coord y is negative."""
    fig, ax = _plain_axes()
    r = Ruler.from_axes_fraction((0.0, -0.05), (1.0, -0.05), ax=ax)
    # y_frac=-0.05 on a 100-unit y range → data y = -5
    assert r.xy1[1] == pytest.approx(-5.0)
    assert r.xy2[1] == pytest.approx(-5.0)
    plt.close(fig)


def test_from_axes_fraction_clip_on_defaults_to_false():
    """``from_axes_fraction`` defaults to ``clip_on=False`` so the
    spine renders cleanly when placed outside the axes box."""
    fig, ax = _plain_axes()
    r = Ruler.from_axes_fraction((0.0, -0.05), (1.0, -0.05), ax=ax)
    assert r._clip_on is False
    plt.close(fig)


def test_from_axes_fraction_clip_on_override():
    fig, ax = _plain_axes()
    r = Ruler.from_axes_fraction((0.0, 0.0), (1.0, 0.0), ax=ax,
                                  clip_on=True)
    assert r._clip_on is True
    plt.close(fig)


def test_from_axes_fraction_dynamic_pinning_under_pan():
    """When the user pans the axes (xlim changes), the ruler stays
    visually pinned at its axes-fraction position — endpoints
    re-project to new data coords."""
    fig, ax = _plain_axes()
    r = Ruler.from_axes_fraction((0.25, 0.5), (0.75, 0.5), ax=ax)
    r.add_to(ax)
    assert r.xy1 == pytest.approx((25.0, 50.0))
    assert r.xy2 == pytest.approx((75.0, 50.0))

    # Pan: shift xlim to (50, 150). The same axes-fraction positions
    # (0.25, 0.75) now map to data x = 75 and 125.
    ax.set_xlim(50, 150)
    # On Agg the xlim_changed callback should fire; verify
    # by checking the endpoints did refresh.
    # In case it doesn't fire automatically, force it.
    if not r._uses_axes_frac:
        pytest.fail("flag was lost")
    if r.xy1 == pytest.approx((25.0, 50.0)):
        r._on_layout_change()    # manual fire
    assert r.xy1 == pytest.approx((75.0, 50.0), abs=0.5)
    assert r.xy2 == pytest.approx((125.0, 50.0), abs=0.5)
    plt.close(fig)


def test_from_axes_fraction_dynamic_pinning_under_resize():
    """Same as above but for figure resize."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    fig.canvas.draw()
    r = Ruler.from_axes_fraction((0.0, -0.05), (1.0, -0.05), ax=ax)
    r.add_to(ax)
    y_before = r.xy1[1]
    fig.set_size_inches(12, 6)
    r._on_layout_change()
    y_after = r.xy1[1]
    # Same axes-frac position, same data x/y range, but the layout
    # may have shifted slightly due to figure-resize effects.
    # The y position should still round to ~-5 (5% below y=0).
    assert y_after == pytest.approx(y_before, abs=0.5)
    plt.close(fig)


def test_from_axes_fraction_requires_ax():
    """``ax`` is required — the factory needs ``ax.transAxes`` to
    project the fraction coordinates."""
    with pytest.raises(TypeError):
        # Missing ax kwarg
        Ruler.from_axes_fraction((0, 0), (1, 0))


def test_clip_on_canonical_default_is_true():
    """Backward-compat: canonical constructor still defaults to
    ``clip_on=True`` (matplotlib default for Line2D)."""
    r = Ruler((10, 50), (90, 50))
    assert r._clip_on is True


def test_clip_on_propagates_to_all_artists():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0,
              endcap_style="arrow",
              title="x",
              clip_on=False)
    r.add_to(ax)
    assert r.line_artist.get_clip_on() is False
    for tick in r.tick_artists:
        assert tick.get_clip_on() is False
    for lab in r.label_artists:
        assert lab.get_clip_on() is False
    for cap in r.endcap_artists:
        assert cap.get_clip_on() is False
    assert r.title_artist.get_clip_on() is False
    plt.close(fig)


# ---- title_beyond_labels auto-clearance -----------------------------------

def test_title_beyond_labels_default_false():
    """Default ``title_beyond_labels=False`` keeps the v1 "gap
    beyond tick tip" semantic for ``title_offset``."""
    r = Ruler((10, 50), (90, 50), title="x")
    assert r._title_beyond_labels is False


def test_title_beyond_labels_pushes_title_past_labels():
    """``title_beyond_labels=True`` with the title on the same side
    as the labels positions the title past the rendered label
    bboxes — substantially further from the spine than the v1
    "gap beyond tick tip" semantic would put it for the same
    ``title_offset``."""
    fig, ax = _plain_axes()
    # Twin-axis-style layout: vertical spine, labels horizontal on
    # the +x side, title on the same +x side.
    r_auto = Ruler((50, 10), (50, 90), pixscale_asec=1.0,
                    tick_interval=20.0,
                    tick_side="right", label_side="right",
                    label_rotation="horizontal",
                    title="Title",
                    title_side="right",
                    title_offset=4.0,
                    title_beyond_labels=True)
    r_auto.add_to(ax)
    auto_x = r_auto.title_artist.get_position()[0]

    r_manual = Ruler((50, 10), (50, 90), pixscale_asec=1.0,
                      tick_interval=20.0,
                      tick_side="right", label_side="right",
                      label_rotation="horizontal",
                      title="Title",
                      title_side="right",
                      title_offset=4.0,
                      title_beyond_labels=False)
    r_manual.add_to(ax)
    manual_x = r_manual.title_artist.get_position()[0]

    # Auto-clearance pushes the title further (past the label bbox)
    # than the v1 gap-beyond-tick-tip semantic would for the same
    # title_offset value.
    assert auto_x > manual_x
    plt.close(fig)


def test_title_beyond_labels_noop_when_title_on_opposite_side():
    """When the title is on the *opposite* side from the labels,
    the auto-clearance flag has no effect (no labels to clear on
    the title's side)."""
    fig, ax = _plain_axes()
    # Default label_side='auto' on tick_side='both' → label_sign=+1;
    # title_side='auto' → opposite = -1. Different sides.
    r_auto = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
                    tick_interval=20.0,
                    title="Title",
                    title_beyond_labels=True)
    r_auto.add_to(ax)
    pos_auto = r_auto.title_artist.get_position()

    r_off = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
                   tick_interval=20.0,
                   title="Title",
                   title_beyond_labels=False)
    r_off.add_to(ax)
    pos_off = r_off.title_artist.get_position()

    # Same position whether the flag is True or False — the title
    # is on the labels' opposite side, so auto-clearance is a no-op.
    assert pos_auto[0] == pytest.approx(pos_off[0])
    assert pos_auto[1] == pytest.approx(pos_off[1])
    plt.close(fig)


def test_title_beyond_labels_with_no_labels_falls_through():
    """If labels=False (no labels rendered), the auto-clearance
    branch falls through and the title uses the v1 semantic."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              tick_interval=20.0,
              labels=False,
              title="Title only",
              title_side="left",          # same side label_sign would use
              title_beyond_labels=True)
    r.add_to(ax)
    # Title still placed at the v1 offset; no crash.
    assert r.title_artist is not None
    plt.close(fig)


# ---- repr -------------------------------------------------------------------

def test_repr_summarizes_mode_and_distance():
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0)
    text = repr(r)
    assert "straight" in text
    # 80 arcsec total → auto-formats as arcmin (since >= 60″)
    assert "′" in text


def test_repr_marks_geodesic_mode():
    fig, ax = _wcs_axes(cdelt_asec=2.0)
    r = Ruler((35, 50), (65, 50), ax=ax, geodesic=True)
    text = repr(r)
    assert "geodesic" in text
    plt.close(fig)


def test_repr_handles_unknown_distance():
    """Without a pixel scale or WCS, the repr just notes the
    unknown distance rather than crashing."""
    r = Ruler((10, 50), (90, 50))
    text = repr(r)
    assert "unknown" in text


# ===== Stroke kwarg uniformity =====

def test_ruler_stroke_off_by_default():
    """No stroke kwargs → no path effects on the main line."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0)
    r.add_to(ax)
    assert not r.line_artist.get_path_effects()
    plt.close(fig)


def test_ruler_stroke_kwargs_apply_to_line():
    """stroke_color/stroke_lw at construction set path effects on the line."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              stroke_color="white", stroke_lw=3.0)
    r.add_to(ax)
    assert len(r.line_artist.get_path_effects()) == 1
    plt.close(fig)


def test_ruler_set_line_updates_stroke():
    """set_line(stroke_color=...) rebuilds the path effects in place."""
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0)
    r.add_to(ax)
    assert not r.line_artist.get_path_effects()
    r.set_line(stroke_color="k", stroke_lw=2.5)
    assert len(r.line_artist.get_path_effects()) == 1
    # Disable explicitly.
    r.set_line(stroke_color=None)
    assert not r.line_artist.get_path_effects()
    plt.close(fig)


def test_ruler_explicit_path_effects_wins_over_stroke_kwargs():
    """If both path_effects= and stroke_color= are given, path_effects wins."""
    import matplotlib.patheffects as PathEffects
    fig, ax = _plain_axes()
    custom_pe = [PathEffects.withStroke(linewidth=5.0, foreground="magenta")]
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0,
              path_effects=custom_pe,
              stroke_color="white", stroke_lw=2.0)
    r.add_to(ax)
    pe = r.line_artist.get_path_effects()
    assert len(pe) == 1
    assert pe[0] is custom_pe[0]
    plt.close(fig)


# ---- minor ticks -----------------------------------------------------------

def test_ruler_minor_ticks_off_by_default():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0).add_to(ax)
    assert r.minor_tick_artists == []
    plt.close(fig)


def test_ruler_minor_ticks_subdivide_major_interval():
    """minor_ticks=n splits each major interval into n (n-1 minors between
    adjacent majors), matching matplotlib's AutoMinorLocator semantic."""
    fig, ax = _plain_axes()
    # 80 px long, majors every 20 → 5 majors; minor_ticks=4 → step 5 →
    # 17 grid points − 5 coinciding with majors = 12 minor positions.
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0, tick_interval=20,
              minor_ticks=4).add_to(ax)
    assert len(r._minor_tick_positions()) == 12
    # 'both' sides → one Line2D per side per position
    assert len(r.minor_tick_artists) == 24
    plt.close(fig)


def test_ruler_minor_ticks_never_overprint_a_major():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0, tick_interval=20,
              minor_ticks=4)
    majors = [t for _, t in r._major_tick_positions()]
    for m in r._minor_tick_positions():
        assert all(abs(m - mm) > 1e-9 for mm in majors)
    plt.close(fig)


def test_ruler_minor_tick_interval_overrides_subdivision():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0, tick_interval=20,
              minor_ticks=4, minor_tick_interval=10).add_to(ax)
    # step 10 over 80 → 9 grid points − 5 majors = 4 minors, ×2 sides
    assert len(r.minor_tick_artists) == 8
    plt.close(fig)


def test_ruler_minor_tick_side_follows_tick_side_then_overrides():
    fig, ax = _plain_axes()
    one = Ruler((10, 50), (90, 50), pixscale_asec=1.0, tick_interval=20,
                minor_ticks=4, tick_side="left").add_to(ax)
    assert len(one.minor_tick_artists) == 12          # 'auto' → one side
    both = Ruler((10, 70), (90, 70), pixscale_asec=1.0, tick_interval=20,
                 minor_ticks=4, tick_side="left",
                 minor_tick_side="both").add_to(ax)
    assert len(both.minor_tick_artists) == 24         # explicit override
    off = Ruler((10, 30), (90, 30), pixscale_asec=1.0, tick_interval=20,
                minor_ticks=4, minor_tick_side="none").add_to(ax)
    assert off.minor_tick_artists == []
    plt.close(fig)


def test_ruler_minor_ticks_shorter_than_major_by_default():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0, tick_interval=20,
              minor_ticks=4, tick_length=8.0).add_to(ax)
    assert r._resolved_minor_length() == 4.0

    def _len(line):
        (x0, x1), (y0, y1) = line.get_xdata(), line.get_ydata()
        return float(np.hypot(x1 - x0, y1 - y0))
    assert _len(r.minor_tick_artists[0]) < _len(r.tick_artists[0])
    plt.close(fig)


def test_ruler_minor_ticks_auto_subdivision():
    """'auto' mirrors AutoMinorLocator: a 2-leading major step → 4, else 5."""
    from skyplothelper.overlays.ruler import _auto_minor_subdivisions
    assert _auto_minor_subdivisions(20.0) == 4
    assert _auto_minor_subdivisions(10.0) == 5
    assert _auto_minor_subdivisions(50.0) == 5
    assert _auto_minor_subdivisions(0.0) == 5        # degenerate → fallback


def test_ruler_minor_ticks_removed_with_ruler():
    fig, ax = _plain_axes()
    r = Ruler((10, 50), (90, 50), pixscale_asec=1.0, minor_ticks=5).add_to(ax)
    assert r.minor_tick_artists
    r.remove()
    assert r.minor_tick_artists == []
    plt.close(fig)


@pytest.mark.parametrize("bad", [0, 1, "nope", 2.5])
def test_ruler_minor_ticks_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        Ruler((0, 0), (1, 1), minor_ticks=bad)


def test_ruler_minor_tick_side_rejects_unknown():
    with pytest.raises(ValueError):
        Ruler((0, 0), (1, 1), minor_tick_side="sideways")


def test_ruler_minor_ticks_inherit_and_pin_style():
    fig, ax = _plain_axes()
    inherit = Ruler((10, 50), (90, 50), pixscale_asec=1.0, tick_interval=20,
                    minor_ticks=4, tick_color="red", tick_lw=2.0).add_to(ax)
    assert inherit.minor_tick_artists[0].get_color() == "red"
    assert inherit.minor_tick_artists[0].get_linewidth() == 2.0
    pinned = Ruler((10, 70), (90, 70), pixscale_asec=1.0, tick_interval=20,
                   minor_ticks=4, tick_color="red",
                   minor_tick_color="blue", minor_tick_lw=0.5).add_to(ax)
    assert pinned.minor_tick_artists[0].get_color() == "blue"
    assert pinned.minor_tick_artists[0].get_linewidth() == 0.5
    plt.close(fig)


# ---- coord_type: what a bare numeric pair means ----------------------------

def test_bare_pair_means_world_on_a_sky_axes():
    """Default 'auto': on a WCS axes a numeric pair is SKY DEGREES, matching
    Reticle and the add_* family. Previously it silently meant pixels, so the
    two sibling overlays disagreed on identical input."""
    fig, ax = _wcs_axes(cdelt_asec=2.0, npix=100)
    # The axes is centered on (180, 0); the center in world coords must land
    # near the middle pixel, NOT at pixel (180, 0).
    r = Ruler((180.0, 0.0), (180.0, 0.01), ax=ax)
    assert 30 < r.xy1[0] < 70
    plt.close(fig)


def test_bare_pair_means_data_coords_on_a_plain_axes():
    """'auto' on a non-WCS axes keeps matplotlib's own meaning: data coords."""
    fig, ax = _plain_axes()
    r = Ruler((10, 20), (90, 80), ax=ax)
    assert r.xy1 == (10.0, 20.0)
    plt.close(fig)


def test_coord_type_pixel_escape_hatch_on_a_sky_axes():
    fig, ax = _wcs_axes(cdelt_asec=2.0, npix=100)
    r = Ruler((30, 50), (70, 50), ax=ax, coord_type="pixel")
    assert r.xy1 == (30.0, 50.0)
    plt.close(fig)


def test_coord_type_world_without_a_wcs_raises():
    fig, ax = _plain_axes()
    with pytest.raises(ValueError, match="WCS"):
        Ruler((180.0, 0.0), (180.0, 1.0), ax=ax, coord_type="world")
    plt.close(fig)


def test_skycoord_ignores_coord_type():
    """A SkyCoord carries its own frame, so coord_type never applies to it."""
    from astropy.coordinates import SkyCoord

    fig, ax = _wcs_axes(cdelt_asec=2.0, npix=100)
    c1 = SkyCoord(180.0, -0.01, unit="deg")
    c2 = SkyCoord(180.0, +0.01, unit="deg")
    a = Ruler(c1, c2, ax=ax, coord_type="pixel")
    b = Ruler(c1, c2, ax=ax, coord_type="world")
    assert a.xy1 == b.xy1 and a.xy2 == b.xy2
    plt.close(fig)


def test_coord_type_rejects_unknown():
    fig, ax = _wcs_axes(cdelt_asec=2.0, npix=100)
    with pytest.raises(ValueError, match="coord_type"):
        Ruler((10, 10), (20, 20), ax=ax, coord_type="sideways")
    plt.close(fig)


def test_stroke_lw_survives_a_later_set_line():
    """A constructor stroke width must not be lost on set_line().

    ``set_line`` rebuilds the stroke from ``stroke_color`` plus "the current
    stroke width" — but the width was never stored, so it silently snapped
    back to the 2.5 default and discarded what the ruler was built with.
    """
    r = sph.Ruler((0, 0), (1, 1), stroke_color="w", stroke_lw=6.0)
    assert r._stroke_lw == pytest.approx(6.0)
    r.set_line(stroke_color="k")                 # no width given -> keep 6.0
    assert r._stroke_lw == pytest.approx(6.0)
    widths = [getattr(pe, "_gc", {}).get("linewidth")
              for pe in (r._path_effects or [])]
    assert any(w == pytest.approx(6.0) for w in widths if w is not None)


def test_set_line_can_still_change_the_stroke_width():
    r = sph.Ruler((0, 0), (1, 1), stroke_color="w", stroke_lw=6.0)
    r.set_line(stroke_color="w", stroke_lw=2.0)
    assert r._stroke_lw == pytest.approx(2.0)


def test_ruler_stroke_reaches_labels_and_title():
    """The cartographic outline stroke backs the tick labels + title too, not
    only the line/ticks — text most needs the legibility outline."""
    import matplotlib.pyplot as plt

    import skyplothelper as sph
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    r = sph.Ruler((10, 20), (90, 20), coord_type="pixel", pixscale_asec=1.0,
                  title="scale", stroke_color="k", stroke_lw=3).add_to(ax)
    assert r._label_artists and all(
        lab.get_path_effects() for lab in r._label_artists)
    assert r._title_artist.get_path_effects()
    plt.close(fig)


def test_ruler_no_stroke_leaves_text_plain():
    import matplotlib.pyplot as plt

    import skyplothelper as sph
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    r = sph.Ruler((10, 20), (90, 20), coord_type="pixel", pixscale_asec=1.0,
                  title="scale", stroke_color=None).add_to(ax)
    assert not any(lab.get_path_effects() for lab in r._label_artists)
    plt.close(fig)
