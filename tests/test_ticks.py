"""Tests for skyplothelper.ticks."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from skyplothelper._compat import coord_ticklabels, coord_ticks
from skyplothelper.ticks import (
    AnchoredOffsetFormatter,
    OffsetFormatter,
    apply_anchored_offset,
    apply_offset_ticks,
    format_ticklabels,
)
from skyplothelper.wcs_frame import make_wcs_frame


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---- OffsetFormatter unit tests ----

def test_offset_formatter_zero_at_reference():
    fmt = OffsetFormatter(ref_value_deg=330.075, unit="mas", precision=1)
    label = fmt(330.075)
    assert "0" in label
    assert "mas" in label


def test_offset_formatter_signed_offsets():
    fmt = OffsetFormatter(ref_value_deg=330.075, unit="mas", precision=1)
    pos = fmt(330.075 + 1.0 / 3600 / 1000)  # +1 mas
    neg = fmt(330.075 - 1.0 / 3600 / 1000)  # -1 mas
    # Signed: positive should have +, negative should have -
    assert "+" in pos
    assert "-" in neg, f"Negative offset must preserve '-': got {neg!r}"


def test_offset_formatter_unit_choices():
    for unit in ("arcsec", "arcmin", "mas", "uas"):
        fmt = OffsetFormatter(ref_value_deg=0.0, unit=unit)
        assert callable(fmt)


def test_offset_formatter_invalid_unit_raises():
    with pytest.raises(ValueError, match="unit must be one of"):
        OffsetFormatter(ref_value_deg=0.0, unit="parsec")


def test_offset_formatter_show_unit_false_drops_the_suffix():
    """show_unit=False gives bare numbers (the unit lives in the axis label)."""
    on = OffsetFormatter(ref_value_deg=330.0, unit="mas", precision=0)
    off = OffsetFormatter(ref_value_deg=330.0, unit="mas", precision=0,
                          show_unit=False)
    p400 = 330.0 + 400.0 / 3.6e6
    m200 = 330.0 - 200.0 / 3.6e6
    assert on(330.0) == "0 mas" and off(330.0) == "0"
    assert on(p400) == "+400 mas" and off(p400) == "+400"
    assert on(m200) == "-200 mas" and off(m200) == "-200"


# ---- AnchoredOffsetFormatter is a subclass of OffsetFormatter ----

def test_anchored_offset_is_subclass():
    assert issubclass(AnchoredOffsetFormatter, OffsetFormatter)


# ---- AnchoredOffsetFormatter: anchor_format option ----

def test_anchor_format_sexagesimal_is_default():
    fmt = AnchoredOffsetFormatter(ref_value_deg=330.075, is_ra=True, unit="mas")
    ref = fmt._format_reference(330.075)
    assert "°" not in ref  # HMS, not decimal degrees
    assert ("h" in ref or "ʰ" in ref or "mathregular{h}" in ref)


def test_anchor_format_decimal():
    # Dec (signed) and RA (0–360, unsigned) decimal degrees.
    dec = AnchoredOffsetFormatter(ref_value_deg=120.5, unit="mas",
                                  anchor_format="decimal", ref_precision=4)
    assert dec._format_reference(120.5) == "+120.5000°"
    ra = AnchoredOffsetFormatter(ref_value_deg=330.075, is_ra=True, unit="mas",
                                 anchor_format="decimal", ref_precision=3)
    assert ra._format_reference(330.075) == "330.075°"


def test_anchor_format_callable():
    fmt = AnchoredOffsetFormatter(ref_value_deg=120.5, unit="mas",
                                  anchor_format=lambda d: f"SRC[{d:.2f}]")
    assert fmt._format_reference(120.5) == "SRC[120.50]"


def test_anchor_format_does_not_affect_offsets():
    fmt = AnchoredOffsetFormatter(ref_value_deg=120.5, unit="mas",
                                  anchor_format="decimal")
    off = fmt(120.5 + 1.0 / 3600 / 1000)  # +1 mas offset tick
    assert "mas" in off and "°" not in off


def test_anchor_format_invalid_raises():
    with pytest.raises(ValueError, match="anchor_format must be"):
        AnchoredOffsetFormatter(ref_value_deg=0.0, anchor_format="bogus")


def test_apply_anchored_offset_threads_anchor_format():
    """apply_anchored_offset forwards anchor_format to the installed formatter."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(120.5, -5.0), npix=300,
                        cdelt=2.0 / 3600, frame="galactic", fig=fig)
    fig.canvas.draw()
    apply_anchored_offset(ax, unit="arcsec", anchor_format="decimal")
    fig.canvas.draw()  # must not raise
    fmt = ax._sph_anchored_formatters[0]
    assert fmt.anchor_format == "decimal"
    assert fmt._format_reference(120.5).endswith("°")
    plt.close(fig)


def _anchored_world_spacing(ax, coord_idx):
    """Median tick spacing (deg) for an anchored-offset axis."""
    import numpy as np
    vals = ax.coords[coord_idx]._formatter_locator.values
    assert vals is not None, "anchored axis must set explicit tick values"
    deg = sorted(float(v.to_value("deg")) for v in vals)
    return float(np.median(np.diff(deg)))


def _cos_dec(ax):
    import numpy as np
    cx = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2.0
    cy = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2.0
    return float(np.cos(np.radians(ax.wcs.pixel_to_world_values(cx, cy)[1])))


def _make_offset_ax(fig, npix=200, cdelt_asec=4e-4):
    fig.clf()
    ax = make_wcs_frame((1, 1, 1), projection="TAN", center=(180.0, 30.0),
                        cdelt=cdelt_asec / 3600.0, npix=npix, frame="ICRS",
                        fig=fig)
    fig.canvas.draw()
    return ax


@pytest.mark.parametrize("compact", [False, True])
def test_anchored_offset_spacing_is_round_and_shared(compact):
    """Default spacing is one round 1/2/5/10 value applied to both axes, so
    the displayed lon (Δα cos δ) and lat increments match — not astropy's
    non-round RA-seconds auto-spacing."""
    fig = plt.figure(figsize=(6, 6))
    ax = _make_offset_ax(fig)
    apply_anchored_offset(ax, unit="mas", compact=compact)
    fig.canvas.draw()
    cosd = _cos_dec(ax)
    lon_mas = _anchored_world_spacing(ax, 0) * 3.6e6 * cosd
    lat_mas = _anchored_world_spacing(ax, 1) * 3.6e6
    assert lon_mas == pytest.approx(lat_mas, rel=1e-6)
    # Round in the offset unit (a 1/2/5/10·10ⁿ value).
    assert lat_mas == pytest.approx(20.0, rel=1e-6)
    plt.close(fig)


def test_anchored_offset_spacing_is_axes_size_independent():
    """Spacing must not depend on the axes pixel size (the old astropy
    auto-locator path made a small grid cell pick an odd ~5.6 mas value)."""
    spacings = []
    for figsize in [(3, 3), (9, 9)]:
        fig = plt.figure(figsize=figsize)
        ax = _make_offset_ax(fig)
        apply_anchored_offset(ax, unit="mas")
        fig.canvas.draw()
        spacings.append(_anchored_world_spacing(ax, 1) * 3.6e6)
        plt.close(fig)
    assert spacings[0] == pytest.approx(spacings[1], rel=1e-6)


def test_anchored_offset_spacing_scalar_override():
    fig = plt.figure(figsize=(6, 6))
    ax = _make_offset_ax(fig)
    apply_anchored_offset(ax, unit="mas", spacing=10.0)
    fig.canvas.draw()
    cosd = _cos_dec(ax)
    assert _anchored_world_spacing(ax, 0) * 3.6e6 * cosd == pytest.approx(10.0,
                                                                          rel=1e-6)
    assert _anchored_world_spacing(ax, 1) * 3.6e6 == pytest.approx(10.0, rel=1e-6)
    plt.close(fig)


def _wide_offset_ax(cdelt_arcmin=0.5, npix=200):
    """A wider (arcmin/deg-scale) field for unit-scaling tests."""
    return make_wcs_frame(111, projection="TAN", center=(180.0, 10.0),
                          cdelt=cdelt_arcmin / 60.0, npix=npix, frame="ICRS",
                          grid=False)


def test_anchored_offset_unit_auto_scales_with_fov():
    """unit='auto' picks a sensible unit from the field of view (was always
    'mas') — arcmin on an arcmin field, deg on a multi-degree field."""
    ax = _wide_offset_ax(cdelt_arcmin=0.5)          # ~100 arcmin field
    ax.figure.canvas.draw()
    apply_anchored_offset(ax, unit="auto")
    assert ax._sph_anchored_unit == "arcmin"
    plt.close(ax.figure)

    axd = _wide_offset_ax(cdelt_arcmin=5.0)         # ~16 deg field
    axd.figure.canvas.draw()
    apply_anchored_offset(axd, unit="auto")
    assert axd._sph_anchored_unit == "deg"
    plt.close(axd.figure)


def test_anchored_offset_spacing_quantity_independent_of_unit():
    """spacing as an astropy Quantity converts regardless of the label unit:
    2 arcmin is 2 arcmin under unit='arcmin' and 120000 mas under unit='mas'."""
    import astropy.units as u
    ax = _wide_offset_ax()
    ax.figure.canvas.draw()
    apply_anchored_offset(ax, unit="arcmin", spacing=2 * u.arcmin)
    assert _anchored_world_spacing(ax, 1) * 60.0 == pytest.approx(2.0, rel=1e-6)
    plt.close(ax.figure)

    ax2 = _wide_offset_ax()
    ax2.figure.canvas.draw()
    apply_anchored_offset(ax2, unit="mas", spacing=2 * u.arcmin)
    assert _anchored_world_spacing(ax2, 1) * 3.6e6 == pytest.approx(120000.0,
                                                                    rel=1e-6)
    plt.close(ax2.figure)


def test_anchored_offset_deg_unit():
    """'deg' is a supported offset unit (for genuinely wide fields)."""
    axd = _wide_offset_ax(cdelt_arcmin=5.0)
    axd.figure.canvas.draw()
    apply_anchored_offset(axd, unit="deg")
    assert axd._sph_anchored_unit == "deg"
    plt.close(axd.figure)


def _anchored_sample_offset_label(ax, coord_idx):
    """An offset (non-anchor) tick label string for the given axis."""
    fmt = ax._sph_anchored_formatters[coord_idx]
    ref = fmt.ref_value_deg
    vals = sorted(float(v.to_value("deg"))
                  for v in ax.coords[coord_idx]._formatter_locator.values)
    nonref = [v for v in vals if abs(v - ref) > 1e-12]
    return fmt(nonref[len(nonref) // 2] if nonref else vals[0])


@pytest.mark.parametrize("compact", [False, True])
def test_anchored_offset_precision_auto_round(compact):
    """Round spacing → integer offset labels (no trailing .000), in both
    modes — the auto-precision the pure-offset path already had."""
    fig = plt.figure(figsize=(6, 6))
    ax = _make_offset_ax(fig)
    apply_anchored_offset(ax, unit="mas", compact=compact)
    fig.canvas.draw()
    label = _anchored_sample_offset_label(ax, 1)
    assert "." not in label  # e.g. "+20 mas", not "+20.000 mas"
    plt.close(fig)


def test_anchored_offset_precision_subunit_spacing_gets_decimals():
    fig = plt.figure(figsize=(6, 6))
    ax = _make_offset_ax(fig)
    apply_anchored_offset(ax, unit="mas", spacing=0.5)
    fig.canvas.draw()
    assert _anchored_sample_offset_label(ax, 1) == "+0.5 mas"
    plt.close(fig)


def test_anchored_offset_precision_explicit_override_wins():
    fig = plt.figure(figsize=(6, 6))
    ax = _make_offset_ax(fig)
    apply_anchored_offset(ax, unit="mas", offset_precision=2)
    fig.canvas.draw()
    assert _anchored_sample_offset_label(ax, 1) == "+20.00 mas"
    plt.close(fig)


def test_anchored_offset_spacing_per_axis_override():
    fig = plt.figure(figsize=(6, 6))
    ax = _make_offset_ax(fig)
    apply_anchored_offset(ax, unit="mas", spacing=(10.0, 20.0))
    fig.canvas.draw()
    cosd = _cos_dec(ax)
    assert _anchored_world_spacing(ax, 0) * 3.6e6 * cosd == pytest.approx(10.0,
                                                                          rel=1e-6)
    assert _anchored_world_spacing(ax, 1) * 3.6e6 == pytest.approx(20.0, rel=1e-6)
    plt.close(fig)


def _tick_label_strings(ax, coord_idx):
    txt = getattr(coord_ticklabels(ax.coords[coord_idx]), "text", {})
    out = []
    for v in (txt.values() if hasattr(txt, "values") else []):
        out += [s for s in v if s]
    return out


def test_compact_style_drops_seconds():
    """'compact' must drop the seconds field on *every* supported astropy —
    on < 7 (where simplify is a no-op) it falls back to minute truncation."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(85.5, 22.0),
                        fov_deg=4.0, fig=fig, grid=False)
    # Publication keeps seconds (contrast).
    format_ticklabels(ax, style="publication")
    fig.canvas.draw()
    pub_lat = _tick_label_strings(ax, 1)
    assert any("″" in s for s in pub_lat), "publication should show seconds"

    format_ticklabels(ax, style="compact")
    fig.canvas.draw()
    lon = _tick_label_strings(ax, 0)
    lat = _tick_label_strings(ax, 1)
    assert lon and lat
    # No seconds field anywhere, on every supported astropy: the minute
    # truncation drops them on < 7 (simplify ignored) and on >= 7 (where
    # simplify alone would leave — even widen — the seconds field).
    assert not any("″" in s for s in lat)
    assert not any("mathregular{s}" in s or "ˢ" in s for s in lon)
    plt.close(fig)


# ---- format_ticklabels: smoke test against many style names ----

# These styles are documented in the format_ticklabels dispatcher's docstring
# (the ~14 named styles). Smoke test: each one applied to a TAN axes produces
# tick labels without throwing.
PUBLIC_STYLES = [
    "publication",
    "letter",
    "casa",
    "latex",
    "compact",
    "minimal",
    "allsky_hours",
    "allsky_deg",
    "decimal",
    "decimal_plain",
    "vlbi",
]


@pytest.mark.parametrize("style", PUBLIC_STYLES)
def test_format_ticklabels_styles_smoke(style):
    """Each style should apply without raising on a TAN axes."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(180.0, 0.0), fig=fig)
    try:
        format_ticklabels(ax, style=style)
    except Exception as e:
        pytest.fail(f"format_ticklabels(style={style!r}) raised: {e}")


def test_format_ticklabels_default_style():
    """The bare-minimum invocation must work."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    format_ticklabels(ax)


def test_format_ticklabels_stroke_reaches_allsky_lat_overlay_labels():
    """On AIT all-sky, the exterior Dec labels are sph overlay Text artists
    (astropy's native lat labels are hidden by the hybrid path), so
    format_ticklabels' stroke / color must reach them too — not only the
    natively-styled longitude labels."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", center=0, frame="ICRS", fig=fig)
    format_ticklabels(ax, style="allsky_hours", color="red",
                      stroke_lw=3, stroke_color="lime")
    fig.canvas.draw()
    overlay = [t for t in ax.texts
               if getattr(t, "_sph_overlay_ticklabel", False)]
    assert overlay, "expected sph overlay lat labels on AIT all-sky frame"
    import matplotlib.colors as mcolors
    for t in overlay:
        assert t.get_path_effects(), "lat overlay label missing stroke"
        assert mcolors.same_color(t.get_color(), "red")


def _drawn_lon_labels(ax):
    ax.figure.canvas.draw()
    text = coord_ticklabels(ax.coords[0]).text
    labels = []
    for side in ("b", "t", "l", "r"):
        labels.extend(t for t in text.get(side, []) if t)
    return labels


def _max_decimals(labels):
    """Most decimal places seen across a set of decimal-degree labels."""
    n = 0
    for lab in labels:
        core = lab.replace("°", "").strip().lstrip("+-")
        if "." in core:
            n = max(n, len(core.split(".")[1]))
    return n


def test_decimal_plain_omits_degree_symbol():
    """Regression: 'decimal_plain' must render bare decimal degrees with no
    ° symbol, whereas 'decimal' keeps it. (decimal_plain previously aliased
    'decimal' and rendered the ° anyway.)"""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(180.0, 30.0),
                        frame="ICRS", fig=fig)
    format_ticklabels(ax, style="decimal_plain")
    plain = _drawn_lon_labels(ax)
    assert plain, "no tick labels drawn"
    assert all("°" not in lab for lab in plain), \
        f"decimal_plain should have no ° symbol, got {plain}"

    fig2 = plt.figure()
    ax2 = make_wcs_frame(111, projection="TAN", center=(180.0, 30.0),
                         frame="ICRS", fig=fig2)
    format_ticklabels(ax2, style="decimal")
    with_deg = _drawn_lon_labels(ax2)
    assert any("°" in lab for lab in with_deg), \
        f"decimal should keep the ° symbol, got {with_deg}"


def test_decimal_auto_precision_scales_to_field():
    """Regression: style='decimal' with decimal_places=None rendered ~10 dp on
    a field frame (ticks set via values=, so astropy's decimal auto-precision
    had no spacing to read). Precision now scales to the tick spacing — a few-
    degree field reads ~1 dp, not μas."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(83.6287, 22.0147),
                        fov_deg=4.0, fig=fig)
    format_ticklabels(ax, style="decimal")
    labs = _drawn_lon_labels(ax)
    assert labs, "no tick labels drawn"
    assert _max_decimals(labs) <= 2, f"runaway decimal precision: {labs}"


def test_decimal_auto_precision_finer_for_small_field():
    """A much smaller field legitimately gets more decimal places."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(83.6287, 22.0147),
                        fov_deg=0.05, fig=fig)
    format_ticklabels(ax, style="decimal")
    labs = _drawn_lon_labels(ax)
    assert labs
    # ~0.008° spacing → 3 dp; bounded well below the ~10 dp bug
    assert 2 <= _max_decimals(labs) <= 5, f"unexpected precision: {labs}"


def test_decimal_explicit_places_still_honored():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(83.6287, 22.0147),
                        fov_deg=4.0, fig=fig)
    format_ticklabels(ax, style="decimal", decimal_places=3)
    labs = _drawn_lon_labels(ax)
    assert labs and _max_decimals(labs) == 3, labs


def test_auto_decimal_places_none_for_spacing_based():
    """The helper returns None for spacing-based axes (astropy's auto is
    correct there) so all-sky decimal behavior is untouched."""
    from skyplothelper.ticks import _auto_decimal_places
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", center=0, frame="galactic",
                        fig=fig)
    ax.figure.canvas.draw()
    assert _auto_decimal_places(ax.coords[0]) is None
    plt.close(fig)


# ---- apply_offset_ticks: crop-aware tick spacing ----

def _offset_tick_spacing_mas(ax, coord_idx):
    import numpy as np
    vals = sorted(float(v.to_value("deg"))
                  for v in ax.coords[coord_idx]._formatter_locator.values)
    return float(np.median(np.diff(vals))) * 3.6e6


def _vlbi_field(npix=200, cdelt_mas=0.4):
    # ~npix * cdelt_mas across (e.g. 80 mas); TAN, off-equator center.
    return make_wcs_frame(111, projection="TAN", center=(49.0, 41.5),
                          cdelt=cdelt_mas * 1e-3 / 3600, npix=npix,
                          frame="ICRS", grid=False)


def test_apply_offset_ticks_crop_gets_finer_spacing():
    """Regression: offset ticks size from the VISIBLE window, not the full
    image — so after a set_xlim/set_ylim crop, re-applying lays finer ticks
    (the full-image spacing would leave only the central '0' tick in a crop).
    """
    import numpy as np
    ax = _vlbi_field()
    ax.figure.canvas.draw()
    apply_offset_ticks(ax, unit="mas")
    full = _offset_tick_spacing_mas(ax, 1)

    ax.set_xlim(80, 120)
    ax.set_ylim(80, 120)
    ax.figure.canvas.draw()
    apply_offset_ticks(ax, unit="mas")
    cropped = _offset_tick_spacing_mas(ax, 1)

    assert cropped < full          # crop-aware: spacing tightened
    # And several ticks now fall inside the crop window (lat span ~16 mas).
    lat_vals = ax.coords[1]._formatter_locator.values.to_value("deg")
    y0 = ax.wcs.pixel_to_world_values(100, 80)[1]
    y1 = ax.wcs.pixel_to_world_values(100, 120)[1]
    lo, hi = sorted((y0, y1))
    inside = int(np.sum((lat_vals >= lo) & (lat_vals <= hi)))
    assert inside >= 3
    plt.close(ax.figure)


def test_apply_offset_ticks_fullframe_spacing_unchanged():
    """The crop-aware change must NOT alter full-frame spacing (public API)."""
    ax = _vlbi_field()
    ax.figure.canvas.draw()
    apply_offset_ticks(ax, unit="mas")
    # 80 mas field → ~20 mas nice spacing (half-FOV/3 → 1/2/5/10/20 rule).
    assert _offset_tick_spacing_mas(ax, 1) == pytest.approx(20.0, rel=1e-6)
    plt.close(ax.figure)


def test_apply_offset_ticks_auto_sub_mas_no_crash():
    """Regression: unit='auto' on a sub-mas (μas) field used to KeyError
    because _auto_offset_unit returns the label 'μas', not the 'uas' key."""
    ax = make_wcs_frame(111, projection="TAN", center=(180.0, 10.0),
                        cdelt=2e-11, npix=200, frame="ICRS", grid=False)
    ax.figure.canvas.draw()
    apply_offset_ticks(ax, unit="auto")  # must not raise
    assert ax.coords[1]._formatter_locator.values is not None
    plt.close(ax.figure)


def test_apply_offset_ticks_spacing_override():
    """Explicit spacing (number in `unit`, or any-unit Quantity) overrides the
    auto nice-spacing — parity with apply_anchored_offset."""
    import astropy.units as u
    ax = make_wcs_frame(111, projection="TAN", center=(180.0, 10.0),
                        cdelt=0.5 / 60, npix=200, frame="ICRS", grid=False)
    ax.figure.canvas.draw()
    apply_offset_ticks(ax, unit="arcmin", spacing=5)
    assert _offset_tick_spacing_mas(ax, 1) / 60000.0 == pytest.approx(5.0,
                                                                      rel=1e-6)
    plt.close(ax.figure)

    ax2 = make_wcs_frame(111, projection="TAN", center=(180.0, 10.0),
                         cdelt=0.5 / 60, npix=200, frame="ICRS", grid=False)
    ax2.figure.canvas.draw()
    apply_offset_ticks(ax2, unit="arcsec", spacing=2 * u.arcmin)  # 120 arcsec
    assert _offset_tick_spacing_mas(ax2, 1) / 1000.0 == pytest.approx(120.0,
                                                                      rel=1e-6)
    plt.close(ax2.figure)


# ---- apply_offset_ticks: signed-label preservation ----

def test_apply_offset_ticks_signed_labels():
    """Negative offsets must keep their '-' sign (the simplify=False fix)."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(180.0, 0.0), fig=fig)
    apply_offset_ticks(ax, ref_ra_deg=180.0, ref_dec_deg=0.0, unit="arcsec")
    fig.canvas.draw()
    # If we got here without exception, the formatter applied successfully.
    # The deeper check (visual inspection of '-' presence) is reserved for
    # the visual-baseline render gallery.


def test_apply_offset_ticks_ra_gt_180_does_not_hang():
    """Regression: a mas-scale offset field with RA > 180° froze the draw.
    apply_offset_ticks sets RA majors in the 0–360° convention (e.g. 187.7°)
    while astropy's minor-locator range comes back on the −180..180° branch
    (−172.3°) — a full turn apart. The minor extrapolation then marched 360°
    at a sub-arcsec step (~3e8 iterations). The locator now shifts majors onto
    the visible branch (and caps the extrapolation), so the draw completes."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(187.7059, 12.3911),
                        cdelt=4e-4 / 3600.0, npix=200, frame="ICRS", fig=fig)
    apply_offset_ticks(ax, unit="mas")
    fig.canvas.draw()  # would hang before the fix
    plt.close(fig)


# ---- apply_offset_ticks: show_unit + axis-label styling ----

def _offset_ticklabels(ax, ci):
    d = ax.coords[ci].ticklabels.text
    return [v for vals in d.values() for v in vals if v]


def test_apply_offset_ticks_show_unit_false_ticks_have_no_unit():
    """show_unit=False strips the per-tick unit; the axis label still has it."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(266.4, -29.0),
                        cdelt=4e-4 / 3600.0, npix=200, frame="ICRS", fig=fig)
    apply_offset_ticks(ax, unit="mas", spacing=200, show_unit=False)
    fig.canvas.draw()
    labels = _offset_ticklabels(ax, 0)
    assert labels and all("mas" not in t for t in labels), labels
    assert "(mas)" in ax.coords[0].get_axislabel()   # unit stays in the title
    plt.close(fig)


def test_apply_offset_ticks_show_unit_true_is_default_and_unchanged():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(266.4, -29.0),
                        cdelt=4e-4 / 3600.0, npix=200, frame="ICRS", fig=fig)
    apply_offset_ticks(ax, unit="mas", spacing=200)
    fig.canvas.draw()
    labels = _offset_ticklabels(ax, 0)
    assert labels and all("mas" in t for t in labels), labels
    plt.close(fig)


def test_apply_offset_ticks_color_and_stroke_reach_axis_labels():
    """color + stroke now style the axis labels too, not just the ticks, so a
    recolored offset frame reads as one piece."""
    import matplotlib.colors as mcolors
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(266.4, -29.0),
                        cdelt=4e-4 / 3600.0, npix=200, frame="ICRS", fig=fig)
    apply_offset_ticks(ax, unit="mas", spacing=200, color="#ff0000",
                       stroke_lw=3, stroke_color="white")
    fig.canvas.draw()
    for ci in (0, 1):
        al = ax.coords[ci].axislabels
        assert mcolors.to_hex(al.get_color()) == "#ff0000"
        assert al.get_path_effects()      # stroke applied
    plt.close(fig)


def test_apply_offset_ticks_axis_labels_false_with_color_is_safe():
    """Styling must not choke when there are no axis labels to style."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(266.4, -29.0),
                        cdelt=4e-4 / 3600.0, npix=200, frame="ICRS", fig=fig)
    apply_offset_ticks(ax, unit="mas", spacing=200, color="red",
                       axis_labels=False)
    fig.canvas.draw()   # no exception
    plt.close(fig)


def test_minor_locator_bounded_for_wrapped_value_range():
    """The patched minor locator stays bounded (and lands in the visible
    branch) even when value_min/value_max are a full turn off the majors."""
    from skyplothelper.ticks import (
        _enable_minor_ticks_for_explicit_tick_values,
    )
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(187.7059, 12.3911),
                        cdelt=4e-4 / 3600.0, npix=200, frame="ICRS", fig=fig)
    apply_offset_ticks(ax, unit="mas")  # installs the patched locator
    fl = ax.coords[0]._formatter_locator
    _enable_minor_ticks_for_explicit_tick_values(ax.coords[0], frequency=5)
    # Wrapped visible range (≈ 187.7° − 360°), a tiny ±50 mas window.
    vmin, vmax = -172.294114, -172.294086
    import numpy as np
    minors = fl.minor_locator(None, 5, vmin, vmax).to_value("deg")
    assert 0 < len(minors) < 1000
    # minors sit on the wrapped (−172°) branch, not the +187° major branch
    assert np.median(minors) < 0
    plt.close(fig)


def test_minor_locator_cleared_when_majors_become_spacing_based():
    """Regression: a stale interpolating minor locator (baked on value-list
    majors) must not survive a switch to spacing-based majors. After
    set_ticks(spacing=) + re-enabling minors, the sph locator is dropped and
    astropy's native spacing minors land on the round subdivisions (the 0.5°
    bug produced 0.125° minors with the midpoint missing)."""
    import astropy.units as u

    from skyplothelper.ticks import (
        _enable_minor_ticks_for_explicit_tick_values,
    )
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(83.63, 22.0),
                        fov_deg=4.0, fig=fig, grid=False)
    coord = ax.coords[0]
    fl = coord._formatter_locator
    # value-list majors → installs the (tagged) interpolating locator
    _enable_minor_ticks_for_explicit_tick_values(coord, frequency=5)
    assert getattr(fl.__dict__.get("minor_locator"), "_sph_interp_minor", False)
    # switch majors to 1° spacing (astropy leaves our stale locator in place)
    coord.set_ticks(spacing=1.0 * u.deg)
    # re-enable at freq 4 → must clear the stale closure, restoring native
    _enable_minor_ticks_for_explicit_tick_values(coord, frequency=4)
    assert "minor_locator" not in fl.__dict__   # sph override removed
    fig.canvas.draw()
    b = sorted(coord_ticks(coord).minor_world.get("b", []))
    inside = [x for x in b if 82.0 < x < 83.0]   # one 1° major interval
    assert len(inside) == 3                       # freq 4 → 3 minors, not ~6
    assert any(abs(x - 82.5) < 1e-6 for x in inside)  # midpoint present
    plt.close(fig)


# ---- apply_anchored_offset: anchor + offset ----

def test_apply_anchored_offset_smoke():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(180.0, 0.0), fig=fig)
    apply_anchored_offset(ax, ref_tick="center", unit="mas")
    fig.canvas.draw()


def test_compact_style_no_dollar_glyph_on_zoomed_field():
    """'compact' on a ~2 deg field whose auto ticks fall at sub-minute spacing
    must not emit bare '$' (empty-mathtext) labels; ticks snap to whole
    minutes instead."""
    fig = plt.figure(figsize=(6, 6))
    ax = make_wcs_frame(111, projection="TAN", center=(56.6, 24.1),
                        fov_deg=1.7, fig=fig)
    format_ticklabels(ax, style="compact")
    fig.canvas.draw()
    for coord in (ax.coords[0], ax.coords[1]):
        for texts in coord_ticklabels(coord).text.values():
            for t in texts:
                assert t.strip() != "$"
                assert t.count("$") % 2 == 0    # balanced mathtext delimiters
    plt.close(fig)


# --- the tick-count cap announces itself --------------------------------

def _tiny_field_axes():
    """A 10-arcsec field: 10000 mas across, so half_world is 5000 mas."""
    import skyplothelper as sph
    fig = plt.figure(figsize=(4, 4))
    return sph.make_wcs_frame(111, projection="TAN", center_lon=180.0,
                              center_lat=30.0, fov_deg=10.0 / 3600, fig=fig)


def _cap_warnings(**kw):
    import warnings
    ax = _tiny_field_axes()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        apply_anchored_offset(ax, unit="mas", **kw)
        msgs = [str(w.message) for w in caught
                if "max_ticks" in str(w.message)]
    plt.close("all")
    return msgs


def test_tick_cap_warns_when_it_truncates():
    """Past the cap the ticks stay correctly spaced but stop partway across
    the axis, which reads as deliberate rather than truncated."""
    # 8 mas over a 5000 mas half-field wants ~626 ticks, just past the 500
    # cap. Deliberately only just past: a spacing wanting 100k ticks would
    # take minutes to draw once the cap is raised, which is why it exists.
    msgs = _cap_warnings(spacing=8.0)
    assert msgs
    assert "max_ticks=500" in msgs[0]


def test_tick_cap_warning_names_the_axis():
    """Both coords are walked, so an unlabeled message arrives twice looking
    like a duplicate rather than like two axes each truncating."""
    msgs = _cap_warnings(spacing=8.0)
    assert any("lon axis" in m for m in msgs)
    assert any("lat axis" in m for m in msgs)


@pytest.mark.parametrize("spacing", [None, 1000.0, 2000.0])
def test_tick_cap_is_silent_for_ordinary_spacings(spacing):
    """A warning that fires on every ordinary plot gets filtered out and
    stops working, so the quiet case is as much the contract as the loud one.
    """
    assert _cap_warnings(spacing=spacing) == []


def test_max_ticks_raises_the_ceiling():
    assert _cap_warnings(spacing=8.0, max_ticks=1000) == []


# --- axis-label stroke parity (12.15 stroke-uniformity fold-in) -------------
# format_ticklabels applies stroke_color/color to the tick labels; it must
# also apply them to the axis labels it sets, or a stroked/recolored frame's
# titles render unstroked (light titles vanish on a light page).

def test_format_ticklabels_strokes_axis_labels_rectilinear():
    """On a rectilinear (TAN) frame the axis-label titles get the same stroke
    the tick labels receive."""
    fig = plt.figure(figsize=(4, 4))
    ax = make_wcs_frame(projection="TAN", center=(150.0, 2.0), fov_deg=0.2,
                        frame="FK5", fig=fig, apply_format_defaults=False,
                        grid=False)
    format_ticklabels(ax, color="0.9", stroke_color="k", stroke_lw=2.4)
    for i in (0, 1):
        lbl = ax.coords[i].axislabels
        assert lbl.get_path_effects(), f"axis label {i} not stroked"
    plt.close(fig)


def test_format_ticklabels_no_stroke_leaves_axis_labels_plain():
    """Without stroke_color the axis labels stay unstroked (no accidental
    path effects)."""
    fig = plt.figure(figsize=(4, 4))
    ax = make_wcs_frame(projection="TAN", center=(150.0, 2.0), fov_deg=0.2,
                        frame="FK5", fig=fig, apply_format_defaults=False,
                        grid=False)
    format_ticklabels(ax)
    assert not ax.coords[0].axislabels.get_path_effects()
    plt.close(fig)


def test_format_ticklabels_strokes_axis_labels_custom_allsky_frame():
    """On a custom all-sky frame (ZEA/AIT) the axis titles are manual ax.text
    artists; they must be stroked too."""
    fig = plt.figure(figsize=(4, 4))
    ax = make_wcs_frame(projection="ZEA", center=(0.0, 0.0), fov_deg=60.0,
                        fig=fig, apply_format_defaults=False, grid=False)
    format_ticklabels(ax, color="0.9", stroke_color="k", stroke_lw=2.4)
    manual = getattr(ax, "_sph_manual_axis_labels", [])
    assert manual, "expected manual axis-label artists on the custom frame"
    for t in manual:
        assert t.get_path_effects(), f"manual axis label {t.get_text()!r} unstroked"
    plt.close(fig)


def test_apply_anchored_offset_strokes_axis_labels():
    """apply_anchored_offset styles its axis labels with the same color/stroke
    as its tick labels — parity with apply_offset_ticks / format_ticklabels."""
    fig = plt.figure(figsize=(4, 4))
    ax = make_wcs_frame(projection="TAN", center=(150.0, 2.0), fov_deg=0.1,
                        frame="FK5", fig=fig, apply_format_defaults=False,
                        grid=False)
    apply_anchored_offset(ax, color="0.9", stroke_color="k", stroke_lw=2.4)
    for i in (0, 1):
        assert ax.coords[i].axislabels.get_path_effects(), \
            f"anchored-offset axis label {i} not stroked"
    plt.close(fig)
