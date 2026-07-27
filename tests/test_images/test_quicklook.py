"""Smoke tests for skyplothelper.images.quicklook."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.io import fits

from skyplothelper._compat import coord_ticklabels
from skyplothelper.images.quicklook import (
    BEAM_INK,
    INK_DARK,
    INK_LIGHT,
    _contrast_ink,
    _estimate_background,
    _over_image_ink,
    quicklook_plot,
    simpleimage_figure,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _synthetic_fits_data():
    """A synthetic 2D image with a header for quicklook testing."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((100, 100)).astype(float) + 5.0
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = 100
    hdr["NAXIS2"] = 100
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRPIX1"] = 50.5
    hdr["CRPIX2"] = 50.5
    hdr["CRVAL1"] = 180.0
    hdr["CRVAL2"] = 0.0
    hdr["CDELT1"] = -1.0 / 3600
    hdr["CDELT2"] = 1.0 / 3600
    return data, hdr


def test_quicklook_plot_smoke():
    data, hdr = _synthetic_fits_data()
    result = quicklook_plot(data, header=hdr)
    assert result.fig is not None
    assert result.ax is not None


def test_quicklook_default_colormap_is_sph_deepsky():
    """The default image colormap is the bundled sph.deepsky (unified look)."""
    data, hdr = _synthetic_fits_data()
    r = quicklook_plot(data, header=hdr, image=True, contours=False)
    assert r.image.get_cmap().name == "sph.deepsky"


def test_quicklook_sph_cmap_resolves_without_registry():
    """A bundled 'sph.*' colormap default renders even if the matplotlib
    registry entry is missing — quicklook resolves it via the registration-
    independent colormaps.get_colormap fallback."""
    import matplotlib as mpl
    data, hdr = _synthetic_fits_data()
    saved = mpl.colormaps["sph.deepsky"]
    mpl.colormaps.unregister("sph.deepsky")
    try:
        r = quicklook_plot(data, header=hdr, image=True, contours=False)
        assert r.image.get_cmap().name == "sph.deepsky"
    finally:
        mpl.colormaps.register(saved, name="sph.deepsky")


def _has_frame_stroke(ax):
    return any(p.get_gid() == "_sph_frame_stroke" for p in ax.patches)


def _spine_color(ax):
    import matplotlib.colors as mcolors
    return mcolors.to_hex(next(iter(ax.spines.values())).get_edgecolor())


def test_quicklook_image_mode_default_gray_frame():
    """image=True defaults the spine/tick color to medium gray (0.5) so a
    black frame doesn't hide over a through-black colormap; no stroke."""
    import matplotlib.colors as mcolors
    data, hdr = _synthetic_fits_data()
    r = quicklook_plot(data, header=hdr, image=True, colormap="inferno")
    assert _spine_color(r.ax) == mcolors.to_hex("0.5")
    assert not _has_frame_stroke(r.ax)


def test_quicklook_contour_mode_black_frame_no_stroke():
    """Contour mode (image=False, light bg): black frame, no stroke."""
    import matplotlib.colors as mcolors
    data, hdr = _synthetic_fits_data()
    # image=False is now explicit: it stopped being the default in the
    # 2026-07-19 defaults change, and this test is specifically about the
    # contour-only look.
    r = quicklook_plot(data, header=hdr, image=False)
    assert _spine_color(r.ax) == mcolors.to_hex("black")
    assert not _has_frame_stroke(r.ax)


def test_quicklook_frame_color_overrides_default():
    import matplotlib.colors as mcolors
    data, hdr = _synthetic_fits_data()
    r = quicklook_plot(data, header=hdr, image=True, frame_color="red")
    assert _spine_color(r.ax) == mcolors.to_hex("red")


def test_quicklook_frame_stroke_bare_color_opts_in():
    """A bare color opts into the frame stroke."""
    data, hdr = _synthetic_fits_data()
    r = quicklook_plot(data, header=hdr, image=True, frame_stroke="white")
    assert _has_frame_stroke(r.ax)


def test_quicklook_frame_stroke_dict_custom():
    """A dict opts into a custom stroke (color + lw)."""
    data, hdr = _synthetic_fits_data()
    r = quicklook_plot(data, header=hdr, frame_stroke={"color": "0.1", "lw": 3})
    strokes = [p for p in r.ax.patches if p.get_gid() == "_sph_frame_stroke"]
    assert len(strokes) == 1
    assert strokes[0].get_linewidth() == 3


def test_apply_frame_stroke_public_wcsaxes():
    """Public sph.apply_frame_stroke strokes a WCSAxes frame + clears it."""
    import skyplothelper as sph
    assert "apply_frame_stroke" in sph.__all__
    data, hdr = _synthetic_fits_data()
    r = quicklook_plot(data, header=hdr, image=True)
    sph.apply_frame_stroke(r.ax, "white", 3.0)
    assert _has_frame_stroke(r.ax)
    sph.apply_frame_stroke(r.ax, None)          # clear (idempotent)
    assert not _has_frame_stroke(r.ax)


def test_apply_frame_stroke_plain_axes():
    """Public helper also works on a plain Axes (path_effects on spines)."""
    import skyplothelper as sph
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((8, 8)))
    sph.apply_frame_stroke(ax, "white", 3.0)
    assert all(len(sp.get_path_effects()) >= 1 for sp in ax.spines.values())


def test_apply_frame_stroke_follows_frame_shape():
    """The frame stroke follows the actual frame shape: a 4-corner path on a
    rectangular frame, a many-vertex path on an elliptical (AIT) frame."""
    import skyplothelper as sph

    def _stroke_verts(ax):
        sp = [p for p in ax.patches if p.get_gid() == "_sph_frame_stroke"]
        assert len(sp) == 1
        return sp[0].get_path().vertices.shape[0]

    plt.figure()
    ax_rect = sph.make_wcs_frame(121, projection="TAN", center=(180.0, 0.0))
    sph.apply_frame_stroke(ax_rect, "white", 3.0)
    ax_ell = sph.make_wcs_frame(122, projection="AIT")
    sph.apply_frame_stroke(ax_ell, "white", 3.0)
    assert _stroke_verts(ax_rect) <= 8            # rectangle
    assert _stroke_verts(ax_ell) > 50             # ellipse follows the curve


def _high_dynamic_range_image():
    """A 7500:1 source: faint noise + bright core + a faint jet (~50× rms)."""
    rng = np.random.default_rng(0)
    img = np.abs(rng.normal(0.0, 4e-4, (120, 120)))
    img[60, 60] = 3.0          # bright core
    img[60, 40:60] += 0.02     # faint jet
    return img


def test_quicklook_explicit_stretch_uses_rms_range():
    """An explicit stretch= without vmin/vmax must frame on the rms-based
    range (vmin≈-3·rms, vmax=peak), like the auto path — not the full
    [min, peak], which renders ~linear on a high-dynamic-range image."""
    img = _high_dynamic_range_image()
    r = quicklook_plot(img, contours=False, image=True, stretch="asinh")
    norm = r.ax.images[0].norm
    assert norm.vmin < 0.0                     # ≈ -3·rms, not nanmin (≥0)
    assert abs(norm.vmax - 3.0) < 1e-6         # peak
    # the faint jet must map well above the linear value (jet/peak ≈ 0.0067)
    assert float(norm(0.02)) > 0.1


def test_quicklook_explicit_vmin_vmax_still_honored():
    """Pinned vmin/vmax bypass the rms framing (caller takes control)."""
    img = _high_dynamic_range_image()
    r = quicklook_plot(img, contours=False, image=True, stretch="asinh",
                       vmin=0.0, vmax=1.0)
    norm = r.ax.images[0].norm
    assert abs(norm.vmin - 0.0) < 1e-9 and abs(norm.vmax - 1.0) < 1e-9


def _pedestal_image(pedestal=1000.0, noise=5.0):
    """A bright-sky-pedestal frame (optical/IR survey cutout): constant sky
    level + small noise + a bright compact source. The empty sky sits far
    above zero — the case that washes out a zero-anchored default."""
    rng = np.random.default_rng(3)
    img = pedestal + rng.normal(0.0, noise, (120, 120))
    img[58:62, 58:62] += 4000.0          # bright source well above the sky
    return img


def test_estimate_background_zero_vs_pedestal():
    """_estimate_background reads ~0 for a zero-mean (radio-like) frame and the
    pedestal level for a bright-sky frame."""
    rng = np.random.default_rng(1)
    zero_mean = rng.normal(0.0, 1e-3, (100, 100))
    assert abs(_estimate_background(zero_mean)) < 5e-4
    assert abs(_estimate_background(_pedestal_image()) - 1000.0) < 20.0


def test_quicklook_pedestal_floor_keeps_sky_black():
    """Regression: on a bright-sky pedestal the auto image floor anchors at the
    sky level (not zero), so the empty sky maps near black instead of landing
    mid-colormap and washing the frame out."""
    img = _pedestal_image(pedestal=1000.0)
    r = quicklook_plot(img, contours=False, image=True)
    norm = r.ax.images[0].norm
    # vmin sits just below the pedestal (~1000 − 3·rms), NOT near zero.
    assert norm.vmin > 900.0
    # the empty sky maps into the bottom of the colormap (near black)...
    assert float(norm(1000.0)) < 0.15
    # ...while the bright source still lands high.
    assert float(norm(5000.0)) > 0.7


def test_quicklook_zero_background_floor_unchanged():
    """The floor is a no-op for a zero-mean frame: vmin stays at ≈ −3·rms
    (below zero), preserving the radio-map default."""
    img = _high_dynamic_range_image()
    r = quicklook_plot(img, contours=False, image=True)
    assert r.ax.images[0].norm.vmin < 0.0


def test_simpleimage_figure_smoke():
    data, hdr = _synthetic_fits_data()
    result = simpleimage_figure(data, hdr)
    assert result.fig is not None
    assert result.image is not None


# ---- offset_coords / field_size / labels (quicklook tick/crop cluster) ----

def _lon_labels(ax):
    ax.figure.canvas.draw()
    t = coord_ticklabels(ax.coords[0])
    out = []
    for v in getattr(t, "text", {}).values():
        out += [s for s in v if s]
    return out


def test_quicklook_default_is_absolute_coords():
    """offset_coords=False (default) shows ABSOLUTE RA/Dec ticks + matching
    axis labels — not offset ticks under a "Right Ascension" label."""
    data, hdr = _synthetic_fits_data()
    r = quicklook_plot(data, header=hdr)
    assert r.ax.get_xlabel() == "Right Ascension"
    assert r.ax.get_ylabel() == "Declination"
    labels = _lon_labels(r.ax)
    # Absolute sexagesimal RA, not a relative "+N mas/arcsec" offset.
    assert labels and not any("mas" in s or "arcsec" in s for s in labels)


def test_quicklook_offset_coords_toggle_and_labels():
    """offset_coords=True activates offset ticks + relative labels."""
    data, hdr = _synthetic_fits_data()
    r = quicklook_plot(data, header=hdr, offset_coords=True,
                       offset_units="arcsec")
    assert r.ax.get_xlabel().startswith("Relative RA (arcsec) from ")
    assert r.ax.get_ylabel().startswith("Relative Dec (arcsec) from ")
    # Tick labels are relative offsets (arcsec symbol ″), not absolute HMS.
    labels = _lon_labels(r.ax)
    assert labels and not any("$^\\mathregular{h}$" in s for s in labels)


def test_quicklook_offset_ref_coord_shifts_reference():
    """ref_coord overrides the offset reference (was hardwired to center)."""
    from astropy.coordinates import SkyCoord
    data, hdr = _synthetic_fits_data()
    ref = SkyCoord(180.005, 0.003, unit="deg")
    r = quicklook_plot(data, header=hdr, offset_coords=True, ref_coord=ref)
    # The label encodes the reference; 180.005 deg = 12h00m01.2s, not the
    # image-center 12h00m00s.
    assert "12h00m01" in r.ax.get_xlabel()


def test_quicklook_field_size_crops():
    """field_size crops the view on the FITS path (was a silent no-op).

    The synthetic field is 100 px at 1 arcsec/px (= 100 arcsec wide).
    """
    data, hdr = _synthetic_fits_data()
    r_full = quicklook_plot(data, header=hdr)
    full_span = abs(r_full.ax.get_xlim()[1] - r_full.ax.get_xlim()[0])  # ~100

    # offset path: field_size in offset_units (40 arcsec → ~40 px).
    r_off = quicklook_plot(data, header=hdr, offset_coords=True,
                           offset_units="arcsec", field_size=40.0)
    off_span = abs(r_off.ax.get_xlim()[1] - r_off.ax.get_xlim()[0])
    assert off_span < full_span
    assert off_span == pytest.approx(40.0, abs=1.5)

    # absolute path: field_size in mas per the docstring (40000 mas = 40″).
    r_abs = quicklook_plot(data, header=hdr, field_size=40000.0)
    abs_span = abs(r_abs.ax.get_xlim()[1] - r_abs.ax.get_xlim()[0])
    assert abs_span < full_span
    assert abs_span == pytest.approx(40.0, abs=1.5)


# --- audit Track A: documented parameters that did nothing ---

import skyplothelper as sph  # noqa: E402


def _wcs_hdr(bunit=None):
    import astropy.io.fits as pyfits
    h = pyfits.Header()
    h['NAXIS'], h['NAXIS1'], h['NAXIS2'] = 2, 60, 60
    h['CTYPE1'], h['CTYPE2'] = 'RA---TAN', 'DEC--TAN'
    h['CRVAL1'], h['CRVAL2'] = 83.6, 22.0
    h['CRPIX1'] = h['CRPIX2'] = 30
    h['CDELT1'], h['CDELT2'] = -1e-4, 1e-4
    if bunit:
        h['BUNIT'] = bunit
    return h


def _deep_negative_image():
    rng = np.random.default_rng(0)
    d = rng.normal(0, 1, (80, 80))
    d[40, 40] = 40.0
    d[10:12, 10:12] = -12.0      # deep enough to admit two negative levels
    return d


def _negative_levels(**kw):
    fig, ax = plt.subplots()
    res = sph.quicklook_plot(_deep_negative_image(), ax=ax, contours=True,
                             colorbar=False, **kw)
    cs = res.neg_contour_set
    levels = [] if cs is None else [lv for lv in cs.levels if lv < 0]
    plt.close(fig)
    return levels


def test_n_negative_caps_the_negative_contour_levels():
    """`n_negative` was documented but never read."""
    assert len(_negative_levels(n_negative=1)) == 1
    assert len(_negative_levels(n_negative=2)) == 2


def test_n_negative_none_draws_as_many_as_fit():
    assert len(_negative_levels(n_negative=None)) >= 2


def test_n_negative_cannot_exceed_what_the_data_admits():
    assert len(_negative_levels(n_negative=99)) == \
        len(_negative_levels(n_negative=None))


def _unit_text(**kw):
    from astropy.wcs import WCS
    hdr = kw.pop('header', None)
    rng = np.random.default_rng(0)
    d = rng.normal(0, 1e-3, (60, 60))
    d[30, 30] = 1e-2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=WCS(hdr)) if hdr is not None \
        else fig.add_subplot(111)
    res = sph.quicklook_plot(d, ax=ax, header=hdr, contours=False,
                             colorbar=False, **kw)
    text = res.info_text.get_text()
    plt.close(fig)
    return [ln for ln in text.splitlines() if 'Peak =' in ln][0]


def test_bunit_is_read_from_the_header():
    """The docstring promised BUNIT auto-detection; nothing implemented it,
    so optical frames were labeled with the radio default."""
    assert 'electron/s' in _unit_text(header=_wcs_hdr('electron/s'))
    assert 'Jy/beam' not in _unit_text(header=_wcs_hdr('electron/s'))


def test_bunit_folds_the_display_factor_into_an_si_prefix():
    assert 'mJy/beam' in _unit_text(header=_wcs_hdr('Jy/beam'),
                                    display_factor=1e3)


def test_non_jy_bunit_keeps_an_explicit_factor():
    assert '[x1e+03] electron/s' in _unit_text(
        header=_wcs_hdr('electron/s'), display_factor=1e3)


def test_explicit_unit_still_overrides_bunit():
    assert 'counts' in _unit_text(header=_wcs_hdr('electron/s'),
                                  unit='counts')


def test_missing_bunit_asserts_no_unit_at_all():
    """The radio Jy/beam fallback was retired deliberately (2026-07-20).

    It was a guess, and it labeled every survey cutout without a BUNIT with a
    unit its data never claimed. A radio map whose header omits BUNIT now
    reads unitless until the caller says ``unit='Jy/beam'`` explicitly.
    """
    assert 'Jy/beam' not in _unit_text(header=_wcs_hdr())
    assert 'Jy/beam' in _unit_text(header=_wcs_hdr(), unit='Jy/beam')


# --- colorbar ticks: were pinned to absolute values 1..5000 (audit track C) ---

def _cbar(data, **kw):
    fig, ax = plt.subplots()
    res = sph.quicklook_plot(data, ax=ax, image=True, colorbar=True,
                             contours=False, **kw)
    fig.canvas.draw()
    major = [t.get_text() for t in res.colorbar.ax.get_yticklabels()]
    minor = list(res.colorbar.ax.yaxis.get_minorticklocs())
    plt.close(fig)
    return major, minor


def _peaked(peak=3.0, scale=0.2):
    rng = np.random.default_rng(0)
    d = rng.normal(0, scale, (60, 60))
    d[30, 30] = peak
    return d


def test_decade_minor_ticks_follow_the_data_scale():
    """The old FixedLocator listed absolute values 1..5000, so an image in
    0-1 or 0-1e6 display units got no usable minor ticks. The logic now lives
    in the shared _colorbar helper (used by both quicklook and add_colorbar)."""
    from skyplothelper._colorbar import decade_minor_ticks
    for lo, hi in ((0, 1), (0, 1e6), (0, 1e-3)):
        ticks = decade_minor_ticks(lo, hi)
        assert ticks, f"no minor ticks for range {lo}..{hi}"
        assert all(lo <= t <= hi for t in ticks)
    assert decade_minor_ticks(0, 0) == []


def test_cbar_tick_text_precision_follows_the_range():
    # Logic now lives in the shared _colorbar helper (used by both quicklook
    # and add_colorbar).
    from skyplothelper._colorbar import tick_text
    # a 0-3 bar must not collapse every tick to an integer
    labels = [tick_text(v, 0, 3) for v in (0, 0.75, 1.5, 2.25, 3)]
    assert len(set(labels)) == 5
    # a wide bar stays integral
    assert tick_text(2500, 0, 5000) == "2500"


def test_cbar_minor_ticks_linear_even_compressed_decade():
    """Linear bars get an even subdivision; compressed bars get decade
    multiples (1/2/3/5 x 10^k).

    The compressed side asserts the decade *mantissas* rather than a
    value-gap ratio. The old ratio>10 check keyed on the low-decade pile-up
    at the bar's base — which is exactly what the de-crowd filter now removes,
    so a ratio test would flag the fix as a regression. The mantissas are the
    stable property: de-crowd keeps a display-separated SUBSET of the decade
    multiples, so they stay within {1, 2, 3, 5}.
    """
    _, minor_linear = _cbar(_peaked(), stretch="linear")
    _, minor_log = _cbar(_peaked(), stretch="log")
    assert minor_linear and minor_log

    lin_gaps = np.diff(sorted(minor_linear))
    assert lin_gaps.max() / lin_gaps.min() <= 2.5, (
        "linear bar minor ticks should be near-uniform, got "
        f"{lin_gaps.min():g}..{lin_gaps.max():g}")

    mant = {round(v / 10 ** np.floor(np.log10(v)), 3)
            for v in minor_log if v > 0}
    assert mant and mant <= {1.0, 2.0, 3.0, 5.0}, (
        f"compressed bar should use decade multiples, got mantissas {mant}")


def test_cbar_knobs_override_the_defaults():
    major, _ = _cbar(_peaked(), cbar_format="%.2f")
    assert all(t == "" or t.count(".") == 1 for t in major)
    _, minor = _cbar(_peaked(), cbar_minor_ticks=[0.5, 1.5, 2.5])
    assert minor == [0.5, 1.5, 2.5]
    _, none_minor = _cbar(_peaked(), cbar_minor_ticks=False)
    assert none_minor == []


# --- defaults for the image-on world (2026-07-19) ---

def test_image_and_colorbar_are_on_by_default():
    """A bare quicklook_plot used to draw contours only, which surprised
    everyone who called it."""
    data, hdr = _synthetic_fits_data()
    r = sph.quicklook_plot(data, header=hdr)
    assert r.image is not None
    assert r.colorbar is not None
    plt.close(r.fig)


def test_contour_only_mode_drops_the_colorbar():
    """image=False is the classic difmap-style rendering; a colorbar with no
    mappable would be meaningless."""
    data, hdr = _synthetic_fits_data()
    r = sph.quicklook_plot(data, header=hdr, image=False)
    assert r.image is None
    assert r.colorbar is None
    plt.close(r.fig)


@pytest.mark.parametrize("cmap,expect_light", [
    ("sph.deepsky", True), ("viridis", True),
    ("gray_r", False), ("Blues", False),
])
def test_contour_ink_follows_the_colormap(cmap, expect_light):
    """Light contours vanish on a reversed map, which is common for optical
    data — so the ink is sampled from the colormap rather than hard-coded.

    Asserts the exact ink rather than a brightness threshold: the pair sits
    deliberately inside pure white/black, so a 'near-white' test would have
    to be re-tuned every time the constants are adjusted.
    """
    data, hdr = _synthetic_fits_data()
    r = sph.quicklook_plot(data, header=hdr, colormap=cmap)
    rgb = tuple(np.round(r.contour_set.get_edgecolor()[0][:3], 4))
    expected = mcolors.to_rgb(INK_LIGHT if expect_light else INK_DARK)
    assert rgb == pytest.approx(expected, abs=1e-3)
    plt.close(r.fig)


@pytest.mark.parametrize("cmap", ["sph.deepsky", "viridis", "gray_r", "Blues"])
def test_contour_stroke_never_matches_its_own_ink(cmap):
    """The stroke must be the ink's opposite, not a fixed color.

    A fixed dark stroke sat behind the dark ink a reversed colormap selects,
    so the stroke did nothing on exactly the maps where contrast is tightest.
    """
    ink = _over_image_ink(cmap, True, "k")
    assert mcolors.to_rgb(_contrast_ink(ink)) != mcolors.to_rgb(ink)


def test_contour_only_mode_keeps_the_callers_color():
    data, hdr = _synthetic_fits_data()
    r = sph.quicklook_plot(data, header=hdr, image=False)
    assert r.contour_set.get_edgecolor()[0][:3].max() < 0.1   # black
    plt.close(r.fig)


def test_contours_are_stroked_and_semi_transparent_by_default():
    data, hdr = _synthetic_fits_data()
    r = sph.quicklook_plot(data, header=hdr)
    assert r.contour_set.get_path_effects()
    assert r.contour_set.get_alpha() == pytest.approx(0.9)
    # 0.5 core + 0.7 stroke = 0.1 visible per side, 0.7 total -- the same
    # visual weight an unstroked 0.7 contour had. stroke_lw is a TOTAL
    # width, so leaving the core at 0.7 would have drawn a 1.1-wide line.
    assert float(np.atleast_1d(r.contour_set.get_linewidth())[0]) \
        == pytest.approx(0.5)
    plt.close(r.fig)


def test_contour_style_knobs_override_the_defaults():
    data, hdr = _synthetic_fits_data()
    r = sph.quicklook_plot(data, header=hdr, contour_color="#ff00ff",
                           contour_alpha=1.0, contour_stroke_color=None)
    assert tuple(r.contour_set.get_edgecolor()[0][:3].round(2)) == (1.0, 0.0, 1.0)
    assert r.contour_set.get_alpha() == pytest.approx(1.0)
    assert not r.contour_set.get_path_effects()
    plt.close(r.fig)


# --- beam legibility on masked maps -----------------------------------------

_ASEC = 1.0 / 3600.0


def _masked_moment_like():
    """Signal in the middle, NaN elsewhere -- including the beam's corner.

    The shape of a moment map or any cutout with blank sky, which is where the
    beam's colormap-derived ink stops being ink-on-image.
    """
    n = 120
    y, x = np.mgrid[0:n, 0:n]
    r = np.hypot(x - n / 2, y - n / 2)
    img = np.exp(-(r / 18.0) ** 2)
    img[r > 34] = np.nan
    h = fits.Header()
    h["NAXIS"], h["NAXIS1"], h["NAXIS2"] = 2, n, n
    h["CTYPE1"], h["CTYPE2"] = "RA---SIN", "DEC--SIN"
    h["CRPIX1"] = h["CRPIX2"] = n / 2
    h["CRVAL1"], h["CRVAL2"] = 180.0, 30.0
    h["CDELT1"], h["CDELT2"] = -_ASEC, _ASEC
    h["BUNIT"] = "JY/BEAM"
    return img, h


def _beam_pixels(colormap, **kw):
    """Pixels that change when the beam is drawn -- i.e. is it actually there.

    Measured on the rendered canvas rather than by inspecting the patch: the
    bug this guards against drew a perfectly good patch in the page color.
    """
    img, hdr = _masked_moment_like()
    bufs = []
    for beam in (False, True):
        bkw = (dict(beam_maj=14 * _ASEC, beam_min=8 * _ASEC, beam_pa=30.0)
               if beam else {})
        r = quicklook_plot(img, header=hdr, contours=False, colorbar=False,
                           show_info=False, colormap=colormap, **bkw, **kw)
        fig = r.ax.figure
        fig.patch.set_facecolor("white")
        fig.canvas.draw()
        bufs.append(np.asarray(fig.canvas.buffer_rgba()).copy())
        plt.close("all")
    a, b = bufs
    return int((np.abs(a.astype(int) - b.astype(int)).sum(axis=2) > 8).sum())


@pytest.mark.parametrize("colormap", ["sph.diff_blueorange", "sph.deepsky",
                                      "sph.penumbra"])
def test_beam_is_visible_on_a_masked_map(colormap):
    """Regression: the beam vanished entirely on masked maps.

    These colormaps all sample to white ink, and on a masked map the beam's
    corner is transparent -- so white ink landed on the white page.
    """
    assert _beam_pixels(colormap) > 100


def _beam_patch(**kw):
    """The Beam artist itself, for the knobs a pixel count cannot resolve."""
    img, hdr = _masked_moment_like()
    r = quicklook_plot(img, header=hdr, contours=False, colorbar=False,
                       show_info=False, colormap="sph.deepsky",
                       beam_maj=14 * _ASEC, beam_min=8 * _ASEC, beam_pa=30.0,
                       **kw)
    patch = r.ax.patches[-1]
    plt.close("all")
    return patch


def test_beam_stroke_can_be_disabled():
    """Checked on the artist, not the canvas: the beam's own ink is a mid-gray
    that stays visible without a stroke, so 'did it vanish' cannot answer this.
    """
    assert not _beam_patch(beam_stroke_color=None).get_path_effects()
    assert _beam_patch().get_path_effects()


def test_beam_ink_is_distinct_from_the_contour_ink():
    """The beam is an instrument property, not data — drawing it in the
    contour color makes it read as one more contour."""
    beam_rgb = mcolors.to_rgb(_beam_patch().get_edgecolor())
    assert beam_rgb == pytest.approx(mcolors.to_rgb(BEAM_INK), abs=1e-3)
    for cmap in ("sph.deepsky", "gray_r"):
        assert beam_rgb != mcolors.to_rgb(_over_image_ink(cmap, True, "k"))


def test_beam_stroke_color_is_honored():
    default = _beam_pixels("sph.deepsky")
    assert _beam_pixels("sph.deepsky", beam_stroke_color="red") != default


def test_beam_stroke_lw_is_honored():
    assert (_beam_pixels("sph.deepsky", beam_stroke_lw=5.0)
            > _beam_pixels("sph.deepsky"))


def test_dark_ink_beam_still_drawn_on_reversed_map():
    """The case that already worked must not regress."""
    assert _beam_pixels("gray_r") > 100


# --- info-block contour ladder ----------------------------------------------

def _contour_line(**kw):
    n = 120
    y, x = np.mgrid[0:n, 0:n]
    r = np.hypot(x - n / 2, y - n / 2)
    rng = np.random.default_rng(0)
    data = np.exp(-(r / 14.0) ** 2) * 4.0 + rng.normal(0, 1e-3, (n, n))
    h = _wcs_hdr("JY/BEAM")
    h["NAXIS1"] = h["NAXIS2"] = n
    h["CRPIX1"] = h["CRPIX2"] = n / 2
    res = quicklook_plot(data, header=h, rms=1e-3, contour_start=3,
                         contour_factor=2, show_info=True, colorbar=False,
                         **kw)
    txt = [t.get_text() for t in res.ax.figure.findobj(plt.Text)
           if "Contours:" in t.get_text()]
    plt.close("all")
    return txt[0]


def test_long_sigma_ladder_is_elided():
    """An unbounded ladder ran to ~108 chars and set the saved figure width
    under bbox_inches='tight'. The explicit-levels branch already elided."""
    line = _contour_line()
    ladder = line.split("Contours:")[-1]
    assert "..." in ladder
    assert len(ladder.splitlines()[0]) < 90


def test_elided_ladder_keeps_its_endpoints():
    """A geometric ladder is described by its endpoints, so both must survive."""
    ladder = _contour_line().split("$\\times$")[-1]
    assert "-1" in ladder          # first (negative) rung
    assert "1024" in ladder        # last rung


# --- auto stroke on the contour_cmap path -----------------------------------

def _contour_stroke_color(**kw):
    """The stroke color matplotlib actually received.

    Reads the path effect's own gc rather than re-deriving it: the bug this
    guards against came from re-deriving the value in the wrong place, so a
    test that repeats the derivation would have agreed with the bug.
    """
    data, hdr = _synthetic_fits_data()
    r = quicklook_plot(data, header=hdr, colorbar=False, show_info=False, **kw)
    pe = r.contour_set.get_path_effects()
    out = pe[0]._gc.get("foreground") if pe else None
    plt.close("all")
    return out


@pytest.mark.parametrize("colormap,expected", [
    ("gist_yarg", INK_DARK),     # light map -> dark stroke
    ("sph.deepsky", INK_LIGHT),  # dark map  -> light stroke
])
def test_contour_cmap_stroke_contrasts_with_the_image(colormap, expected):
    """With contour_cmap there is no single contour ink to contrast against,
    so the stroke must be chosen against the IMAGE.

    Regression: contour_kw carries 'cmap' and no 'colors' on this path, so the
    lookup fell through to `color` — the info-text color — and picked a stroke
    unrelated to what sits under the contours. On a light map that produced a
    white stroke, i.e. less legible than no stroke at all.
    """
    got = _contour_stroke_color(colormap=colormap, contour_cmap="viridis")
    assert mcolors.to_rgb(got) == pytest.approx(mcolors.to_rgb(expected),
                                                abs=1e-3)


def test_contour_cmap_stroke_is_not_derived_from_the_text_color():
    """The info-text `color` must not steer the contour stroke."""
    a = _contour_stroke_color(colormap="gist_yarg", contour_cmap="viridis")
    b = _contour_stroke_color(colormap="gist_yarg", contour_cmap="viridis",
                              color="white")
    assert mcolors.to_rgb(a) == pytest.approx(mcolors.to_rgb(b), abs=1e-3)


def test_plain_contour_stroke_still_contrasts_with_the_ink():
    """The non-cmap path is unchanged: stroke opposes the contour ink."""
    got = _contour_stroke_color(colormap="sph.deepsky")   # ink is INK_LIGHT
    assert mcolors.to_rgb(got) == pytest.approx(mcolors.to_rgb(INK_DARK),
                                                abs=1e-3)


# --- no BUNIT must not assert a unit ----------------------------------------

def _hdr_with(bunit=None):
    data, hdr = _synthetic_fits_data()
    hdr.pop("BUNIT", None)
    if bunit:
        hdr["BUNIT"] = bunit
    return data, hdr


def _bar_label(bunit=None, **kw):
    data, hdr = _hdr_with(bunit)
    r = quicklook_plot(data, header=hdr, colorbar=True, show_info=False, **kw)
    out = r.colorbar.ax.get_ylabel() if r.colorbar is not None else None
    plt.close("all")
    return out


def test_no_bunit_leaves_the_colorbar_unlabeled():
    """Regression: this fell through to the radio Jy/beam family, so a survey
    cutout with no BUNIT was labeled with a unit its data never claimed.

    A wrong unit in a published figure is worse than no unit, and an
    unlabeled bar still carries the stretch and the range.
    """
    assert _bar_label(None) == ""


def test_no_bunit_keeps_the_display_factor():
    """The scaling is a fact about the numbers even when the unit is not."""
    assert _bar_label(None, display_factor=1e3) == "[x1e+03]"


@pytest.mark.parametrize("bunit", ["JY/BEAM", "electron/s", "K"])
def test_bunit_is_still_honored(bunit):
    assert _bar_label(bunit) == bunit


def test_explicit_unit_still_overrides_a_missing_bunit():
    assert _bar_label(None, unit="Jy/beam") == "Jy/beam"


def _info_lines(bunit=None):
    data, hdr = _hdr_with(bunit)
    r = quicklook_plot(data, header=hdr, rms=0.02, colorbar=False,
                       show_info=True)
    lines = []
    for t in r.ax.figure.findobj(plt.Text):
        lines.extend(t.get_text().splitlines())
    plt.close("all")
    return lines


def test_unitless_info_text_has_no_dangling_space():
    """An empty unit must not leave 'Peak = 4.000 ' or a contour '(0.06 )'."""
    lines = _info_lines(None)
    peak = [ln for ln in lines if ln.startswith("Peak =")]
    cont = [ln for ln in lines if ln.startswith("Contours:")]
    assert peak and "  RMS" not in peak[0].replace("    RMS", "")
    assert peak[0].split("    ")[0] == peak[0].split("    ")[0].rstrip()
    assert cont and " )" not in cont[0].split("$")[0]


def test_simpleimageplot_colorbar_label_follows_labelcolor():
    """simpleimageplot styles its colorbar label + tick labels with labelcolor,
    not add_colorbar's black default (which vanishes on a dark facecolor)."""
    import numpy as np
    from matplotlib.colors import to_hex

    import skyplothelper as sph
    res = sph.simpleimageplot(np.random.default_rng(0).random((16, 16)),
                              colorbar=True, labelcolor="cyan", cbar_label="Jy")
    assert to_hex(res.colorbar.ax.yaxis.label.get_color()) == "#00ffff"
    plt.close("all")
