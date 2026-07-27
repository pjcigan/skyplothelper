"""Tests for the :class:`skyplothelper.Beam` class.

Coverage: construction, style switching, semantic-property mutators,
``from_header`` factory, axes wiring (``add_to`` / ``remove``), and the
PA convention (BPA = degrees east of north on an N-up E-left image).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

import skyplothelper as sph  # noqa: E402
from skyplothelper.overlays.beam import (  # noqa: E402
    _BEAM_STYLES,
    Beam,
    BeamStack,
)

# ---- helpers -----------------------------------------------------------------

def _plain_axes():
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    return fig, ax


def _synthetic_header(bmaj_deg=12.0 / 3600.0, bmin_deg=7.0 / 3600.0,
                       bpa_deg=35.0, cdelt_deg=1.0 / 3600.0,
                       naxis=128):
    """Construct a minimal FITS header with BMAJ/BMIN/BPA + WCS."""
    import astropy.io.fits as pyfits

    hdr = pyfits.Header()
    hdr["BMAJ"] = bmaj_deg
    hdr["BMIN"] = bmin_deg
    hdr["BPA"] = bpa_deg
    hdr["CDELT1"] = -cdelt_deg
    hdr["CDELT2"] = cdelt_deg
    hdr["CRPIX1"] = naxis // 2
    hdr["CRPIX2"] = naxis // 2
    hdr["CRVAL1"] = 180.0
    hdr["CRVAL2"] = 0.0
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = naxis
    hdr["NAXIS2"] = naxis
    return hdr


# ---- basic construction ------------------------------------------------------

def test_beam_is_an_ellipse_subclass():
    """``Beam`` should be a drop-in :class:`Ellipse` so all existing
    patch machinery (set_fc / set_ec / path_effects / zorder / ...)
    works without special-casing."""
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10)
    assert isinstance(beam, Ellipse)


def test_default_facecolor_is_transparent():
    """Beams default to an unfilled outline — the canonical
    'beam-marker' look. Caller passes ``fc=...`` to override."""
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10)
    assert beam.get_facecolor()[3] == 0.0  # alpha=0


def test_default_style_is_ellipse():
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10)
    assert beam.style == "ellipse"


def test_semantic_properties_match_underlying_ellipse():
    """``bmaj_pix`` / ``bmin_pix`` / ``bpa_deg`` are aliases for the
    underlying Ellipse's ``height`` / ``width`` / ``angle`` (with
    the astronomical major-along-y convention)."""
    beam = Beam((10, 10), bmaj_pix=20.0, bmin_pix=8.0, bpa_deg=30.0)
    assert beam.bmaj_pix == 20.0
    assert beam.bmin_pix == 8.0
    assert beam.bpa_deg == 30.0
    # And they map to the right Ellipse fields:
    assert beam.height == 20.0      # major along the height axis
    assert beam.width == 8.0
    assert beam.angle == 30.0


# ---- style switching ---------------------------------------------------------

@pytest.mark.parametrize("style", sorted(_BEAM_STYLES))
def test_every_style_constructs(style):
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10, style=style)
    assert beam.style == style


def test_unknown_style_raises():
    with pytest.raises(ValueError, match="style must be one of"):
        Beam((10, 10), bmaj_pix=20, bmin_pix=10, style="lollipop")


def test_crosshair_style_builds_two_lines():
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10, style="crosshair")
    assert len(beam._decorations) == 2
    assert all(isinstance(d, Line2D) for d in beam._decorations)


def test_ellipse_style_has_no_decorations():
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10, style="ellipse")
    assert beam._decorations == []


def test_hatch_style_sets_hatch_pattern():
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10, style="hatch")
    assert beam.get_hatch() == "///"


def test_crosshairgrid_sets_grid_hatch_and_crosshairs():
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10, style="crosshairgrid")
    assert beam.get_hatch() == "++++++"
    assert len(beam._decorations) == 2


def test_explicit_hatch_overrides_plain_style_hatch():
    """An explicit ``hatch=`` must survive the style-driven hatch reset —
    previously ``style='ellipse'`` silently cleared it to None."""
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10, style="ellipse",
                hatch="////")
    assert beam.get_hatch() == "////"
    # ... and it wins over a style's implied pattern too.
    beam2 = Beam((10, 10), bmaj_pix=20, bmin_pix=10, style="hatch",
                 hatch="xx")
    assert beam2.get_hatch() == "xx"
    # No explicit hatch on a plain style stays None (unchanged behavior).
    beam3 = Beam((10, 10), bmaj_pix=20, bmin_pix=10, style="ellipse")
    assert beam3.get_hatch() is None


def test_filled_style_inherits_edgecolor_as_facecolor():
    """``filled`` defaults to ``fc=ec`` so the beam reads as solid
    in its assigned color without the caller having to set ``fc=``
    twice."""
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10,
                style="filled", ec="red")
    fc = beam.get_facecolor()
    ec = beam.get_edgecolor()
    np.testing.assert_allclose(fc[:3], ec[:3])
    assert fc[3] > 0


def test_filled_style_respects_explicit_facecolor():
    """If the caller passes ``fc=`` explicitly, the filled-style
    default doesn't clobber it."""
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10,
                style="filled", fc="blue", ec="red")
    fc = beam.get_facecolor()
    # Blue: rgb ~ (0, 0, 1)
    assert fc[2] > 0.9
    assert fc[0] < 0.1


def test_style_switch_swaps_hatch_pattern():
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10, style="hatch")
    assert beam.get_hatch() == "///"
    beam.set_style("ellipse")
    assert beam.get_hatch() is None
    beam.set_style("crosshairgrid")
    assert beam.get_hatch() == "++++++"


# ---- mutators trigger crosshair rebuild --------------------------------------

def _line_endpoints(line):
    xs, ys = line.get_data()
    return np.array(xs), np.array(ys)


def test_bmaj_setter_rebuilds_crosshair_geometry():
    beam = Beam((50, 50), bmaj_pix=20, bmin_pix=10, bpa_deg=0,
                style="crosshair")
    old_xs, old_ys = _line_endpoints(beam._decorations[0])
    beam.bmaj_pix = 40
    new_xs, new_ys = _line_endpoints(beam._decorations[0])
    # Major line is along +y at bpa=0 — its span doubles when bmaj doubles
    assert (new_ys.max() - new_ys.min()) == pytest.approx(40.0)
    assert (old_ys.max() - old_ys.min()) == pytest.approx(20.0)


def test_bpa_setter_rotates_crosshair_lines():
    beam = Beam((50, 50), bmaj_pix=20, bmin_pix=20, bpa_deg=0,
                style="crosshair")
    # At bpa=0 the major line runs along +y (constant x at center)
    xs0, ys0 = _line_endpoints(beam._decorations[0])
    assert xs0[0] == pytest.approx(50.0)
    assert xs0[1] == pytest.approx(50.0)
    beam.bpa_deg = 90
    # After PA=90° the major axis runs along -x (constant y)
    xs1, ys1 = _line_endpoints(beam._decorations[0])
    assert ys1[0] == pytest.approx(50.0)
    assert ys1[1] == pytest.approx(50.0)


def test_set_center_rebuilds_crosshair_position():
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=20, bpa_deg=0,
                style="crosshair")
    beam.set_center((80, 80))
    xs, ys = _line_endpoints(beam._decorations[0])
    assert xs.mean() == pytest.approx(80.0)
    assert ys.mean() == pytest.approx(80.0)


# ---- axes wiring -------------------------------------------------------------

def test_add_to_attaches_patch_and_crosshair_lines():
    fig, ax = _plain_axes()
    beam = Beam((50, 50), bmaj_pix=20, bmin_pix=10, style="crosshair")
    beam.add_to(ax)
    assert beam in ax.patches
    for d in beam._decorations:
        assert d in ax.lines
    plt.close(fig)


def test_add_to_returns_self_for_chaining():
    fig, ax = _plain_axes()
    beam = Beam((50, 50), bmaj_pix=20, bmin_pix=10)
    ret = beam.add_to(ax)
    assert ret is beam
    plt.close(fig)


def test_style_switch_after_attach_resyncs_crosshair_lines():
    """Changing the style on a beam that's already on an axes must
    add / remove the crosshair child Line2Ds in place."""
    fig, ax = _plain_axes()
    beam = Beam((50, 50), bmaj_pix=20, bmin_pix=10, style="ellipse")
    beam.add_to(ax)
    assert len(ax.lines) == 0
    beam.set_style("crosshair")
    assert len(ax.lines) == 2
    beam.set_style("ellipse")
    assert len(ax.lines) == 0
    plt.close(fig)


def test_geometry_change_after_attach_keeps_crosshairs_attached():
    fig, ax = _plain_axes()
    beam = Beam((50, 50), bmaj_pix=20, bmin_pix=10, style="crosshair")
    beam.add_to(ax)
    n_lines_before = len(ax.lines)
    beam.bmaj_pix = 40
    n_lines_after = len(ax.lines)
    assert n_lines_after == n_lines_before  # still 2 crosshair lines
    plt.close(fig)


def test_remove_takes_crosshair_lines_with_it():
    fig, ax = _plain_axes()
    beam = Beam((50, 50), bmaj_pix=20, bmin_pix=10, style="crosshair")
    beam.add_to(ax)
    beam.remove()
    assert beam not in ax.patches
    assert len(ax.lines) == 0
    plt.close(fig)


# ---- from_header factory -----------------------------------------------------

def test_from_header_reads_bmaj_bmin_bpa_and_converts_to_pixels():
    fig, ax = _plain_axes()
    hdr = _synthetic_header(bmaj_deg=20.0 / 3600.0,
                              bmin_deg=10.0 / 3600.0,
                              bpa_deg=45.0,
                              cdelt_deg=1.0 / 3600.0)
    beam = Beam.from_header(hdr, ax=ax)
    # bmaj=20" / 1"/px = 20 px
    assert beam.bmaj_pix == pytest.approx(20.0)
    assert beam.bmin_pix == pytest.approx(10.0)
    assert beam.bpa_deg == pytest.approx(45.0)
    plt.close(fig)


def test_from_header_accepts_wcs_input():
    """A WCS object should work as input — the factory pulls
    ``to_header()`` internally to read BMAJ/BMIN/BPA + CDELT."""
    from astropy.wcs import WCS

    fig, ax = _plain_axes()
    hdr = _synthetic_header()
    wcs = WCS(hdr)
    # WCS dropped BMAJ/BMIN/BPA when constructed from a header without
    # those non-WCS keys, so re-attach them
    h = wcs.to_header()
    h["BMAJ"] = hdr["BMAJ"]
    h["BMIN"] = hdr["BMIN"]
    h["BPA"] = hdr["BPA"]
    beam = Beam.from_header(h, ax=ax)
    assert beam.bmaj_pix > 0
    plt.close(fig)


def test_from_header_auto_positions_at_lower_left_corner():
    fig, ax = _plain_axes()
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 100)
    hdr = _synthetic_header()
    beam = Beam.from_header(hdr, ax=ax, pad_frac=0.08)
    cx, cy = beam.get_center()
    # 8% in from (xlim[0], ylim[0])
    assert cx == pytest.approx(0 + 0.08 * 200)
    assert cy == pytest.approx(0 + 0.08 * 100)
    plt.close(fig)


def test_from_header_xy_override_wins_over_axes():
    fig, ax = _plain_axes()
    hdr = _synthetic_header()
    beam = Beam.from_header(hdr, ax=ax, xy=(50, 50))
    assert beam.get_center() == (50, 50)
    plt.close(fig)


def test_from_header_without_ax_or_xy_raises():
    hdr = _synthetic_header()
    with pytest.raises(ValueError, match="ax= or xy="):
        Beam.from_header(hdr)


def test_from_header_missing_beam_keys_raises():
    import astropy.io.fits as pyfits

    fig, ax = _plain_axes()
    hdr = pyfits.Header()
    hdr["CDELT1"] = -1.0 / 3600.0
    hdr["CDELT2"] = 1.0 / 3600.0
    with pytest.raises(KeyError, match="BMAJ"):
        Beam.from_header(hdr, ax=ax)
    plt.close(fig)


def test_from_header_forwards_style_and_patch_kwargs():
    fig, ax = _plain_axes()
    hdr = _synthetic_header()
    beam = Beam.from_header(hdr, ax=ax,
                              style="crosshair", ec="C0", lw=1.2)
    assert beam.style == "crosshair"
    assert beam.get_linewidth() == 1.2
    plt.close(fig)


# ---- pa_convention -----------------------------------------------------------

def test_pa_convention_fits_is_default():
    """Default convention is FITS (degrees east of north). The stored
    ``bpa_deg`` equals the input."""
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10, bpa_deg=42)
    assert beam.bpa_deg == 42.0


def test_pa_convention_plot_shifts_by_90():
    """A plot-convention input is shifted by -90° internally so that
    'major axis at θ° CCW from +x' is preserved when round-tripping
    through the Ellipse machinery (Ellipse height=bmaj at angle=0 is
    along +y, i.e. 90° CCW from +x)."""
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10,
                bpa_deg=0, pa_convention='plot')
    # plot=0 → major along +x → FITS BPA = -90 (i.e. major along +x
    # is at -90° east of north when north is +y)
    assert beam.bpa_deg == pytest.approx(-90.0)
    assert beam.bpa_plot == pytest.approx(0.0)


def test_bpa_plot_round_trips_through_setter():
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10)
    beam.bpa_plot = 45.0
    assert beam.bpa_plot == pytest.approx(45.0)
    assert beam.bpa_deg == pytest.approx(45.0 - 90.0)


def test_pa_convention_aliases():
    """``astro`` and ``iau`` are aliases for ``fits``."""
    for alias in ('astro', 'iau', 'FITS', 'Astro'):
        b = Beam((0, 0), 1, 1, bpa_deg=30, pa_convention=alias)
        assert b.bpa_deg == 30.0


def test_unknown_pa_convention_raises():
    with pytest.raises(ValueError, match="pa_convention"):
        Beam((10, 10), bmaj_pix=20, bmin_pix=10,
             pa_convention="galactic")


# ---- from_arcsec / set_size_arcsec ------------------------------------------

def test_from_arcsec_converts_via_pixscale():
    fig, ax = _plain_axes()
    beam = Beam.from_arcsec(bmaj_asec=20, bmin_asec=10, bpa_deg=15,
                              pixscale_asec=0.5, ax=ax)
    assert beam.bmaj_pix == pytest.approx(40)   # 20" / 0.5"/px
    assert beam.bmin_pix == pytest.approx(20)
    assert beam.bpa_deg == pytest.approx(15)
    plt.close(fig)


def test_from_arcsec_requires_positive_pixscale():
    with pytest.raises(ValueError, match="pixscale_asec"):
        Beam.from_arcsec(20, 10, pixscale_asec=0, xy=(0, 0))
    with pytest.raises(ValueError, match="pixscale_asec"):
        Beam.from_arcsec(20, 10, pixscale_asec=-0.5, xy=(0, 0))


def test_from_arcsec_auto_positions_with_ax():
    fig, ax = _plain_axes()
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 100)
    beam = Beam.from_arcsec(20, 10, pixscale_asec=0.5, ax=ax,
                              pad_frac=0.10)
    cx, cy = beam.get_center()
    assert cx == pytest.approx(20.0)
    assert cy == pytest.approx(10.0)
    plt.close(fig)


def test_from_arcsec_without_ax_or_xy_raises():
    with pytest.raises(ValueError, match="ax= or xy="):
        Beam.from_arcsec(20, 10, pixscale_asec=0.5)


def test_set_size_arcsec_updates_pix_sizes():
    beam = Beam((0, 0), bmaj_pix=10, bmin_pix=5)
    ret = beam.set_size_arcsec(bmaj_asec=30, bmin_asec=15,
                                 pixscale_asec=0.5)
    assert ret is beam
    assert beam.bmaj_pix == pytest.approx(60)
    assert beam.bmin_pix == pytest.approx(30)


def test_set_size_arcsec_requires_positive_pixscale():
    beam = Beam((0, 0), 10, 5)
    with pytest.raises(ValueError, match="pixscale_asec"):
        beam.set_size_arcsec(20, 10, pixscale_asec=0)


# ---- crosshair styling -------------------------------------------------------

def _color_eq(a, b):
    """RGBA tuple equality with tolerance."""
    import matplotlib.colors as mcolors
    return np.allclose(mcolors.to_rgba(a), mcolors.to_rgba(b))


def test_crosshair_color_kwarg_overrides_ec_for_crosshair_lines():
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10, style="crosshair",
                ec="black", crosshair_color="red")
    line_color = beam._decorations[0].get_color()
    assert _color_eq(line_color, "red")


def test_crosshair_color_none_inherits_from_edgecolor():
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10, style="crosshair",
                ec="blue")
    line_color = beam._decorations[0].get_color()
    assert _color_eq(line_color, "blue")


def test_crosshair_lw_kwarg_overrides_scaled_default():
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10, style="crosshair",
                lw=2.0, crosshair_lw=0.5)
    assert beam._decorations[0].get_linewidth() == 0.5


def test_crosshair_lw_default_uses_lw_scale():
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10, style="crosshair",
                lw=2.0, crosshair_lw_scale=0.5)
    assert beam._decorations[0].get_linewidth() == 1.0


def test_crosshair_ls_kwarg_propagates_to_lines():
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10, style="crosshair",
                crosshair_ls="--")
    assert beam._decorations[0].get_linestyle() == "--"


def test_crosshair_length_scale_changes_line_length():
    """``length_scale=0.25`` should halve each crosshair line's
    span from the full-FWHM default."""
    beam = Beam((50, 50), bmaj_pix=40, bmin_pix=20, bpa_deg=0,
                style="crosshair")
    _, ys = _line_endpoints(beam._decorations[0])
    full_span = ys.max() - ys.min()
    assert full_span == pytest.approx(40.0)
    beam.set_crosshair(length_scale=0.25)
    _, ys = _line_endpoints(beam._decorations[0])
    short_span = ys.max() - ys.min()
    assert short_span == pytest.approx(20.0)


def test_set_crosshair_returns_self_for_chaining():
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10, style="crosshair")
    assert beam.set_crosshair(color="red", lw=2) is beam


def test_set_crosshair_keeps_unspecified_attrs():
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10, style="crosshair",
                crosshair_color="red", crosshair_lw=1.5,
                crosshair_ls="--")
    beam.set_crosshair(color="blue")  # only color changes
    line = beam._decorations[0]
    assert _color_eq(line.get_color(), "blue")
    assert line.get_linewidth() == 1.5      # unchanged
    assert line.get_linestyle() == "--"     # unchanged


def test_crosshair_lines_property_returns_decorations():
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10, style="crosshair")
    lines = beam.crosshair_lines
    assert len(lines) == 2
    assert all(isinstance(line, Line2D) for line in lines)


def test_crosshair_lines_empty_for_non_crosshair_style():
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10, style="ellipse")
    assert beam.crosshair_lines == []


# ---- grid customization ------------------------------------------------------

def test_grid_marker_and_density_construct_hatch_string():
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10,
                style="crosshairgrid",
                grid_marker="x", grid_density=4)
    assert beam.get_hatch() == "xxxx"


def test_default_grid_is_plus_six():
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10, style="crosshairgrid")
    assert beam.get_hatch() == "++++++"


def test_grid_marker_setter_rebuilds_hatch():
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10,
                style="filledgrid", grid_marker="+", grid_density=3)
    assert beam.get_hatch() == "+++"
    beam.grid_marker = "o"
    assert beam.get_hatch() == "ooo"


def test_grid_density_setter_rebuilds_hatch():
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10,
                style="crosshairgrid",
                grid_marker="+", grid_density=2)
    assert beam.get_hatch() == "++"
    beam.grid_density = 8
    assert beam.get_hatch() == "++++++++"


def test_set_grid_returns_self_for_chaining():
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10, style="crosshairgrid")
    assert beam.set_grid(marker="x", density=3) is beam


def test_grid_changes_invisible_for_non_grid_style():
    """Changing ``grid_marker`` / ``grid_density`` on a non-grid style
    doesn't sneak a hatch onto the patch (style controls whether the
    hatch is applied at all)."""
    beam = Beam((10, 10), bmaj_pix=20, bmin_pix=10, style="ellipse")
    beam.set_grid(marker="x", density=8)
    assert beam.get_hatch() is None
    # But once the style switches to a grid one, the new settings take
    # over.
    beam.set_style("crosshairgrid")
    assert beam.get_hatch() == "xxxxxxxx"


# ---- BeamStack ---------------------------------------------------------------

def _three_beams():
    return [
        Beam((50, 50), bmaj_pix=40, bmin_pix=30, bpa_deg=20,
             style="ellipse", ec="C3", lw=1.5, label="7 m"),
        Beam((50, 50), bmaj_pix=24, bmin_pix=16, bpa_deg=20,
             style="ellipse", ec="C0", lw=1.5, label="12 m"),
        Beam((50, 50), bmaj_pix=10, bmin_pix=7, bpa_deg=20,
             style="filled", ec="C2", label="combined"),
    ]


def test_beamstack_construction_holds_member_beams():
    beams = _three_beams()
    stack = BeamStack(beams)
    assert len(stack) == 3
    for orig, member in zip(beams, stack):
        assert member is orig


def test_beamstack_rejects_non_beam_inputs():
    with pytest.raises(TypeError, match="Beam instances"):
        BeamStack([Beam((0, 0), 10, 5), "not a beam"])


def test_beamstack_indexing_and_iteration():
    stack = BeamStack(_three_beams())
    assert stack[0].get_edgecolor()[0] > 0   # C3 has red component
    seen = list(stack)
    assert len(seen) == 3
    assert seen == stack.beams       # iteration order matches .beams


def test_beamstack_beams_property_returns_copy():
    """The ``beams`` property should hand back a copy so external
    mutation can't quietly reorder / drop members of the stack."""
    stack = BeamStack(_three_beams())
    sweep = stack.beams
    sweep.pop()
    assert len(stack) == 3       # internal list untouched


def test_beamstack_add_to_attaches_every_member_in_order():
    fig, ax = _plain_axes()
    stack = BeamStack(_three_beams())
    ret = stack.add_to(ax)
    assert ret is stack
    for b in stack:
        assert b in ax.patches
    # Patches stored in add order — last-added paints on top.
    assert ax.patches.index(stack[0]) < ax.patches.index(stack[-1])
    plt.close(fig)


def test_beamstack_add_to_includes_crosshair_lines():
    fig, ax = _plain_axes()
    stack = BeamStack([
        Beam((50, 50), 20, 10, style="crosshair"),
        Beam((50, 50), 30, 15, style="crosshairgrid"),
    ])
    stack.add_to(ax)
    # 2 crosshair members × 2 lines each = 4 lines on the axes
    assert len(ax.lines) == 4
    plt.close(fig)


def test_beamstack_remove_strips_every_member():
    fig, ax = _plain_axes()
    stack = BeamStack(_three_beams())
    stack.add_to(ax)
    stack.remove()
    for b in stack:
        assert b not in ax.patches
    plt.close(fig)


def test_beamstack_remove_is_safe_if_never_attached():
    """remove() on an unattached stack shouldn't raise."""
    stack = BeamStack(_three_beams())
    stack.remove()  # no-op — no axes


def test_beamstack_set_center_moves_every_member():
    stack = BeamStack(_three_beams())
    stack.set_center((10, 80))
    for b in stack:
        assert b.get_center() == (10, 80)


def test_beamstack_set_visible_propagates_to_members():
    """set_visible(False) should hide every member beam AND every
    crosshair child of every member beam."""
    fig, ax = _plain_axes()
    stack = BeamStack([
        Beam((50, 50), 20, 10, style="crosshair"),
        Beam((50, 50), 30, 15, style="ellipse"),
    ])
    stack.add_to(ax)
    stack.set_visible(False)
    for b in stack:
        assert not b.get_visible()
        for line in b.crosshair_lines:
            assert not line.get_visible()
    stack.set_visible(True)
    for b in stack:
        assert b.get_visible()
    plt.close(fig)


# ---- BeamStack.from_specs ---------------------------------------------------

def test_beamstack_from_specs_builds_n_beams_at_shared_xy():
    specs = [
        dict(bmaj_pix=40, bmin_pix=30, bpa_deg=20, ec="C3", label="7 m"),
        dict(bmaj_pix=24, bmin_pix=16, bpa_deg=20, ec="C0", label="12 m"),
        dict(bmaj_pix=10, bmin_pix=7, bpa_deg=20, ec="C2",
             style="filled", label="combined"),
    ]
    stack = BeamStack.from_specs(specs, xy=(50, 50))
    assert len(stack) == 3
    for b in stack:
        assert b.get_center() == (50, 50)


def test_beamstack_from_specs_shared_kwargs_apply_to_every_member():
    stack = BeamStack.from_specs(
        [dict(bmaj_pix=20, bmin_pix=10, ec="C0"),
         dict(bmaj_pix=30, bmin_pix=20, ec="C3")],
        xy=(50, 50),
        style="crosshair", lw=2.0)
    for b in stack:
        assert b.style == "crosshair"
        assert b.get_linewidth() == 2.0


def test_beamstack_from_specs_per_member_wins_on_collision():
    """When a key appears both in shared_kwargs and a per-member
    spec, the per-member value wins."""
    stack = BeamStack.from_specs(
        [dict(bmaj_pix=20, bmin_pix=10, style="filled"),
         dict(bmaj_pix=30, bmin_pix=20)],     # inherits style
        xy=(0, 0), style="ellipse")
    assert stack[0].style == "filled"        # per-member override
    assert stack[1].style == "ellipse"       # shared default


def test_beamstack_from_specs_auto_positions_with_ax():
    fig, ax = _plain_axes()
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 100)
    stack = BeamStack.from_specs(
        [dict(bmaj_pix=20, bmin_pix=10),
         dict(bmaj_pix=30, bmin_pix=20)],
        ax=ax, pad_frac=0.10)
    for b in stack:
        assert b.get_center() == (20.0, 10.0)
    plt.close(fig)


def test_beamstack_from_specs_without_ax_or_xy_raises():
    with pytest.raises(ValueError, match="ax= or xy="):
        BeamStack.from_specs([dict(bmaj_pix=20, bmin_pix=10)])


# ---- public API surface ------------------------------------------------------

def test_beam_exported_at_package_top_level():
    assert sph.Beam is Beam


def test_beamstack_exported_at_package_top_level():
    assert sph.BeamStack is BeamStack


# ---- add_anchored ------------------------------------------------------------

def test_add_anchored_returns_anchored_offsetbox():
    from matplotlib.offsetbox import AnchoredOffsetbox

    fig, ax = _plain_axes()
    beam = Beam((0, 0), bmaj_pix=12, bmin_pix=6, bpa_deg=20)
    anchored = beam.add_anchored(ax, loc="lower left")
    assert isinstance(anchored, AnchoredOffsetbox)
    plt.close(fig)


def test_add_anchored_adds_artist_to_axes():
    fig, ax = _plain_axes()
    beam = Beam((0, 0), bmaj_pix=12, bmin_pix=6, bpa_deg=20)
    anchored = beam.add_anchored(ax)
    assert anchored in ax.artists or anchored in ax.get_children()
    plt.close(fig)


def test_add_anchored_wraps_beam_in_aux_transform_box():
    """The anchored artist should hold a child that contains the
    beam — confirms the beam is reachable for downstream tweaks."""
    fig, ax = _plain_axes()
    beam = Beam((0, 0), bmaj_pix=12, bmin_pix=6, bpa_deg=20,
                style="crosshair")
    anchored = beam.add_anchored(ax)
    aux = anchored.get_child()
    children = list(aux.get_children())
    assert beam in children
    # Crosshair child lines accompany the beam in the box
    n_lines = sum(1 for c in children if isinstance(c, Line2D))
    assert n_lines == 2  # crosshair style → 2 lines
    plt.close(fig)


def test_add_anchored_passes_loc_to_offsetbox():
    """Spot-check that the loc kwarg propagates to the wrapper."""
    fig, ax = _plain_axes()
    for loc in ("lower left", "lower right",
                "upper right", "upper left"):
        beam = Beam((0, 0), bmaj_pix=10, bmin_pix=6, bpa_deg=0)
        anchored = beam.add_anchored(ax, loc=loc)
        # AnchoredOffsetbox exposes loc as a property or attr;
        # matplotlib stores it as ``self.loc`` (int or str)
        assert anchored.loc is not None
    plt.close(fig)


def test_add_anchored_borderpad_propagates():
    fig, ax = _plain_axes()
    beam = Beam((0, 0), bmaj_pix=10, bmin_pix=6)
    anchored = beam.add_anchored(ax, borderpad=2.5)
    assert anchored.borderpad == 2.5
    plt.close(fig)


def test_add_anchored_forwards_offsetbox_kwargs():
    """Extra kwargs (e.g. ``frameon``) should flow to AnchoredOffsetbox."""
    fig, ax = _plain_axes()
    beam = Beam((0, 0), bmaj_pix=10, bmin_pix=6)
    anchored = beam.add_anchored(ax, frameon=True)
    # If frameon=True, the patch should be visible
    assert anchored.patch.get_visible() is True
    plt.close(fig)


def test_add_anchored_works_with_filled_style():
    """Filled-style beams should anchor too — confirms the hatch +
    facecolor pass through the AuxTransformBox."""
    fig, ax = _plain_axes()
    beam = Beam((0, 0), bmaj_pix=10, bmin_pix=6, bpa_deg=10,
                style="filled", ec="C2")
    anchored = beam.add_anchored(ax, loc="upper right")
    aux = anchored.get_child()
    children = list(aux.get_children())
    assert beam in children
    # No crosshair lines on filled style
    assert all(not isinstance(c, Line2D) for c in children)
    plt.close(fig)


# ---- from_psf_fit ------------------------------------------------------------

def _synthetic_psf(nx=65, ny=65, fwhm_major=12.0, fwhm_minor=6.0,
                    pa_plot_deg=30.0, noise=0.0, seed=0):
    """Synthetic elliptical Gaussian on a square grid.

    ``pa_plot_deg`` is the orientation in matplotlib's CCW-from-+x
    convention of the *frame* the gaussian is sampled in — i.e. the
    rotation applied to the underlying coords before stamping. The
    actual major-axis orientation in the rendered image is
    ``pa_plot_deg + 90`` (since the larger sigma is along +y at PA=0
    by construction here).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    sig_x = fwhm_minor / 2.3548
    sig_y = fwhm_major / 2.3548
    th = np.radians(pa_plot_deg)
    c, s = np.cos(th), np.sin(th)
    xr = (xx - cx) * c + (yy - cy) * s
    yr = -(xx - cx) * s + (yy - cy) * c
    psf = np.exp(-0.5 * ((xr / sig_x) ** 2 + (yr / sig_y) ** 2))
    if noise > 0:
        psf = psf + noise * rng.standard_normal(psf.shape)
    return psf


def test_from_psf_fit_recovers_fwhm():
    psf = _synthetic_psf(nx=65, ny=65,
                          fwhm_major=12.0, fwhm_minor=6.0,
                          pa_plot_deg=30.0)
    beam = Beam.from_psf_fit(psf, xy=(0, 0))
    assert beam.bmaj_pix == pytest.approx(12.0, abs=0.5)
    assert beam.bmin_pix == pytest.approx(6.0, abs=0.5)


def test_from_psf_fit_recovers_orientation():
    """Synthetic frame-rotation by 30° plus the gaussian's
    major-along-+y convention means the actual on-image major axis is
    at 120° (== -60° in the [-90, 90] folded convention)."""
    psf = _synthetic_psf(nx=65, ny=65,
                          fwhm_major=12.0, fwhm_minor=6.0,
                          pa_plot_deg=30.0)
    beam = Beam.from_psf_fit(psf, xy=(0, 0))
    # PA expressed in (-90, 90]; equivalent to 120° (line orientation
    # is ambiguous by 180°)
    folded = ((beam.bpa_plot + 90.0) % 180.0) - 90.0
    assert folded == pytest.approx(-60.0, abs=2.0)


def test_from_psf_fit_handles_noise():
    psf = _synthetic_psf(nx=65, ny=65,
                          fwhm_major=10.0, fwhm_minor=8.0,
                          pa_plot_deg=0.0, noise=0.02, seed=1)
    beam = Beam.from_psf_fit(psf, xy=(0, 0))
    # Modest tolerance — noisy data
    assert beam.bmaj_pix == pytest.approx(10.0, abs=1.0)
    assert beam.bmin_pix == pytest.approx(8.0, abs=1.0)


def test_from_psf_fit_masks_nans():
    """NaN entries in the PSF should be excluded from the fit, not
    propagate to the fitted parameters."""
    psf = _synthetic_psf(nx=51, ny=51,
                          fwhm_major=10.0, fwhm_minor=6.0,
                          pa_plot_deg=0.0)
    # Punch out a corner with NaN
    psf[0:5, 0:5] = np.nan
    beam = Beam.from_psf_fit(psf, xy=(0, 0))
    assert beam.bmaj_pix == pytest.approx(10.0, abs=0.5)


def test_from_psf_fit_rejects_non_2d_input():
    with pytest.raises(ValueError, match="psf_image must be 2D"):
        Beam.from_psf_fit(np.zeros((3, 3, 3)), xy=(0, 0))


def test_from_psf_fit_raises_on_all_nan_input():
    psf = np.full((20, 20), np.nan)
    with pytest.raises(RuntimeError, match="non-finite"):
        Beam.from_psf_fit(psf, xy=(0, 0))


def test_from_psf_fit_uses_ax_for_auto_position():
    fig, ax = _plain_axes()
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 100)
    psf = _synthetic_psf(nx=31, ny=31, fwhm_major=8, fwhm_minor=5)
    beam = Beam.from_psf_fit(psf, ax=ax, pad_frac=0.10)
    cx, cy = beam.get_center()
    assert cx == pytest.approx(20.0)
    assert cy == pytest.approx(10.0)
    plt.close(fig)


def test_from_psf_fit_without_ax_or_xy_raises():
    psf = _synthetic_psf()
    with pytest.raises(ValueError, match="ax= or xy="):
        Beam.from_psf_fit(psf)


def test_from_psf_fit_forwards_style_and_patch_kwargs():
    psf = _synthetic_psf(nx=31, ny=31, fwhm_major=8, fwhm_minor=5)
    beam = Beam.from_psf_fit(psf, xy=(0, 0),
                              style="crosshair", ec="C1", lw=1.5)
    assert beam.style == "crosshair"
    assert beam.get_linewidth() == 1.5


# ---- add_psf_inset -----------------------------------------------------------

def _parent_axes_with_psf():
    """Return (fig, ax, psf) for an inset-test parent figure."""
    fig, ax = plt.subplots(figsize=(5, 5))
    psf = _synthetic_psf(nx=51, ny=51, fwhm_major=10, fwhm_minor=6,
                          pa_plot_deg=20.0)
    ax.imshow(psf, cmap="gray_r", origin="lower")
    return fig, ax, psf


def test_add_psf_inset_returns_an_inset_axes():
    fig, ax, psf = _parent_axes_with_psf()
    beam = Beam((25, 25), bmaj_pix=10, bmin_pix=6, bpa_deg=20)
    iax = beam.add_psf_inset(ax, psf)
    # Inset axes is added to the figure
    assert iax in fig.axes
    # And it's a different axes from the parent
    assert iax is not ax
    plt.close(fig)


def test_add_psf_inset_renders_an_image():
    fig, ax, psf = _parent_axes_with_psf()
    beam = Beam((25, 25), bmaj_pix=10, bmin_pix=6, bpa_deg=20)
    iax = beam.add_psf_inset(ax, psf)
    # imshow puts the result in ax.images
    assert len(iax.images) == 1
    plt.close(fig)


def test_add_psf_inset_overlays_a_beam_by_default():
    fig, ax, psf = _parent_axes_with_psf()
    beam = Beam((25, 25), bmaj_pix=10, bmin_pix=6, bpa_deg=20)
    iax = beam.add_psf_inset(ax, psf)
    # One Ellipse patch (a Beam) on the inset
    ellipses = [p for p in iax.patches if hasattr(p, "bmaj_pix")]
    assert len(ellipses) == 1
    assert ellipses[0].bmaj_pix == pytest.approx(10.0)
    assert ellipses[0].bmin_pix == pytest.approx(6.0)
    plt.close(fig)


def test_add_psf_inset_show_beam_false_skips_overlay():
    fig, ax, psf = _parent_axes_with_psf()
    beam = Beam((25, 25), bmaj_pix=10, bmin_pix=6, bpa_deg=20)
    iax = beam.add_psf_inset(ax, psf, show_beam=False)
    ellipses = [p for p in iax.patches if hasattr(p, "bmaj_pix")]
    assert ellipses == []
    plt.close(fig)


def test_add_psf_inset_beam_kwargs_propagate_to_overlay():
    fig, ax, psf = _parent_axes_with_psf()
    beam = Beam((25, 25), bmaj_pix=10, bmin_pix=6, bpa_deg=20)
    iax = beam.add_psf_inset(
        ax, psf,
        beam_kwargs={'style': 'crosshair', 'ec': 'red', 'lw': 2.5},
    )
    [overlay] = [p for p in iax.patches if hasattr(p, "bmaj_pix")]
    assert overlay.style == "crosshair"
    assert overlay.get_linewidth() == 2.5


def test_add_psf_inset_title_appears():
    fig, ax, psf = _parent_axes_with_psf()
    beam = Beam((25, 25), bmaj_pix=10, bmin_pix=6, bpa_deg=20)
    iax = beam.add_psf_inset(ax, psf, title="dirty beam")
    assert iax.get_title() == "dirty beam"
    plt.close(fig)


def test_add_psf_inset_rejects_non_2d_input():
    fig, ax, psf = _parent_axes_with_psf()
    beam = Beam((25, 25), bmaj_pix=10, bmin_pix=6, bpa_deg=20)
    with pytest.raises(ValueError, match="psf_image must be 2D"):
        beam.add_psf_inset(ax, np.zeros((3, 3, 3)))
    plt.close(fig)


def test_add_psf_inset_accepts_tuple_size():
    fig, ax, psf = _parent_axes_with_psf()
    beam = Beam((25, 25), bmaj_pix=10, bmin_pix=6, bpa_deg=20)
    iax = beam.add_psf_inset(ax, psf, size=("30%", "20%"))
    assert iax in fig.axes
    plt.close(fig)


def test_add_psf_inset_rejects_invalid_size():
    fig, ax, psf = _parent_axes_with_psf()
    beam = Beam((25, 25), bmaj_pix=10, bmin_pix=6, bpa_deg=20)
    with pytest.raises(ValueError, match="size must be"):
        beam.add_psf_inset(ax, psf, size=42)
    plt.close(fig)


def test_add_psf_inset_borderless():
    fig, ax, psf = _parent_axes_with_psf()
    beam = Beam((25, 25), bmaj_pix=10, bmin_pix=6, bpa_deg=20)
    iax = beam.add_psf_inset(ax, psf, border=False)
    for spine in iax.spines.values():
        assert not spine.get_visible()
    plt.close(fig)


def test_beam_resolve_xy_accepts_skycoord():
    """Beam anchors in DATA coords, so a SkyCoord is projected through the WCS
    — matching the anchoring Reticle and Ruler already support."""
    import matplotlib.pyplot as plt
    from astropy.coordinates import SkyCoord

    import skyplothelper as sph
    from skyplothelper.overlays.beam import Beam
    fig, ax = sph.offset_figure(center=(83.6, 22.0), fov_deg=0.2)
    px, py = Beam._resolve_xy(SkyCoord(83.6, 22.0, unit="deg"), ax, 0.08)
    # centered coord -> center pixel of the frame
    assert 200 < px < 300 and 200 < py < 300
    # a plain tuple is passed through untouched
    assert Beam._resolve_xy((10.0, 20.0), ax, 0.08) == (10.0, 20.0)
    plt.close(fig)


def test_beam_skycoord_without_axes_raises():
    import pytest
    from astropy.coordinates import SkyCoord

    from skyplothelper.overlays.beam import Beam
    with pytest.raises(ValueError):
        Beam._resolve_xy(SkyCoord(83.6, 22.0, unit="deg"), None, 0.08)


# ---- add_beam: the one-call form ----

def _beam_fig():
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    hdr = fits.Header({
        "NAXIS": 2, "NAXIS1": 100, "NAXIS2": 100,
        "CTYPE1": "RA---TAN", "CRVAL1": 180.0, "CRPIX1": 50.0,
        "CDELT1": -1.0 / 3600, "CUNIT1": "deg",
        "CTYPE2": "DEC--TAN", "CRVAL2": 0.0, "CRPIX2": 50.0,
        "CDELT2": 1.0 / 3600, "CUNIT2": "deg",
        "BMAJ": 5.0 / 3600, "BMIN": 3.0 / 3600, "BPA": 30.0})
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=WCS(hdr))
    ax.imshow(np.zeros((100, 100)), origin="lower")
    return fig, ax, hdr


def test_add_beam_draws_in_one_call():
    """Regression: Beam.from_header(hdr, ax=ax) POSITIONS but does not draw,
    which reads like a complete call and silently renders nothing. add_beam is
    the one-call form (mirroring add_reticle for Reticle)."""
    import matplotlib.pyplot as plt

    import skyplothelper as sph
    fig, ax, hdr = _beam_fig()
    n_before = len(ax.patches)
    beam = sph.add_beam(ax, hdr)
    assert len(ax.patches) == n_before + 1
    assert beam in list(ax.patches)
    plt.close(fig)


def test_add_beam_keeps_style_decorations():
    """add_to() attaches the crosshair child lines; a bare ax.add_patch()
    silently drops them, which is why add_beam must not use add_patch."""
    import matplotlib.pyplot as plt

    import skyplothelper as sph
    fig, ax, hdr = _beam_fig()
    sph.add_beam(ax, hdr, style="crosshair")
    assert len(ax.lines) >= 2
    plt.close(fig)


def test_add_beam_anchored():
    import matplotlib.pyplot as plt

    import skyplothelper as sph
    fig, ax, hdr = _beam_fig()
    assert sph.add_beam(ax, hdr, anchored=True) is not None
    plt.close(fig)


def test_add_beam_requires_header():
    import pytest

    import skyplothelper as sph
    fig, ax, hdr = _beam_fig()
    with pytest.raises(TypeError, match="header"):
        sph.add_beam(ax)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_beam_color_kwarg_does_not_warn():
    """Beam always setdefault'd facecolor, so passing color= tripped
    matplotlib's 'color will override edgecolor/facecolor' UserWarning on every
    beam built that way."""
    import warnings

    import matplotlib.pyplot as plt

    import skyplothelper as sph
    fig, ax, hdr = _beam_fig()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sph.add_beam(ax, hdr, color="red")
    assert not [w for w in caught
                if "override the edgecolor" in str(w.message)]
    plt.close(fig)
