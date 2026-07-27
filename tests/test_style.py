"""Tests for the style layers: themes, cycle palettes, the WCSAxes
styling bridge, annotation palettes, and the composable entry point.

These go beyond smoke tests — they assert the helpers actually mutate
the target properties (tick color/size, frame color, prop_cycle,
backgrounds), since the whole point of ``style_wcs_axes`` is to reach
properties that WCSAxes hides from the rcParams machinery.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pytest
from matplotlib import rcParams

import skyplothelper as sph
from skyplothelper import make_wcs_frame
from skyplothelper._compat import coord_ticks
from skyplothelper.style import _THEMES


@pytest.fixture(autouse=True)
def _restore_rcparams():
    """Snapshot rcParams around every test so style mutations don't leak."""
    saved = rcParams.copy()
    try:
        yield
    finally:
        rcParams.update(saved)
        plt.close("all")


def _is_color(c) -> bool:
    try:
        mcolors.to_rgba(c)
        return True
    except (ValueError, TypeError):
        return False


# ---- themes ----------------------------------------------------------

def test_twilight_replaces_dark():
    # The generic mpl dark theme was renamed to 'twilight'; the old key is
    # gone (pre-release, no alias).
    assert "twilight" in _THEMES
    assert "dark" not in _THEMES
    assert "dark_sky" in _THEMES


def test_set_theme_twilight_applies_background():
    sph.set_theme("twilight")
    assert mcolors.same_color(rcParams["figure.facecolor"], "#0f0f23")


# ---- cycle palettes --------------------------------------------------

def test_palettes_well_formed():
    for name, spec in sph.CYCLE_PALETTES.items():
        assert spec["mode"] in ("dual", "light", "dark"), name
        colors = spec["colors"]
        assert len(colors) >= 5, name
        assert all(_is_color(c) for c in colors), name


def test_set_palette_named_sets_prop_cycle():
    sph.set_palette("speakeasy")
    cycle = rcParams["axes.prop_cycle"].by_key()["color"]
    assert cycle == sph.CYCLE_PALETTES["speakeasy"]["colors"]


def test_set_palette_explicit_list():
    custom = ["#264653", "#2A9D8F", "#E9C46A"]
    sph.set_palette(custom)
    assert rcParams["axes.prop_cycle"].by_key()["color"] == custom


def test_set_palette_unknown_raises():
    with pytest.raises(ValueError, match="Unknown palette"):
        sph.set_palette("not_a_palette")


# ---- font layer (set_font / FONT_PRESETS) ----------------------------

def test_font_presets_well_formed():
    valid_kinds = {"serif", "sans", "display", "handwriting", "mono"}
    tier1 = {"DejaVu Serif", "DejaVu Sans", "DejaVu Sans Mono", "monospace"}
    for name, p in sph.FONT_PRESETS.items():
        assert set(p) >= {"stack", "kind", "math"}, name
        assert p["kind"] in valid_kinds, name
        assert isinstance(p["stack"], list) and len(p["stack"]) >= 2, name
        # every stack must END in a guaranteed tier-1 face (graceful degrade)
        assert p["stack"][-1] in tier1, f"{name}: {p['stack'][-1]}"


def test_set_font_preset_sets_stack_and_pairs_math():
    # serif preset -> serif family + cm math, stack ends in DejaVu Serif
    sph.set_font("journal")
    assert rcParams["font.family"] == ["serif"]
    assert rcParams["font.serif"][0] == "TeX Gyre Termes"
    assert rcParams["font.serif"][-1] == "DejaVu Serif"
    assert rcParams["mathtext.fontset"] == "cm"
    # sans preset -> sans-serif family + stixsans math
    sph.set_font("web")
    assert rcParams["font.family"] == ["sans-serif"]
    assert rcParams["font.sans-serif"][-1] == "DejaVu Sans"
    assert rcParams["mathtext.fontset"] == "stixsans"


def test_set_font_appends_tier1_fallback_to_explicit_stack():
    sph.set_font(["EB Garamond", "TeX Gyre Termes"])
    assert rcParams["font.serif"][-1] == "DejaVu Serif"
    assert rcParams["font.serif"][:2] == ["EB Garamond", "TeX Gyre Termes"]


def test_set_font_explicit_math_and_none():
    sph.set_font("journal", math="stixsans")     # explicit override
    assert rcParams["mathtext.fontset"] == "stixsans"
    rcParams["mathtext.fontset"] = "stix"
    sph.set_font("web", math=None)                # leave mathtext untouched
    assert rcParams["mathtext.fontset"] == "stix"


def test_set_font_poster_aliases_talk():
    sph.set_font("poster")
    assert rcParams["font.serif"][0] == "TeX Gyre Pagella"


def test_set_font_always_sets_monospace_stack():
    sph.set_font("journal")
    assert rcParams["font.monospace"] == list(sph.MONO_STACK)
    sph.set_font("web", mono=["Fira Code", "monospace"])
    assert rcParams["font.monospace"] == ["Fira Code", "monospace"]


def test_set_font_register_path_adds_font():
    import matplotlib.font_manager as fm
    fdir = (Path(__file__).resolve().parents[1]
            / "hidden" / "scratchpad" / "fonts")
    ttf = fdir / "patrickhand.ttf"
    if not ttf.exists():
        pytest.skip("font asset not available")
    sph.set_font("sketch", register=str(ttf))
    assert "Patrick Hand" in {f.name for f in fm.fontManager.ttflist}
    assert rcParams["mathtext.fontset"] == "dejavusans"


def test_set_font_unknown_warns_and_leaves_family_unchanged():
    sph.set_font("journal")
    before = list(rcParams["font.serif"])
    with pytest.warns(UserWarning, match="unknown preset"):
        sph.set_font("jurnal")          # typo, not an installed face
    assert list(rcParams["font.serif"]) == before


def test_set_font_default_restores_mpl_fonts():
    sph.set_font("journal")
    sph.set_font("default")
    assert rcParams["font.family"] == plt.rcParamsDefault["font.family"]
    assert rcParams["mathtext.fontset"] == plt.rcParamsDefault["mathtext.fontset"]


def test_set_font_restored_by_style_context():
    sph.set_font("default")
    fam0 = list(rcParams["font.family"])
    math0 = rcParams["mathtext.fontset"]
    with sph.style_context(base="standard", font="journal"):
        assert rcParams["mathtext.fontset"] == "cm"
        assert rcParams["font.serif"][0] == "TeX Gyre Termes"
    assert list(rcParams["font.family"]) == fam0
    assert rcParams["mathtext.fontset"] == math0


def test_set_style_font_layer_applies():
    sph.set_style(base="structural", font="web")
    assert rcParams["mathtext.fontset"] == "stixsans"
    assert rcParams["font.family"] == ["sans-serif"]


def test_structural_base_is_color_and_font_agnostic():
    # 'structural' is the "improve the defaults, leave my colors/fonts alone"
    # preset — it must not touch the prop_cycle or font family.
    plt.rcdefaults()
    before_cycle = rcParams["axes.prop_cycle"]
    before_family = list(rcParams["font.family"])
    sph.set_base_style("structural")
    assert rcParams["axes.prop_cycle"] == before_cycle
    assert list(rcParams["font.family"]) == before_family


def test_standard_base_sets_font_stack_and_cycle():
    # 'standard' (the opinionated default) DOES set a font stack + a default
    # cycle (which set_palette can still override afterward).
    plt.rcdefaults()
    sph.set_base_style("standard")
    assert rcParams["font.sans-serif"][0] == "TeX Gyre Heros"
    assert len(rcParams["axes.prop_cycle"]) == 6


def test_pretty1_alias_resolves_to_standard():
    plt.rcdefaults()
    sph.set_base_style("pretty1")
    assert rcParams["font.sans-serif"][0] == "TeX Gyre Heros"


def test_base_preset_font_keys_are_stacks():
    # Every preset's font keys must be lists (stacks), never bare strings,
    # so the look degrades gracefully instead of silently hitting DejaVu.
    for name, preset in sph.BASE_PRESETS.items():
        for key in ("font.sans-serif", "font.serif", "font.family"):
            if key in preset and key != "font.family":
                assert isinstance(preset[key], list), f"{name}:{key} not a list"


def test_base_style_specific_rcs_merge_on_preset():
    plt.rcdefaults()
    sph.set_base_style("journal", {"axes.grid": False, "font.size": 13.0})
    assert rcParams["font.family"][0] == "serif"   # preset applied
    assert rcParams["axes.grid"] is False          # override merged
    assert rcParams["font.size"] == 13.0


def test_set_base_style_accepts_dict():
    sph.set_base_style({"xtick.direction": "out", "font.size": 11.0})
    assert rcParams["xtick.direction"] == "out"
    assert rcParams["font.size"] == 11.0


def test_set_base_style_dict_with_overrides():
    # specific_RCs wins over the same key in the dict preset.
    sph.set_base_style({"font.size": 11.0}, specific_RCs={"font.size": 13.0})
    assert rcParams["font.size"] == 13.0


def test_set_theme_accepts_matplotlib_builtin():
    # Any name in matplotlib.style.library resolves as a theme.
    sph.set_theme("ggplot")
    assert rcParams["axes.facecolor"] != "white"  # ggplot's gray background


def test_set_theme_sph_preset_wins_over_mpl():
    # skyplothelper preset names resolve from _THEMES, not the mpl library.
    sph.set_theme("dark_sky")
    assert rcParams["figure.facecolor"] == "#010409"


def test_set_theme_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown theme"):
        sph.set_theme("definitely_not_a_theme_anywhere")


# ---- annotation palettes ---------------------------------------------

_ANNOT_ROLES = {"fig_bg", "ax_bg", "text", "text2", "label", "compass",
                "grid", "grid2", "frame", "stars", "accent", "accent2"}


def test_annotation_palettes_well_formed():
    for name, pal in sph.ANNOTATION_PALETTES.items():
        assert set(pal) == _ANNOT_ROLES, name
        assert all(_is_color(v) for v in pal.values()), name


def test_annotation_palette_coffee_aliases_denim():
    # 'coffee' was renamed to 'denim'; the legacy name still resolves.
    assert "denim" in sph.ANNOTATION_PALETTES
    assert "coffee" not in sph.ANNOTATION_PALETTES
    fig, ax = plt.subplots()
    via_alias = sph.style_annotation(ax, "coffee")
    plt.close(fig)
    assert via_alias is sph.ANNOTATION_PALETTES["denim"]


def test_publication_palette_axes_are_journal_dark():
    """The 'publication' annotation palette drives the spine/ticks ('frame')
    and ticklabels ('text2'); both must be journal-dark, not the old light-
    gray 'draft' look (#9A9A9A / #8C8C8C). Frame is near-black charcoal;
    ticklabels match (no lighter than) the dark axis-label gray ('text')."""
    pal = sph.ANNOTATION_PALETTES["publication"]

    def _lum(c):
        r, g, b = mcolors.to_rgb(c)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    assert _lum(pal["frame"]) < 0.15, pal["frame"]    # spine + ticks: charcoal
    assert _lum(pal["text2"]) < 0.25, pal["text2"]    # ticklabels: dark
    # ticklabels no lighter than the dark axis labels (the hierarchy split)
    assert _lum(pal["text2"]) <= _lum(pal["text"]) + 1e-6


# ---- WCSAxes styling bridge ------------------------------------------

def test_style_wcs_axes_modifies_ticks_and_frame():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    out = sph.style_wcs_axes(ax, tick_color="#ff0000", frame_color="#00ff00",
                             major_size=9, width=2.0, minor_frequency=4,
                             grid=True, grid_color="#123456")
    assert out is ax
    assert mcolors.same_color(coord_ticks(ax.coords[0]).get_color(), "#ff0000")
    assert coord_ticks(ax.coords[0]).get_ticksize() == 9
    assert mcolors.same_color(ax.coords.frame.get_color(), "#00ff00")


def _overlay_tick_dirs(ax):
    """Signed radial direction of each sph overlay tick mark: +1 outward
    (away from frame center), -1 inward. Empty list if none present."""
    import numpy as np
    ax.figure.canvas.draw()
    cx = float(np.mean(ax.get_xlim()))
    cy = float(np.mean(ax.get_ylim()))
    signs = []
    for ln in ax.lines:
        if not getattr(ln, "_sph_overlay_tick", False):
            continue
        anchor = getattr(ln, "_sph_tick_anchor", None)
        if anchor is None:
            continue
        xs, ys = ln.get_data()
        far = ((xs[1], ys[1]) if (xs[0], ys[0]) == tuple(anchor)
               else (xs[0], ys[0]))
        vx, vy = far[0] - anchor[0], far[1] - anchor[1]
        rx, ry = anchor[0] - cx, anchor[1] - cy
        signs.append(1 if (vx * rx + vy * ry) > 0 else -1)
    return signs


def test_elliptical_overlay_ticks_build_honor_rc_direction():
    """AIT all-sky Dec boundary ticks (sph overlay, since astropy can't draw
    inward ticks on a curved spine) follow the active base style's tick
    direction at build time: 'standard' sets xtick.direction='in' -> inward."""
    sph.set_base_style("standard")
    ax = make_wcs_frame(111, projection="AIT", fig=plt.figure())
    dirs = _overlay_tick_dirs(ax)
    assert dirs, "no sph overlay ticks found on AIT frame"
    assert all(d == -1 for d in dirs), dirs


def test_style_wcs_axes_redirects_elliptical_overlay_ticks():
    """style_wcs_axes(direction=...) flips the sph-drawn elliptical boundary
    ticks post-build — the report's case (set_tick_out / structural / explicit
    direction all previously no-ops on these curved-spine ticks)."""
    with plt.style.context("default"):  # mpl default xtick.direction='out'
        ax = make_wcs_frame(111, projection="AIT", fig=plt.figure())
        ax.figure.canvas.draw()
        sph.style_wcs_axes(ax, direction="in")
        assert all(d == -1 for d in _overlay_tick_dirs(ax))
        sph.style_wcs_axes(ax, direction="out")
        assert all(d == 1 for d in _overlay_tick_dirs(ax))


def test_style_wcs_axes_tick_stroke_wraps_idempotently():
    """stroke_lw/stroke_color install a two-pass tick stroke (astropy's tick
    renderer ignores path_effects); the draw wrap is idempotent and removable.
    """
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(83.6, 22.0),
                        fov_deg=4.0, fig=fig, grid=False)
    ticks = coord_ticks(ax.coords[0])
    # The stored original must always be astropy's class draw (a bound method
    # of Ticks.draw) — never a wrap — which is what prevents wrap-stacking.
    def _orig_is_class_method():
        return getattr(ticks._sph_orig_tick_draw, "__func__", None) \
            is type(ticks).draw

    # No stroke by default → no wrap (no instance-level draw override).
    sph.style_wcs_axes(ax, tick_color="white", width=1.5)
    assert not hasattr(ticks, "_sph_orig_tick_draw")
    assert "draw" not in ticks.__dict__
    # With stroke → wrapped; the two-pass draw renders without error.
    sph.style_wcs_axes(ax, tick_color="white", width=1.5,
                       stroke_lw=3.0, stroke_color="black")
    assert "draw" in ticks.__dict__ and _orig_is_class_method()
    fig.canvas.draw()
    # Re-styling with stroke must not stack wraps: the stored original is still
    # astropy's class method, not the previous wrap.
    sph.style_wcs_axes(ax, tick_color="white", width=1.5,
                       stroke_lw=4.0, stroke_color="red")
    assert _orig_is_class_method()
    fig.canvas.draw()
    # Re-styling without a stroke removes the wrap.
    sph.style_wcs_axes(ax, tick_color="white", width=1.5)
    assert not hasattr(ticks, "_sph_orig_tick_draw")
    assert "draw" not in ticks.__dict__
    fig.canvas.draw()
    plt.close(fig)


def test_style_wcs_axes_minor_size_explicit_sets_minor_ticksize():
    # Explicit minor_size must reach astropy's set_minor_ticksize so minors
    # render as short subdivisions rather than inheriting the major length.
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(83.6, 22.0),
                        fov_deg=4.0, fig=fig, grid=False)
    sph.style_wcs_axes(ax, direction="in", major_size=10,
                       minor_ticks=True, minor_size=4)
    assert coord_ticks(ax.coords[0]).get_minor_ticksize() == 4
    assert coord_ticks(ax.coords[0]).get_ticksize() == 10


def test_style_wcs_axes_minor_size_omitted_leaves_inherited_length():
    # Opt-in: omitting minor_size must not touch the inherited (major) length.
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(83.6, 22.0),
                        fov_deg=4.0, fig=fig, grid=False)
    before = coord_ticks(ax.coords[0]).get_minor_ticksize()
    sph.style_wcs_axes(ax, major_size=10, minor_ticks=True)
    assert coord_ticks(ax.coords[0]).get_minor_ticksize() == before


def _count_minor_ticks(ax):
    n = 0
    for ci in (0, 1):
        for v in coord_ticks(ax.coords[ci]).minor_world.values():
            n += len(v)
    return n


def test_style_wcs_axes_minor_ticks_render_on_value_based_majors():
    """minor_ticks=True must actually place minors on a make_wcs_frame field
    frame, whose majors come from an explicit value list — astropy's native
    minor locator places none there, so an interpolating one is installed."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(83.6287, 22.0147),
                        fov_deg=4.0, fig=fig, grid=False)
    sph.style_wcs_axes(ax, minor_ticks=True, minor_frequency=4)
    fig.canvas.draw()
    assert _count_minor_ticks(ax) > 0


def test_style_wcs_axes_minor_ticks_false_places_none():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(83.6287, 22.0147),
                        fov_deg=4.0, fig=fig, grid=False)
    sph.style_wcs_axes(ax, minor_ticks=False)
    fig.canvas.draw()
    assert _count_minor_ticks(ax) == 0


def test_style_wcs_axes_defaults_from_rcparams():
    # With no explicit args it should pull tick color/size from rcParams.
    rcParams["xtick.color"] = "#abcdef"
    rcParams["xtick.major.size"] = 7
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    sph.style_wcs_axes(ax)
    assert mcolors.same_color(coord_ticks(ax.coords[0]).get_color(), "#abcdef")
    assert coord_ticks(ax.coords[0]).get_ticksize() == 7


def test_style_wcs_axes_plain_axes_warns_and_noops():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    with pytest.warns(UserWarning, match="non-WCSAxes"):
        out = sph.style_wcs_axes(ax)
    assert out is ax


def test_style_wcs_axes_coords_subset():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    # Style only coord 0; coord 1 keeps its prior color.
    before1 = coord_ticks(ax.coords[1]).get_color()
    sph.style_wcs_axes(ax, tick_color="#ff0000", coords=[0])
    assert mcolors.same_color(coord_ticks(ax.coords[0]).get_color(), "#ff0000")
    assert coord_ticks(ax.coords[1]).get_color() == before1


def test_apply_annotation_palette_wcsaxes():
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    pal = sph.style_annotation(ax, "night")
    assert pal is sph.ANNOTATION_PALETTES["night"]
    assert mcolors.same_color(fig.patch.get_facecolor(), pal["fig_bg"])
    assert mcolors.same_color(ax.get_facecolor(), pal["ax_bg"])
    assert mcolors.same_color(ax.coords.frame.get_color(), pal["frame"])


def test_apply_annotation_palette_plain_axes():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pal = sph.style_annotation(ax, "parchment")
    assert mcolors.same_color(ax.get_facecolor(), pal["ax_bg"])
    for spine in ax.spines.values():
        assert mcolors.same_color(spine.get_edgecolor(), pal["frame"])


def test_apply_annotation_palette_unknown_raises():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    with pytest.raises(ValueError, match="Unknown annotation palette"):
        sph.style_annotation(ax, "nope")


# ---- composable entry point ------------------------------------------

def test_set_style_composes_all_layers():
    sph.set_style(base="pretty1", theme="dark_sky", palette="nightcap")
    assert rcParams["xtick.direction"] == "in"             # base
    assert mcolors.same_color(rcParams["figure.facecolor"], "#010409")  # theme
    assert (rcParams["axes.prop_cycle"].by_key()["color"]
            == sph.CYCLE_PALETTES["nightcap"]["colors"])         # palette


def test_set_style_palette_only():
    before_face = rcParams["figure.facecolor"]
    sph.set_style(palette="atlas")
    assert (rcParams["axes.prop_cycle"].by_key()["color"]
            == sph.CYCLE_PALETTES["atlas"]["colors"])
    assert rcParams["figure.facecolor"] == before_face     # theme untouched


def test_style_context_triple_restores():
    before = rcParams["figure.facecolor"]
    with sph.style_context(theme="twilight", palette="velvet"):
        assert mcolors.same_color(rcParams["figure.facecolor"], "#0f0f23")
        assert (rcParams["axes.prop_cycle"].by_key()["color"]
                == sph.CYCLE_PALETTES["velvet"]["colors"])
    assert rcParams["figure.facecolor"] == before


def test_style_context_legacy_form_still_works():
    with sph.style_context("pretty1"):
        assert rcParams["xtick.direction"] == "in"


# ---- font stacks must be able to render the marks sph emits ----

# matplotlib selects ONE installed family per string (no per-glyph fallback),
# so the DejaVu backstop at the end of each stack is unreachable whenever an
# earlier face is installed. EVERY face in a stack must therefore carry the
# prime marks, which appear in every arcmin/arcsec label. Carlito led the sans
# stack and has no U+2032, so ordinary Linux boxes rendered arcmin as tofu.

_PRIME_MARKS = "′″"  # arcmin / arcsec

# Known to lack U+2032. Calibri is here too, which is why Carlito could not be
# swapped for a Calibri-metric equivalent: the metric target itself lacks it.
_NO_PRIME = {"Carlito", "Calibri", "Fira Sans", "Ubuntu"}


import warnings  # noqa: E402


def _missing_glyph_warnings(text):
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, ax = plt.subplots()
        ax.set_xlabel(text)
        fig.canvas.draw()
        plt.close(fig)
        return [str(w.message) for w in caught
                if "missing from" in str(w.message).lower()]


@pytest.mark.parametrize("base", ["standard", "minimalist", "journal",
                                  "structural"])
def test_base_preset_renders_prime_marks(base):
    sph.set_style(base=base)
    assert _missing_glyph_warnings(f"({_PRIME_MARKS})") == []


def test_prime_check_would_catch_a_glyph_incomplete_stack():
    """The guard above is only meaningful if it can fail — prove it does."""
    from matplotlib import font_manager as fm
    try:
        resolved = fm.findfont(fm.FontProperties(family=["Carlito"]),
                               fallback_to_default=False)
    except Exception:
        pytest.skip("Carlito not installed on this machine")
    if "carlito" not in resolved.lower():
        pytest.skip("Carlito not installed on this machine")
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Carlito", "DejaVu Sans"]
    assert _missing_glyph_warnings("′") != []


def test_prime_less_stack_warns_but_is_honored():
    """A face that can't render ′ is reported, NOT swapped out.

    Promoting the fallback rewrote font.serif / font.sans-serif figure-wide,
    so a preset naming a specific face silently rendered in DejaVu — 'journal'
    advertised TeX Gyre Termes and drew the default-matplotlib look it exists
    to avoid. Losing a whole figure's typeface over two glyphs that appear
    only in sexagesimal tick labels is the disproportionate remedy; the tofu
    it permits is visible, and the warning names both fixes.
    """
    from skyplothelper.style import _coerce_font_stack, _face_has_marks
    if _face_has_marks("Carlito") is not False:
        pytest.skip("needs a known prime-less face installed to exercise this")
    with pytest.warns(UserWarning, match="prime marks"):
        out = _coerce_font_stack(["Carlito"], "DejaVu Sans")
    assert out[0] == "Carlito", "the requested face must still win"
    assert "DejaVu Sans" in out, "the not-installed-at-all net stays appended"


def test_repair_leaves_a_capable_stack_alone():
    """It must not fire spuriously and override a perfectly good font."""
    from skyplothelper.style import _coerce_font_stack, _face_has_marks
    if not _face_has_marks("DejaVu Sans"):
        pytest.skip("DejaVu Sans unavailable")
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any warning fails the test
        out = _coerce_font_stack(["DejaVu Sans"], "DejaVu Sans")
    assert out[0] == "DejaVu Sans"


def test_no_preset_stack_contains_a_prime_less_face():
    """Membership guard: a face that can't render ′ must not be in a stack.

    Stronger than an ordering rule — ordering only helps when a later face
    happens to be installed too.
    """
    for name, preset in sph.BASE_PRESETS.items():
        for key in ("font.sans-serif", "font.serif", "font.monospace"):
            stack = preset.get(key) or []
            bad = _NO_PRIME.intersection(stack)
            assert not bad, f"{name}/{key} contains prime-less {sorted(bad)}"


def test_installed_faces_in_the_default_sans_stack_carry_primes():
    """Whatever is actually installed here must be able to render ′ ″."""
    from skyplothelper.style import _face_has_marks
    checked = 0
    for fam in sph.BASE_PRESETS["standard"].get("font.sans-serif") or []:
        has = _face_has_marks(fam)
        if has is None:
            continue                 # not installed here — nothing to check
        assert has, f"{fam} is in the default stack but lacks ′ or ″"
        checked += 1
    assert checked, "no font from the stack was installed — test proved nothing"


# ---- shared frame -> short label mapping ----

def test_frame_short_labels_covers_the_equatorial_aliases():
    """Three private copies existed and had diverged: the cartopy one lacked
    fk4/fk5, so an FK5 frame silently labeled its axes 'Lon'/'Lat'."""
    from skyplothelper.constants import frame_short_labels
    for frame in ("icrs", "fk5", "fk4"):
        assert frame_short_labels(frame) == ("RA", "Dec")
    assert frame_short_labels("galactic") == ("l", "b")
    assert frame_short_labels("supergalactic") == ("SGL", "SGB")
    for frame in ("ecliptic", "geocentrictrueecliptic",
                  "barycentrictrueecliptic", "heliocentrictrueecliptic"):
        assert frame_short_labels(frame) == ("λ", "β")


def test_frame_short_labels_is_case_insensitive_and_has_a_default():
    from skyplothelper.constants import frame_short_labels
    assert frame_short_labels("GALACTIC") == ("l", "b")
    assert frame_short_labels(None) == ("lon", "lat")
    assert frame_short_labels("nonsense") == ("lon", "lat")
    assert frame_short_labels(None, default=("X", "Y")) == ("X", "Y")


# ---- muted_ink: the B2 mechanism ----

def test_muted_ink_preserves_the_light_tone():
    """Muted tones must NOT be promoted to full ink — that would flatten a
    deliberate hierarchy (these labels are smaller AND lighter on purpose)."""
    from skyplothelper.style import muted_ink
    assert muted_ink("white", light="0.35") == "0.35"
    assert muted_ink("#ffffff", light="0.4") == "0.4"


def test_muted_ink_mirrors_the_tone_on_a_dark_background():
    from skyplothelper.style import muted_ink
    assert muted_ink("#0d1117", light="0.35") == "0.65"
    assert muted_ink("#0d1117", light="0.3") == "0.70"


def test_muted_ink_honors_an_explicit_dark_tone():
    from skyplothelper.style import muted_ink
    assert muted_ink("#0d1117", light="0.4", dark="0.7") == "0.7"


def test_is_dark_background_uses_luminance_not_a_prefix():
    from skyplothelper.style import is_dark_background
    assert is_dark_background("#1D1C1A")      # dark, no leading '#0'
    assert is_dark_background("black")
    assert not is_dark_background("white")
    assert not is_dark_background("#ffffff")
    assert not is_dark_background("not-a-color")
