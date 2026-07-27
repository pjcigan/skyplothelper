"""Tests for skyplothelper.star_colors (perceived star colors)."""

import numpy as np
import pytest

from skyplothelper import star_colors as _sc
from skyplothelper.star_colors import (
    bp_rp_to_rgb,
    bv_to_rgb,
    color_index_to_rgb,
    teff_to_rgb,
)


def test_scalar_returns_rgb_triplet():
    rgb = teff_to_rgb(5778)
    assert rgb.shape == (3,)
    assert np.all((rgb >= 0) & (rgb <= 1))


def test_array_returns_n_by_3():
    rgb = teff_to_rgb([3500, 5778, 10000])
    assert rgb.shape == (3, 3)


def test_sun_is_white():
    """A Sun-temperature star reads white (not green) — the tristimulus point."""
    r, g, b = teff_to_rgb(5778)
    assert min(r, g, b) > 0.9          # all channels high → white-ish
    assert abs(r - b) < 0.12           # roughly neutral


def test_hot_star_is_blue_white():
    r, g, b = teff_to_rgb(10000)
    assert b > r                        # blue dominates


def test_cool_star_is_orange():
    r, g, b = teff_to_rgb(3500)
    assert r > b                        # red dominates


def test_saturation_desaturates_toward_white():
    vivid = teff_to_rgb(10000, saturation=1.0)
    muted = teff_to_rgb(10000, saturation=0.4)
    assert np.ptp(muted) < np.ptp(vivid)          # less color spread
    assert np.all(teff_to_rgb(10000, saturation=0.0) > 0.99)   # → white


def test_out_of_range_teff_clipped_not_erroring():
    assert teff_to_rgb(500).shape == (3,)         # below 1667 K
    assert teff_to_rgb(50000).shape == (3,)       # above 25000 K


# ---------------------------------------------------------------------------
# bv_to_rgb (Ballesteros 2012 B-V → Teff, then teff_to_rgb)
# ---------------------------------------------------------------------------

def test_bv_sun_matches_teff_5778():
    """B-V 0.65 → ~5778 K (the Ballesteros relation), so same color as the Sun."""
    assert np.allclose(bv_to_rgb(0.65), teff_to_rgb(5778), atol=0.02)


def test_bv_blue_and_red_ends():
    blue = bv_to_rgb(0.0)              # Sirius/Vega-like
    red = bv_to_rgb(1.85)             # Betelgeuse-like
    assert blue[2] > blue[0]          # blue-white
    assert red[0] > red[2]           # orange


def test_bv_vectorized():
    assert bv_to_rgb([0.0, 0.65, 1.85]).shape == (3, 3)


@pytest.mark.parametrize("fn,arg", [(teff_to_rgb, 6000), (bv_to_rgb, 0.5)])
def test_usable_as_matplotlib_color(fn, arg):
    import matplotlib
    rgb = fn(arg)
    assert matplotlib.colors.to_rgba(tuple(rgb))     # valid mpl color


# ---------------------------------------------------------------------------
# color_index_to_rgb / bp_rp_to_rgb (Pecaut & Mamajek 2013 sequence)
# ---------------------------------------------------------------------------

# The Sun's color index in each system (Pecaut & Mamajek G2V row).
_SUN_INDICES = {"B-V": 0.65, "BP-RP": 0.823, "g-r": 0.476, "J-K": 0.366}


@pytest.mark.parametrize("index,value", list(_SUN_INDICES.items()))
def test_color_index_sun_agrees_across_systems(index, value):
    """The Sun reads the same near-white color whichever index you arrive with
    — the whole point of routing every index through one Teff scale."""
    assert np.allclose(color_index_to_rgb(value, index), teff_to_rgb(5772),
                       atol=0.02)


def test_anchor_tables_are_strictly_monotonic():
    """Each survey index's interpolation anchors must be strictly increasing in
    color (so np.interp is well-posed) after the fold/plateau cleaning."""
    for key in ("bp-rp", "g-r", "j-k"):
        colors, teffs = _sc._INDEX_ANCHORS[key]
        assert len(colors) > 5
        assert np.all(np.diff(colors) > 0)          # strictly increasing color
        assert np.all(np.diff(teffs) < 0)           # → strictly decreasing Teff


def test_bp_rp_hue_ramps_blue_to_red():
    ramp = bp_rp_to_rgb([-0.1, 0.8, 2.5, 4.0])
    # Blue channel falls, red rises, monotonically, as the star reddens.
    assert np.all(np.diff(ramp[:, 2]) < 0)          # B decreasing
    assert ramp[0, 2] > ramp[0, 0]                  # bluest is blue-white
    assert ramp[-1, 0] > ramp[-1, 2]                # reddest is orange


def test_bp_rp_is_not_naive_bv():
    """BP-RP spans wider than B-V; a proper transform must NOT over-redden the
    way bv_to_rgb(bp_rp) would for the same numeric value."""
    v = 2.5
    proper = bp_rp_to_rgb(v)
    naive = bv_to_rgb(v)
    # The naive path treats 2.5 as a super-cool B-V star → far too red/dark blue.
    assert proper[2] > naive[2]                     # proper keeps more blue


def test_bp_rp_to_rgb_matches_general_entry_point():
    assert np.allclose(bp_rp_to_rgb(1.3), color_index_to_rgb(1.3, "BP-RP"))


def test_color_index_scalar_and_array_shapes():
    assert bp_rp_to_rgb(1.0).shape == (3,)
    assert bp_rp_to_rgb([0.0, 1.0, 2.0]).shape == (3, 3)


def test_color_index_aliases_and_case_insensitive():
    ref = color_index_to_rgb(1.0, "BP-RP")
    for alias in ("bp-rp", "bp_rp", "BP_RP", "gaia", " Bp-Rp "):
        assert np.allclose(color_index_to_rgb(1.0, alias), ref)
    assert np.allclose(color_index_to_rgb(0.5, "j_k"),
                       color_index_to_rgb(0.5, "J-Ks"))


def test_color_index_bv_matches_bv_to_rgb():
    """index='B-V' keeps the dedicated Ballesteros closed form."""
    assert np.allclose(color_index_to_rgb(0.65, "B-V"), bv_to_rgb(0.65))


def test_color_index_unknown_raises():
    with pytest.raises(ValueError, match="unknown color index"):
        color_index_to_rgb(1.0, "u-g")


def test_color_index_out_of_range_clipped():
    # Bluer/redder than the table extremes clip rather than erroring.
    assert bp_rp_to_rgb(-5.0).shape == (3,)
    assert bp_rp_to_rgb(99.0).shape == (3,)


def test_color_index_nan_input_is_masked():
    """A non-finite input color yields a masked (non-finite) RGB row so callers
    can detect missing photometry, instead of silently folding to gray."""
    out = bp_rp_to_rgb([0.0, np.nan, 2.5])
    assert out.shape == (3, 3)
    assert np.isfinite(out[0]).all() and np.isfinite(out[2]).all()
    assert not np.isfinite(out[1]).any()
