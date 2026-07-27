"""overlays.surveys + overlays.constellations verification.

Where the canonical suites smoke-test 3 of 21 surveys and only basic
constellation behavior, this file:

  * Iterates every entry in SURVEY_FOOTPRINTS and verifies it can be
    rendered (in both 'polygon' and 'box' boundary_style modes).
  * Verifies fuzzy survey-name lookup, color propagation, and the
    box-vs-polygon style switch.
  * Iterates the 88 constellations and verifies labels can be filtered
    by subset.
  * Verifies add_constellation_boundaries returns a sensible artist count.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from skyplothelper.overlays.constellations import (
    _CONSTELLATION_CENTERS,
    _CONSTELLATION_NAMES,
    add_constellation_boundaries,
    add_constellation_labels,
)
from skyplothelper.overlays.surveys import (
    SURVEY_FOOTPRINTS,
    add_survey_footprint,
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
# SURVEY_FOOTPRINTS — every registered survey renders
# ============================================================

@pytest.mark.parametrize("survey_key", sorted(SURVEY_FOOTPRINTS.keys()))
def test_add_survey_footprint_every_registered_survey(survey_key, allsky_axes):
    """Each registered survey must render in default (polygon) mode
    without raising at draw time."""
    fig, ax = allsky_axes
    patches = add_survey_footprint(ax, survey_key)
    fig.canvas.draw()
    assert isinstance(patches, list)
    assert len(patches) >= 1, f"survey {survey_key!r} produced no artists"


@pytest.mark.parametrize("survey_key", sorted(SURVEY_FOOTPRINTS.keys()))
def test_add_survey_footprint_box_style_works(survey_key, allsky_axes):
    """boundary_style='box' should also work for every survey."""
    fig, ax = allsky_axes
    patches = add_survey_footprint(ax, survey_key, boundary_style="box")
    fig.canvas.draw()
    assert isinstance(patches, list)


# ============================================================
# Survey lookup — fuzzy match, custom color
# ============================================================

@pytest.mark.parametrize("alias", ["sdss", "SDSS", "SDss", "Pan-STARRS"])
def test_survey_name_normalization(alias, allsky_axes):
    """Survey lookup is case-insensitive and tolerant of separators
    (dashes and underscores are stripped before matching)."""
    fig, ax = allsky_axes
    patches = add_survey_footprint(ax, alias)
    fig.canvas.draw()
    assert len(patches) >= 1


def test_unknown_survey_raises(allsky_axes):
    fig, ax = allsky_axes
    with pytest.raises(ValueError, match="(?i)unknown"):
        add_survey_footprint(ax, "this_survey_does_not_exist")


def test_add_survey_footprint_color_propagates(allsky_axes):
    """Custom color overrides the survey's default."""
    fig, ax = allsky_axes
    patches = add_survey_footprint(ax, "sdss", color="C9", lw=2.5)
    # All patches inherit the explicit color (compare RGB, ignoring alpha).
    from matplotlib.colors import to_rgba
    expected = to_rgba("C9")[:3]
    for p in patches:
        ec = p.get_edgecolor()
        # ec may be 4-tuple (rgba) or array-like
        if hasattr(ec, "__len__") and len(ec) >= 3:
            ec_rgb = tuple(ec[:3])
        else:
            ec_rgb = to_rgba(ec)[:3]
        assert ec_rgb == pytest.approx(expected, abs=1e-3)


# ============================================================
# Constellations — registry consistency + rendering
# ============================================================

def test_88_constellations_in_registries():
    """The IAU recognizes 88 constellations; the package should match."""
    assert len(_CONSTELLATION_CENTERS) == 88
    assert len(_CONSTELLATION_NAMES) == 88


def test_constellation_centers_and_names_share_keys():
    """Both registries must use the same abbreviation keys."""
    assert set(_CONSTELLATION_CENTERS.keys()) == set(_CONSTELLATION_NAMES.keys())


def test_add_constellation_boundaries_renders(allsky_axes):
    fig, ax = allsky_axes
    n_before = len(ax.lines) + len(ax.collections)
    add_constellation_boundaries(ax)
    fig.canvas.draw()
    n_after = len(ax.lines) + len(ax.collections)
    # Should add at least a handful of line/collection artists
    assert n_after > n_before


def test_add_constellation_labels_default_returns_88_labels(allsky_axes):
    fig, ax = allsky_axes
    n_before = len(ax.texts)
    add_constellation_labels(ax, labels="abbr")
    n_after = len(ax.texts)
    # Default: one label per constellation (88 of them)
    assert (n_after - n_before) == 88


def test_add_constellation_labels_subset_filter(allsky_axes):
    """The constellations= parameter limits labels to a subset."""
    fig, ax = allsky_axes
    subset = ["UMA", "UMI", "CAS", "CYG", "LYR"]  # registry uses uppercase keys
    n_before = len(ax.texts)
    add_constellation_labels(ax, labels="abbr", constellations=subset)
    n_after = len(ax.texts)
    assert (n_after - n_before) == len(subset)


def test_add_constellation_labels_full_names(allsky_axes):
    """labels='name' should use the full Latin names instead of abbrs."""
    fig, ax = allsky_axes
    add_constellation_labels(ax, labels="name",
                             constellations=["UMA", "ORI"])
    texts_added = ax.texts[-2:]
    contents = {t.get_text() for t in texts_added}
    expected = {_CONSTELLATION_NAMES["UMA"], _CONSTELLATION_NAMES["ORI"]}
    assert contents == expected


def test_add_constellation_lines_default_returns_lines(allsky_axes):
    """add_constellation_lines() draws all 89 entries' polylines
    (88 IAU constellations + Serpens Caput/Cauda split)."""
    from skyplothelper.overlays.constellations import add_constellation_lines

    fig, ax = allsky_axes
    n_before = len(ax.lines)
    out = add_constellation_lines(ax)
    n_after = len(ax.lines)
    assert n_after - n_before == len(out)
    # 150 polyline segments in the bundled dataset.
    assert len(out) >= 100


def test_add_constellation_lines_rank_max_filters_subset(allsky_axes):
    """rank_max=1 yields strictly fewer lines than the full dataset."""
    from skyplothelper.overlays.constellations import add_constellation_lines

    fig1, ax1 = allsky_axes
    all_lines = add_constellation_lines(ax1)
    fig2, ax2 = allsky_axes
    rank1 = add_constellation_lines(ax2, rank_max=1)
    assert 0 < len(rank1) < len(all_lines)


def test_add_constellation_lines_constellations_filter(allsky_axes):
    """The constellations= filter limits output to the named subset."""
    from skyplothelper.overlays.constellations import add_constellation_lines

    fig, ax = allsky_axes
    subset = ["ORI", "UMA", "CAS"]
    out = add_constellation_lines(ax, constellations=subset)
    # Each constellation has multiple segments; the three together
    # should be well under the full 100+ dataset.
    assert len(out) > 0
    assert len(out) < 50


def test_add_constellation_polygon_returns_one_patch_for_single_constellation(allsky_axes):
    """Most constellations have a single polygon — add_constellation_polygon
    returns one patch (or one ``add_spherical_polygon`` sub-list)."""
    from skyplothelper.overlays.constellations import add_constellation_polygon

    fig, ax = allsky_axes
    out = add_constellation_polygon(ax, 'UMI', facecolor='lightblue',
                                      alpha=0.3)
    assert isinstance(out, list)
    assert len(out) >= 1


def test_add_constellation_polygon_serpens_returns_two_patches(allsky_axes):
    """Serpens has two distinct polygons (Caput + Cauda); both must
    be drawn in a single call."""
    from skyplothelper.overlays.constellations import add_constellation_polygon

    fig, ax = allsky_axes
    out = add_constellation_polygon(ax, 'SER', facecolor='salmon',
                                      alpha=0.35)
    # Each polygon may itself split into sub-patches at the antimeridian
    # via add_spherical_polygon, but the total should be at least 2.
    assert len(out) >= 2


def test_add_constellation_polygon_case_insensitive(allsky_axes):
    """Lower-case / mixed-case codes resolve the same as upper-case."""
    from skyplothelper.overlays.constellations import add_constellation_polygon

    fig, ax = allsky_axes
    out_lower = add_constellation_polygon(ax, 'ori', facecolor='C0')
    fig2, ax2 = allsky_axes
    out_upper = add_constellation_polygon(ax2, 'ORI', facecolor='C0')
    assert len(out_lower) == len(out_upper)


def test_add_constellation_polygon_unknown_raises_keyerror(allsky_axes):
    from skyplothelper.overlays.constellations import add_constellation_polygon

    fig, ax = allsky_axes
    with pytest.raises(KeyError, match="unknown IAU code"):
        add_constellation_polygon(ax, 'XYZ')


def test_load_constellation_polygons_covers_all_88():
    """The polygon loader exposes every constellation in the centers
    dict (88 abbrs; SER appears once but maps to 2 polygons in the
    value list)."""
    from skyplothelper.overlays.constellations import (
        _CONSTELLATION_CENTERS,
        _load_constellation_polygons,
    )

    polys = _load_constellation_polygons()
    assert set(_CONSTELLATION_CENTERS).issubset(polys)
    assert len(polys['SER']) == 2  # Caput + Cauda


def test_constellation_lines_data_has_expected_constellations():
    """The bundled .npz covers all 88 IAU constellations + the Serpens
    Caput/Cauda split (89 entries total)."""
    from skyplothelper.overlays.constellations import (
        _CONSTELLATION_CENTERS,
        _load_constellation_lines,
    )

    data = _load_constellation_lines()
    cst_codes = set(data["cst"].tolist())
    # Every entry in the centers dict should appear in the lines data.
    assert set(_CONSTELLATION_CENTERS).issubset(cst_codes)
    assert "SER" in cst_codes  # Serpens is one of the 88
    # Source string mentions d3-celestial + IAU
    assert "d3-celestial" in data["source"]
    assert "IAU" in data["source"]


def test_add_constellation_labels_default_offsets_shift_flagged(allsky_axes):
    """apply_default_offsets=True (default) moves the flagged labels off
    their raw polygon centroid positions; passing False restores them."""
    from skyplothelper.overlays.constellations import (
        _CONSTELLATION_CENTERS,
        _CONSTELLATION_LABEL_OFFSETS_DEG,
    )

    # All entries in the offset table should be in the centers dict.
    assert set(_CONSTELLATION_LABEL_OFFSETS_DEG).issubset(
        _CONSTELLATION_CENTERS)
    # Coverage sanity: the 8 curated headline entries.
    assert len(_CONSTELLATION_LABEL_OFFSETS_DEG) >= 8

    flagged = sorted(_CONSTELLATION_LABEL_OFFSETS_DEG)

    fig, ax = allsky_axes
    add_constellation_labels(ax, labels="abbr",
                             constellations=flagged,
                             apply_default_offsets=False)
    raw_positions = {
        t.get_text(): t.get_position() for t in ax.texts[-len(flagged):]}

    fig2, ax2 = allsky_axes
    add_constellation_labels(ax2, labels="abbr",
                             constellations=flagged,
                             apply_default_offsets=True)
    offset_positions = {
        t.get_text(): t.get_position() for t in ax2.texts[-len(flagged):]}

    # Every flagged label should have moved.
    for abbr in flagged:
        assert raw_positions[abbr] != offset_positions[abbr]

    # Non-flagged labels are untouched.
    fig3, ax3 = allsky_axes
    add_constellation_labels(ax3, labels="abbr",
                             constellations=["UMA", "ORI", "CYG"],
                             apply_default_offsets=True)
    on_pos = {t.get_text(): t.get_position() for t in ax3.texts[-3:]}

    fig4, ax4 = allsky_axes
    add_constellation_labels(ax4, labels="abbr",
                             constellations=["UMA", "ORI", "CYG"],
                             apply_default_offsets=False)
    off_pos = {t.get_text(): t.get_position() for t in ax4.texts[-3:]}

    for abbr in ["UMA", "ORI", "CYG"]:
        assert on_pos[abbr] == off_pos[abbr]
