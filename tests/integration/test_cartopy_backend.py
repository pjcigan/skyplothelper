"""cartopy_backend verification.

The canonical ``tests/test_cartopy_backend.py`` smoke-tests one
projection. This file exhaustively iterates every entry in
``_CARTOPY_PROJECTIONS`` plus the ``_resolve_cartopy_crs`` lookup
edges (alias normalization, unknown-projection ValueError, central-lat
override).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

cartopy = pytest.importorskip("cartopy")

import cartopy.crs as ccrs  # noqa: E402

from skyplothelper.cartopy_backend import (  # noqa: E402
    _CARTOPY_PROJECTIONS,
    _resolve_cartopy_crs,
    list_cartopy_projections,
    make_cartopy_frame,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ============================================================
# _resolve_cartopy_crs — every registered projection resolves
# ============================================================

@pytest.mark.parametrize("name", sorted(_CARTOPY_PROJECTIONS.keys()))
def test_resolve_cartopy_crs_every_registered_name(name):
    crs = _resolve_cartopy_crs(name, center_lon=180)
    assert isinstance(crs, ccrs.Projection)


@pytest.mark.parametrize("alias, expected_class", [
    ("MOLLWEIDE", "Mollweide"),
    ("Mollweide", "Mollweide"),
    ("mollweide", "Mollweide"),
    ("PLATE-CARREE", "PlateCarree"),
    ("plate carree", "PlateCarree"),
    ("Stereographic", "Stereographic"),
])
def test_resolve_cartopy_crs_normalizes_aliases(alias, expected_class):
    """Lookups are case-insensitive and tolerate dashes/underscores/spaces."""
    crs = _resolve_cartopy_crs(alias, center_lon=0)
    assert type(crs).__name__ == expected_class


def test_resolve_cartopy_crs_unknown_raises():
    with pytest.raises(ValueError, match="(?i)unknown"):
        _resolve_cartopy_crs("not_a_projection", center_lon=0)


def test_resolve_cartopy_crs_central_latitude_override():
    """For azimuthal projections, center_lat changes the resulting CRS."""
    crs_a = _resolve_cartopy_crs("orthographic", center_lon=0, center_lat=0)
    crs_b = _resolve_cartopy_crs("orthographic", center_lon=0, center_lat=45)
    # Different central latitudes should produce different proj4 params.
    assert crs_a.proj4_params != crs_b.proj4_params


# ============================================================
# make_cartopy_frame — return type, projection propagation, options
# ============================================================

def test_make_cartopy_frame_returns_geoaxes():
    fig = plt.figure(figsize=(10, 5))
    ax = make_cartopy_frame(projection="mollweide", fig=fig)
    # GeoAxes is a subclass of matplotlib Axes that uses cartopy's CRS
    assert isinstance(ax.projection, ccrs.Projection)


@pytest.mark.parametrize("projection", [
    "mollweide", "robinson", "sinusoidal", "plate_carree",
    "orthographic", "lambert_azimuthal",
])
def test_make_cartopy_frame_each_projection(projection):
    """A handful of common projections must build a working frame."""
    fig = plt.figure(figsize=(8, 5))
    ax = make_cartopy_frame(projection=projection, fig=fig)
    fig.canvas.draw()
    assert isinstance(ax.projection, ccrs.Projection)


def test_make_cartopy_frame_with_features():
    """coastlines/land/ocean toggles add features to the axes."""
    fig = plt.figure(figsize=(10, 5))
    ax = make_cartopy_frame(projection="mollweide",
                            coastlines=True, land=True, ocean=True,
                            fig=fig)
    fig.canvas.draw()
    # At least one collection (land) and one line set (coastlines) added
    assert len(ax.artists) + len(ax.collections) + len(ax.lines) > 0


# ============================================================
# invert_lon auto-detect
# ============================================================
#
# The default for invert_lon was flipped to None (auto-detect):
#  - Earth-feature axes (coastlines/land/ocean/nightshade) → False
#    so longitude reads west-to-east left-to-right (cartographic).
#  - Sky-only axes → True (RA convention, longitude increases left).
# An explicit bool always overrides.

def _x_inverted(ax):
    """Robust check for axis inversion under cartopy's GeoAxes."""
    xl, xr = ax.get_xlim()
    return bool(xl > xr)


@pytest.mark.parametrize("kwargs,expected_inverted", [
    # Sky-only — no Earth features → invert (RA convention)
    (dict(coastlines=False, land=False, ocean=False), True),
    # Any Earth-feature flag set → don't invert (cartographic)
    (dict(coastlines=True), False),
    (dict(land=True), False),
    (dict(ocean=True), False),
    (dict(coastlines=True, land=True, ocean=True), False),
])
def test_make_cartopy_frame_invert_lon_auto(kwargs, expected_inverted):
    """invert_lon=None (default) auto-detects from Earth-feature flags."""
    fig = plt.figure(figsize=(8, 4))
    ax = make_cartopy_frame(projection="plate_carree", center=0,
                            fig=fig, **kwargs)
    fig.canvas.draw()
    assert _x_inverted(ax) is expected_inverted


def test_make_cartopy_frame_invert_lon_explicit_overrides_auto():
    """An explicit bool wins over the auto-detect."""
    # Earth features set, but explicit invert_lon=True still flips
    fig = plt.figure(figsize=(8, 4))
    ax = make_cartopy_frame(projection="plate_carree", center=0,
                            coastlines=True, invert_lon=True, fig=fig)
    fig.canvas.draw()
    assert _x_inverted(ax) is True

    # Sky-only, but explicit invert_lon=False keeps it cartographic
    fig2 = plt.figure(figsize=(8, 4))
    ax2 = make_cartopy_frame(projection="plate_carree", center=0,
                             invert_lon=False, fig=fig2)
    fig2.canvas.draw()
    assert _x_inverted(ax2) is False


# ============================================================
# list_cartopy_projections — runs and includes registered names
# ============================================================

def test_list_cartopy_projections_runs(capsys):
    list_cartopy_projections()
    out = capsys.readouterr().out
    for required in ("mollweide", "robinson", "plate_carree",
                     "orthographic"):
        assert required in out
