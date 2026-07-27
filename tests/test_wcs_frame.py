"""Tests for skyplothelper.wcs_frame: make_wcs_frame across projections."""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for tests

import matplotlib.pyplot as plt
import pytest
from astropy.visualization.wcsaxes import WCSAxes
from astropy.visualization.wcsaxes.frame import EllipticalFrame

import skyplothelper as sph
from skyplothelper._compat import coord_ticks
from skyplothelper.projections import frames as proj_frames
from skyplothelper.wcs_frame import (
    _clamp_spacing,
    _field_tick_values,
    apply_boundary_labels,
    clip_to_frame,
    dummy_allsky_hdr,
    dummy_offset_hdr,
    dummy_ortho_hdr,
    dummy_standard_hdr,
    make_wcs_frame,
)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all figures after each test to keep memory bounded."""
    yield
    plt.close("all")


# ---- Header generators ----

def test_dummy_allsky_hdr_basic():
    hdr = dummy_allsky_hdr(center_LONdeg=180, projection="AIT")
    assert hdr["CTYPE1"].endswith("AIT")
    assert hdr["CTYPE2"].endswith("AIT")
    assert hdr["CRVAL1"] == 180


def test_dummy_ortho_hdr_basic():
    hdr = dummy_ortho_hdr(center_LONdeg=0, center_LATdeg=0)
    assert hdr["CTYPE1"].endswith("SIN")
    assert hdr["CTYPE2"].endswith("SIN")


def test_dummy_offset_hdr_basic():
    hdr = dummy_offset_hdr(centercoords_deg=(180.0, 0.0))
    assert "CTYPE1" in hdr
    assert "CTYPE2" in hdr


def test_dummy_standard_hdr_basic():
    hdr = dummy_standard_hdr(centercoords_deg=(0.0, 0.0))
    assert "CTYPE1" in hdr
    assert "CTYPE2" in hdr


# ---- make_wcs_frame: one axes for each projection class ----

@pytest.mark.parametrize("projection", [
    # Pseudocylindrical (FITS)
    "AIT", "MOL", "SFL", "PAR",
    # Cylindrical (FITS)
    "CAR",
    # Zenithal (FITS)
    "TAN", "SIN",
])
def test_make_wcs_frame_fits_projections(projection):
    fig = plt.figure()
    ax = make_wcs_frame(111, projection=projection, fig=fig)
    assert isinstance(ax, WCSAxes)
    # CTYPE should reflect the requested projection
    ctype1 = ax.wcs.wcs.ctype[0]
    assert ctype1.endswith(projection), f"{ctype1} should end with {projection}"


@pytest.mark.parametrize("projection, expected_frame", [
    ("robinson", proj_frames.RobinsonFrame),
    ("kavrayskiy", proj_frames.KavrayskiyFrame),
    ("eckert_iv", proj_frames.Eckert4Frame),
    ("winkel_tripel", proj_frames.WinkelTripelFrame),
    ("mcbryde", proj_frames.McBrydeFrame),
])
def test_make_wcs_frame_non_fits_projections(projection, expected_frame):
    fig = plt.figure()
    ax = make_wcs_frame(111, projection=projection, fig=fig)
    # Non-FITS projections still produce WCSAxes (with custom transforms)
    assert isinstance(ax, WCSAxes)


def test_make_wcs_frame_bon_cusp_reaches_frame_bottom():
    """The BON cardioid's south cusp should sit near the bottom of the frame,
    not partway up with a dead band below (the y-extent is fit asymmetrically
    to the lopsided cardioid envelope)."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="BON", center=0, frame="galactic",
                        fig=fig)
    ymin, ymax = ax.get_ylim()
    # south cusp is the south pole on the central meridian
    disp = ax.get_transform("world").transform([[0.0, -89.99]])
    _, ycusp = ax.transData.inverted().transform(disp)[0]
    # cusp should land within the lowest ~8% of the visible y range
    frac = (ycusp - ymin) / (ymax - ymin)
    assert 0.0 <= frac < 0.08, f"BON cusp at {frac:.3f} of frame height"


def test_clip_to_projection_boundary_sets_clip_on_interrupted():
    """clip_to_projection_boundary installs a boundary clip on a bounded/
    interrupted projection (HPX) and leaves a continuous one (AIT) untouched.
    (Note: WCSAxes already auto-clips artists to the frame patch, so the check
    is whether clip_to_projection_boundary *changes* that clip.)"""
    from skyplothelper.wcs_frame import clip_to_projection_boundary
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="HPX", center=0, frame="galactic",
                        fig=fig)
    ln, = ax.plot([-50, 50], [10, -10], transform=ax.get_transform("world"))
    before = ln.get_clip_path()
    clip_to_projection_boundary(ax, ln)
    assert ln.get_clip_path() is not None
    assert ln.get_clip_path() is not before  # diamond boundary installed
    plt.close(fig)

    fig2 = plt.figure()
    ax2 = make_wcs_frame(111, projection="AIT", center=0, frame="galactic",
                         fig=fig2)
    ln2, = ax2.plot([-50, 50], [10, -10], transform=ax2.get_transform("world"))
    before2 = ln2.get_clip_path()
    clip_to_projection_boundary(ax2, ln2)
    assert ln2.get_clip_path() is before2  # continuous projection → no-op
    plt.close(fig2)

    # XPH (butterfly) has no closed-form boundary; it is clipped via the
    # NaN-edge limb detector.
    fig3 = plt.figure()
    ax3 = make_wcs_frame(111, projection="XPH", center=0, frame="galactic",
                         fig=fig3)
    ln3, = ax3.plot([-50, 50], [10, -10], transform=ax3.get_transform("world"))
    before3 = ln3.get_clip_path()
    clip_to_projection_boundary(ax3, ln3)
    assert ln3.get_clip_path() is not None
    assert ln3.get_clip_path() is not before3  # NaN-edge boundary installed
    plt.close(fig3)


@pytest.mark.parametrize("projection", ["TSC", "CSC", "QSC"])
def test_make_wcs_frame_cube_boundary_and_clip(projection):
    """Quadcubes get a cross-perimeter boundary (via _projection_boundary), a
    drawn outline, and a working data clip to it."""
    from skyplothelper.wcs_frame import (
        _projection_boundary,
        clip_to_projection_boundary,
    )
    fig = plt.figure()
    ax = make_wcs_frame(111, projection=projection, center=0,
                        frame="galactic", fig=fig, grid=True)
    p = _projection_boundary(ax)
    assert p is not None
    assert len(p.vertices) >= 12  # the 12-vertex plus/cross
    assert len(ax.lines) > 0      # overlay gridlines + the cross outline
    ln, = ax.plot([10, 200], [10, -10], transform=ax.get_transform("world"))
    before = ln.get_clip_path()
    clip_to_projection_boundary(ax, ln)
    assert ln.get_clip_path() is not before  # cube cross clip installed
    plt.close(fig)


@pytest.mark.parametrize("projection", ["HPX", "BON", "PCO", "COD"])
def test_make_wcs_frame_backfill_grid_builds(projection):
    """The gridline-backfill projections build with a grid without error and
    draw extra line artists (the dense overlay / boundary outline)."""
    center = (180, 30) if projection == "COD" else 0
    fig = plt.figure()
    ax = make_wcs_frame(111, projection=projection, center=center,
                        frame="galactic", fig=fig, grid=True)
    assert isinstance(ax, WCSAxes)
    assert len(ax.lines) > 0  # overlay gridlines and/or boundary outline


def test_hpx_polar_cap_gridlines_break_at_notches():
    """HPX polar-cap parallels (|lat|>45) must break at the stepped-diamond
    V-notches rather than run straight across them. The seam-break (a segment
    whose midpoint falls outside the boundary is split) inserts interior NaNs
    into those parallels; a continuous projection (CAR) gets none."""
    import numpy as np
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="HPX", center=0, frame="galactic",
                        fig=fig, grid=True)

    def has_interior_nan_cap_parallel(axes):
        found = False
        for ln in axes.lines:
            xd, yd = ln.get_data()
            xd = np.asarray(xd, float)
            yd = np.asarray(yd, float)
            if yd.size < 5 or (np.nanmax(yd) - np.nanmin(yd)) > 6:
                continue  # not a near-constant-lat parallel
            if not (45 < abs(np.nanmean(yd)) < 90):
                continue  # only polar-cap parallels
            nan_idx = np.where(np.isnan(xd))[0]
            if np.any((nan_idx > 0) & (nan_idx < xd.size - 1)):
                found = True
        return found

    assert has_interior_nan_cap_parallel(ax), \
        "HPX polar-cap parallels should be split at the diamond notches"
    plt.close(fig)

    # A continuous projection's parallels never get spuriously broken.
    fig2 = plt.figure()
    ax2 = make_wcs_frame(111, projection="CAR", center=0, frame="galactic",
                         fig=fig2, grid=True)
    assert not has_interior_nan_cap_parallel(ax2)
    plt.close(fig2)


def test_auto_tick_styling_never_fails_frame_build(monkeypatch):
    """Auto-routed tick styling is cosmetic and must NEVER break the frame
    build — e.g. a draw/overlay failure on a strongly divergent projection
    (COP) is swallowed when tick_style is left at 'auto'. An EXPLICIT
    tick_style still surfaces the failure."""
    import skyplothelper.coord_overlay as _ov

    def _boom(*a, **k):
        raise RuntimeError("simulated overlay failure")

    monkeypatch.setattr(_ov, "add_overlay_ticks", _boom)
    # auto (default): COP routes to in-frame, overlay raises → swallowed.
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="COP", center=0, frame="galactic",
                        fig=fig)
    assert isinstance(ax, WCSAxes)
    plt.close(fig)
    # explicit tick_style surfaces the failure instead of hiding it.
    fig2 = plt.figure()
    with pytest.raises(RuntimeError, match="simulated overlay failure"):
        make_wcs_frame(111, projection="COP", center=0, frame="galactic",
                       fig=fig2, tick_style="in_frame")
    plt.close(fig2)


def test_zoomed_oddball_ticks_fall_back_to_native(monkeypatch):
    """Regression: the per-FITS-code in-frame/boundary tick defaults
    assume the whole-sky net is in view, so on a ZOOMED field frame their
    labels scatter off-panel. tick_style='auto' now falls back to native
    (spine) ticks when the frame is zoomed (is_allsky False) — i.e. it does
    NOT attach the overlay; the all-sky frame still does."""
    import skyplothelper.coord_overlay as _ov
    calls = []
    monkeypatch.setattr(_ov, "add_overlay_ticks", lambda *a, **k: calls.append(1))
    fig = plt.figure()
    make_wcs_frame(111, projection="CSC", center=(0, 0), fov_deg=70,
                   frame="galactic", fig=fig)
    assert calls == [], "zoomed oddball should use native ticks (no overlay)"
    plt.close(fig)
    fig2 = plt.figure()
    make_wcs_frame(111, projection="CSC", center=(0, 0), frame="galactic",
                   fig=fig2)
    assert len(calls) > 0, "all-sky oddball should attach the in-frame overlay"
    plt.close(fig2)


def test_zoomed_oddball_skips_allsky_grid_backfill(monkeypatch):
    """Regression: the all-sky gridline backfill bled whole-net segments
    through the over-approximating boundary on a zoomed single cube face
    (worst on QSC). The backfill is now gated to the all-sky frame; zoomed
    frames use astropy's default grid."""
    import skyplothelper.wcs_frame as _wf
    calls = []
    monkeypatch.setattr(_wf, "_backfill_overlay_grid",
                        lambda *a, **k: calls.append(1))
    fig = plt.figure()
    make_wcs_frame(111, projection="QSC", center=(0, 0), fov_deg=70,
                   frame="galactic", fig=fig, grid=True)
    assert calls == [], "zoomed cube should not use the all-sky backfill grid"
    plt.close(fig)
    fig2 = plt.figure()
    make_wcs_frame(111, projection="QSC", center=(0, 0), frame="galactic",
                   fig=fig2, grid=True)
    assert len(calls) == 1, "all-sky cube should use the backfill grid"
    plt.close(fig2)


def test_explicit_in_frame_on_zoomed_clips_labels_to_view():
    """Regression: even with EXPLICIT tick_style='in_frame', a zoomed
    field frame must keep its central-crosshair labels inside the panel
    (clipped to the axes view rectangle) rather than scattering the full-net
    labels across the figure."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="CSC", center=(0, 0), fov_deg=70,
                        frame="galactic", fig=fig, tick_style="in_frame")
    fig.canvas.draw()
    xl, yl = ax.get_xlim(), ax.get_ylim()
    deg = [t for t in ax.texts if "°" in t.get_text()]
    assert len(deg) > 0
    for t in deg:
        x, y = t.get_position()
        assert (min(xl) - 5 <= x <= max(xl) + 5
                and min(yl) - 5 <= y <= max(yl) + 5), \
            f"in-frame label at ({x:.0f},{y:.0f}) is off the zoomed view"
    plt.close(fig)


@pytest.mark.parametrize("fov", [1.0, 2.0, 10.0])
def test_in_frame_ticks_on_single_field_projection(fov):
    """tick_style='in_frame' on a single-field
    projection (TAN with fov_deg) produces field-scale central-crosshair
    labels. Previously the all-sky default graticule (30°/15°) contained no
    values inside a few-degree field, so zero in-frame ticks rendered."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(150.0, 22.0),
                        fov_deg=fov, fig=fig, tick_style="in_frame")
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.texts if t.get_text().strip()]
    lat_labels = [t for t in labels if "°" in t]
    lon_labels = [t for t in labels if "°" not in t]  # RA in hours
    assert lat_labels, f"no latitude labels at fov={fov}: {labels}"
    assert lon_labels, f"no longitude labels at fov={fov}: {labels}"
    # labels stay inside the view rectangle
    xl, yl = ax.get_xlim(), ax.get_ylim()
    for t in ax.texts:
        if not t.get_text().strip():
            continue
        x, y = t.get_position()
        assert (min(xl) - 5 <= x <= max(xl) + 5
                and min(yl) - 5 <= y <= max(yl) + 5)
    plt.close(fig)


def test_in_frame_field_labels_are_distinct():
    """Adaptive label precision: closely-spaced field ticks must not collapse
    to identical strings (a ~1° RA span in decimal hours needs >1 decimal)."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(150.0, 22.0),
                        fov_deg=1.0, fig=fig, tick_style="in_frame")
    fig.canvas.draw()
    lon_labels = [t.get_text() for t in ax.texts
                  if t.get_text().strip() and "°" not in t.get_text()]
    assert len(lon_labels) == len(set(lon_labels)), \
        f"RA labels collapsed to duplicates: {lon_labels}"
    plt.close(fig)


def test_field_tick_values_bounded_for_tiny_spacing():
    """The field tick-value helper bounds its count regardless of how small the
    spacing is (vs astropy's full-domain spacing= which would enumerate ~1e8)."""
    # 50 mas field, ~2e-6° spacing → would be ~1e8 ticks over the full domain.
    vals = _field_tick_values(187.7, 2e-6, 1.4e-5)
    assert 2 <= len(vals) <= 200
    # values straddle the field center
    assert vals.min() <= 187.7 <= vals.max()
    # hard cap honored even for an absurd spacing/span ratio
    assert len(_field_tick_values(0.0, 1e-12, 1.0)) <= 200


def test_clamp_spacing_warns_and_falls_back():
    """A spacing that would enumerate >max_ticks over the domain falls back to
    the auto spacing with a warning; a sensible spacing passes through."""
    # passthrough: 30° over 360° → 12 ticks, no clamp, no warning
    assert _clamp_spacing(30.0, 360.0, "longitude") == 30.0
    # degenerate: 1e-6° over 360° → ~3.6e8 ticks → fall back + warn
    with pytest.warns(UserWarning, match="falling back"):
        out = _clamp_spacing(1e-6, 360.0, "longitude")
    assert out > 1e-6 and 360.0 / out <= 1000


def test_degenerate_fov_does_not_freeze_draw():
    """Regression: a degenerately small fov_deg used to make astropy enumerate
    ~1e8 ticks (lon 0–360°, lat ±90°) at draw and freeze. Field frames now use
    explicit field-extent tick values; the draw completing at all is the signal
    (it hung before the fix)."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(187.7, 12.4),
                        fov_deg=1.4e-5, fig=fig)  # ~50 mas
    fig.canvas.draw()  # would hang before the fix
    assert isinstance(ax, WCSAxes)
    plt.close(fig)


def test_field_graticule_vals_within_extent():
    """The field-aware tick values fall inside the visible field extent."""
    from skyplothelper.coord_overlay import (
        _field_graticule_vals,
        _field_world_extent,
    )
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(150.0, 22.0),
                        fov_deg=4.0, fig=fig)
    fig.canvas.draw()
    ext = _field_world_extent(ax)
    assert ext is not None
    _, _, lat_lo, lat_hi = ext
    lon_v, lat_v = _field_graticule_vals(ax)
    assert lon_v is not None and lat_v is not None
    assert lat_v.min() >= lat_lo - 1e-6
    assert lat_v.max() <= lat_hi + 1e-6
    assert len(lat_v) >= 2
    plt.close(fig)


@pytest.mark.parametrize("alias, expected_proj", [
    ("hammer-aitoff", "AIT"),
    ("Mollweide", "MOL"),
    ("plate_carree", "CAR"),
    ("orthographic", "SIN"),
    ("gnomonic", "TAN"),
])
def test_make_wcs_frame_aliases(alias, expected_proj):
    """Case-insensitive, hyphen/underscore-tolerant alias dispatch."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection=alias, fig=fig)
    assert isinstance(ax, WCSAxes)
    assert ax.wcs.wcs.ctype[0].endswith(expected_proj)


def test_make_wcs_frame_uses_elliptical_for_ait():
    """AIT axes should be created with EllipticalFrame as its frame_class."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    # frame_class is stored on the axes (via WCSAxes initialization)
    assert isinstance(ax.coords.frame, EllipticalFrame)


# ---- clip_to_frame + apply_boundary_labels round-trip ----

def test_clip_to_frame_runs_without_error():
    """Smoke test: build a non-FITS axes, draw, clip — no exception."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="robinson", fig=fig)
    fig.canvas.draw()
    clip_to_frame(ax)


def test_apply_boundary_labels_runs_without_error():
    """Smoke test: build a SFL axes, apply boundary labels — no exception."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="SFL", fig=fig)
    fig.canvas.draw()
    # Suppress astropy's auto labels for coord 1, draw boundary labels instead
    apply_boundary_labels(ax, coord_index=1, lat_values=[-60, -30, 0, 30, 60])


@pytest.mark.parametrize("orient", ["perpendicular", "parallel", "horizontal"])
def test_apply_boundary_labels_orient_modes(orient):
    """Each boundary-relative orient mode runs and emits one label per value."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    fig.canvas.draw()
    labels = apply_boundary_labels(
        ax, coord_index=1, side="left", orient=orient,
        lat_values=[-60, -30, 0, 30, 60])
    assert len(labels) == 5


def test_apply_boundary_labels_rejects_old_names():
    """The old 'tangent'/'radial'/'extension' orient names are gone — clean
    rename (no alias shims) per the pre-release API policy."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    fig.canvas.draw()
    for bad in ("tangent", "radial", "extension"):
        with pytest.raises(ValueError, match="perpendicular"):
            apply_boundary_labels(ax, orient=bad)


def test_apply_boundary_labels_near_edge_anchor_centers_on_tick():
    """Labels anchor by their near edge (via _resolve_text_anchor), so a
    single mid-edge label sits centered on its gridline rather than shifted
    tangentially. Check the bbox center's vertical offset from the boundary
    tick is small relative to the label height (the old fixed-ha bug shifted
    near-vertical 'parallel' text by ~half its width along the edge)."""
    fig = plt.figure(figsize=(6, 4))
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Equator label on the right edge: boundary tangent is vertical there,
    # so 'parallel' text is near-vertical — the worst case for the old bug.
    labels = apply_boundary_labels(
        ax, coord_index=1, side="right", orient="parallel",
        lat_values=[0], pad=4)
    assert len(labels) == 1
    bbox = labels[0].get_window_extent(renderer=renderer)
    # The label's vertical center should land near the axes' vertical center
    # (the equator on an all-sky frame), not be shifted up/down by ~half the
    # text length the way fixed va='center' + end-anchored ha did.
    ax_bbox = ax.get_window_extent()
    ax_cy = 0.5 * (ax_bbox.y0 + ax_bbox.y1)
    assert abs(bbox.y0 + 0.5 * bbox.height - ax_cy) < bbox.height


# ---- Hybrid lat-overlay path for EllipticalFrame (AIT / MOL) ----

def _collect_text_artists_under_axes(ax):
    """Return the visible Text artists rendered under *ax* — both
    astropy coord ticklabels and any skyplothelper overlay labels
    added via add_overlay_ticks."""
    import matplotlib.text as mtext
    out = []
    for child in ax.get_children():
        if isinstance(child, mtext.Text) and child.get_visible():
            t = child.get_text()
            if t and t.strip():
                out.append(t)
    return out


@pytest.mark.parametrize("projection", ["AIT", "MOL"])
def test_elliptical_default_routes_through_hybrid_lat_overlay(projection):
    """Under the default ``tick_style='auto'``, AIT and MOL get
    overlay-rendered lat labels via the hybrid path (workaround for
    astropy's NaN tangent-angle bug on the ellipse 'c' spine that
    drops most lat tick marks)."""
    fig = plt.figure(figsize=(6, 4))
    ax = make_wcs_frame(111, projection=projection, center=0, fig=fig)
    fig.canvas.draw()

    texts = _collect_text_artists_under_axes(ax)
    # Hybrid lat overlay renders labels at -60, -30, 0, 30, 60.
    # Each value appears on both side edges → expect each at least
    # twice in the visible text set.
    for v in ('60°', '30°', '0°'):
        count = sum(1 for t in texts if v in t)
        assert count >= 2, (
            f"expected at least 2 overlay labels containing {v!r} "
            f"on {projection}, saw {count} in {texts}")
    plt.close(fig)


def test_elliptical_explicit_native_is_escape_hatch():
    """Explicit ``tick_style='native'`` bypasses the hybrid path —
    users who want the bare astropy default (with its tick-rendering
    bug) can opt out via the kwarg."""
    fig = plt.figure(figsize=(6, 4))
    ax = make_wcs_frame(111, projection="AIT", center=0, fig=fig,
                        tick_style='native')
    fig.canvas.draw()
    # The hybrid would have suppressed astropy's lat ticks via
    # ``suppress_default='lat'``; explicit native skips that, so
    # ``coord_ticks(ax.coords[1])_visible`` stays True (and the hybrid's
    # overlay labels are absent). The most reliable assertion is the
    # absence of the hybrid-side-effect attribute that
    # add_overlay_ticks adds.
    # No skyplothelper CoordinateOverlay attached → no overlay marker.
    assert not any(
        type(c).__module__.startswith('skyplothelper.coord_overlay')
        for c in ax.get_children()
    )
    plt.close(fig)


def test_non_elliptical_is_unaffected_by_hybrid_path():
    """The hybrid path is gated by ``frame_shape=='elliptical'`` —
    rectangular projections (e.g. CAR) and others are routed normally."""
    fig = plt.figure(figsize=(6, 4))
    ax = make_wcs_frame(111, projection="CAR", center=0, fig=fig)
    fig.canvas.draw()
    # CAR has frame_shape='rectangular' — not in the hybrid set, so
    # no overlay artists from add_overlay_ticks were attached.
    assert not any(
        type(c).__module__.startswith('skyplothelper.coord_overlay')
        for c in ax.get_children()
    )
    plt.close(fig)


# ---- direction (longitude orientation) ----

def _cdelt1(**kw):
    res = make_wcs_frame(return_hdr=True, **kw)
    hdr = res[-1] if isinstance(res, (tuple, list)) else res
    return float(hdr["CDELT1"])


@pytest.mark.parametrize("kw", [
    {}, dict(fov_deg=10), dict(projection="SIN"),
    dict(frame="Galactic"), dict(frame="Galactic", fov_deg=10),
    dict(frame="Galactic", projection="SIN"),
])
def test_make_wcs_frame_direction_uniform(kw):
    """direction='sky' puts lon increasing left (CDELT1 < 0) and
    'geographic' increasing right (CDELT1 > 0), uniformly across all-sky /
    field / globe and regardless of frame (fixes the old galactic-field
    inconsistency where galactic defaulted to geographic)."""
    assert _cdelt1(direction="sky", **kw) < 0
    assert _cdelt1(direction="geographic", **kw) > 0


def test_make_wcs_frame_direction_default_is_sky():
    assert _cdelt1() < 0  # default 'sky' → left


def test_make_wcs_frame_direction_alias():
    assert _cdelt1(direction="astro") == _cdelt1(direction="sky")
    assert _cdelt1(direction="geo") == _cdelt1(direction="geographic")


@pytest.mark.parametrize("projection", [
    "robinson", "winkel_tripel", "eckert_iv", "kavrayskiy", "mcbryde",
])
def test_make_wcs_frame_non_fits_direction_orientation(projection):
    """Custom (non-FITS) projections must honor ``direction`` like the FITS
    path: 'sky' (default) puts increasing longitude / east to the LEFT,
    'geographic' to the right. The custom CurvedTransforms have no CDELT1, so
    check the rendered orientation through the axes' world transform
    (regression for the custom path silently ignoring ``direction``)."""
    for direction, east_is_left in [("sky", True), ("geographic", False)]:
        fig = plt.figure()
        ax = make_wcs_frame(111, projection=projection, center=0,
                            frame="galactic", direction=direction, fig=fig)
        t = ax.get_transform("world")
        # display-space x of an eastern (+45°) vs western (-45°) equator point
        (xe, _), (xw, _) = t.transform([[45.0, 0.0], [-45.0, 0.0]])
        if east_is_left:
            assert xe < xw, f"{projection}/{direction}: east should be left"
        else:
            assert xe > xw, f"{projection}/{direction}: east should be right"
        plt.close(fig)


# ---- all-sky center_lat (oblique aspect) ----

@pytest.mark.parametrize("projection", ["AIT", "MOL", "SFL", "PAR", "CAR",
                                        "MER", "CEA", "CYP", "BON", "PCO"])
def test_make_wcs_frame_allsky_honors_center_lat(projection):
    """Standard all-sky FITS projections honor center_lat via CRVAL2 (oblique
    aspect); default is still equatorial (CRVAL2=0)."""
    res = make_wcs_frame(return_hdr=True, projection=projection,
                         center=(45.0, 30.0))
    hdr = res[-1]
    assert float(hdr["CRVAL2"]) == pytest.approx(30.0)
    # default (no center_lat) stays equatorial
    res0 = make_wcs_frame(return_hdr=True, projection=projection, center=45.0)
    assert float(res0[-1]["CRVAL2"]) == pytest.approx(0.0)


@pytest.mark.parametrize("projection", ["HPX", "XPH", "TSC", "CSC", "QSC"])
def test_make_wcs_frame_healpix_cube_locked_to_equatorial(projection):
    """HEALPix/quadcube tilings are pole-relative, so center_lat is ignored
    (CRVAL2 stays 0) even when requested."""
    res = make_wcs_frame(return_hdr=True, projection=projection,
                         center=(45.0, 30.0))
    assert float(res[-1]["CRVAL2"]) == pytest.approx(0.0)


@pytest.mark.parametrize("name", [
    "robinson", "winkel_tripel", "eckert_iv", "kavrayskiy", "mcbryde"])
def test_compromise_projections_auto_route_to_in_frame_ticks(name):
    """The five custom compromise projections route to in-frame ticks under
    tick_style='auto' (their frame_shape is in the in-frame auto set). Guards
    the eckert_iv frame_shape-key mismatch ('eckert4' vs 'eckert_iv') that
    silently dropped it to native (frame-spine) ticks + verbose labels."""
    from skyplothelper.projections.registry import _resolve_projection
    from skyplothelper.wcs_frame import _IN_FRAME_TICK_AUTO_FRAME_SHAPES
    _, info = _resolve_projection(name)
    assert info.frame_shape in _IN_FRAME_TICK_AUTO_FRAME_SHAPES, (
        f"{name} (frame_shape={info.frame_shape!r}) not in the in-frame set")


def test_boundary_tick_curve_traces_true_edge_for_oddballs():
    """``_boundary_tick_curve`` returns the projection's true visible edge for
    the interrupted / non-rectangular all-sky projections (so boundary ticks
    follow the diamond / wedge / egg / cross, not the canvas rectangle), and
    ``None`` for standard projections whose astropy spine already traces it."""
    from skyplothelper.wcs_frame import _boundary_tick_curve
    for proj in ["HPX", "BON", "PCO", "COD", "COE", "TSC", "QSC", "XPH"]:
        ax = make_wcs_frame(111, projection=proj, center=0, frame="galactic")
        assert _boundary_tick_curve(ax, proj) is not None, proj
        plt.close("all")
    for proj in ["MOL", "SFL", "CAR"]:
        ax = make_wcs_frame(111, projection=proj, center=0, frame="galactic")
        assert _boundary_tick_curve(ax, proj) is None, proj
        plt.close("all")


@pytest.mark.parametrize("projection", ["BON", "PCO"])
@pytest.mark.parametrize("center", [(0, 0), (60, 30)])
def test_bon_pco_data_clipped_to_nan_edge_envelope(projection, center):
    """BON cardioid / PCO egg clip their DATA to the NaN-edge visible-region
    envelope (so pcolormesh cells can't bridge the concave top/bottom notch),
    and this works in the oblique aspect too — where the analytic lon=CRVAL±180
    boundary would be wrong. Verifies the data clip installs a (non-rectangular)
    clip path via the public helper."""
    from skyplothelper.wcs_frame import _projection_boundary, clip_to_projection_boundary
    fig = plt.figure()
    ax = make_wcs_frame(111, projection=projection, center=center,
                        frame="galactic", fig=fig)
    # the routed boundary is the NaN-edge limb (a many-vertex contour),
    # not the 2-sided analytic meridian curve.
    path = _projection_boundary(ax)
    assert path is not None and len(path.vertices) > 50, projection
    ln, = ax.plot([10, 200], [10, -10], transform=ax.get_transform("world"))
    before = ln.get_clip_path()
    clip_to_projection_boundary(ax, ln)
    assert ln.get_clip_path() is not before, projection
    plt.close(fig)


@pytest.mark.parametrize("projection", ["TSC", "CSC", "QSC"])
def test_make_wcs_frame_cube_frames_net_without_empty_arm(projection):
    """The quadcube net is offset in x, so the frame uses an asymmetric
    x-extent (shifted CRPIX1) to bound the cross tightly — the kapteyn
    T-shape. A symmetric frame left a large empty arm; verify the projected
    sphere now fills most of the frame width."""
    import numpy as np
    fig = plt.figure()
    ax = make_wcs_frame(111, projection=projection, center=0,
                        frame="galactic", fig=fig)
    nx = 360
    # CRPIX1 is shifted off the panel center (the symmetric placement).
    assert abs(float(ax.wcs.wcs.crpix[0]) - (nx / 2 + 0.5)) > 20.0
    # The projected sphere fills most of the frame width (no big empty arm).
    lon = np.linspace(-180, 180, 361)
    lat = np.linspace(-89.9, 89.9, 181)
    LL, BB = np.meshgrid(lon, lat)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        px, _ = ax.wcs.world_to_pixel_values(LL.ravel(), BB.ravel())
    px = px[np.isfinite(px)]
    px = px[(px >= -0.5) & (px <= nx - 0.5)]
    occupied_frac = (px.max() - px.min()) / nx
    assert occupied_frac > 0.9, f"{projection}: net fills only {occupied_frac:.0%}"
    plt.close(fig)


@pytest.mark.parametrize("projection", ["COD", "COE", "COO", "COP"])
def test_make_wcs_frame_conic_centers_on_standard_parallel(projection):
    """All-sky conics center the reference point on the standard parallel
    (CRVAL2 = PV2_1, the kapteyn recipe), so the apex sits at the pole and
    the wedge frames cleanly. center_lat is ignored (the aspect is fixed by
    the standard parallel); the PV2_1 default is 45, an explicit value wins."""
    # default PV2_1=45: CRVAL2 follows it, not the requested center_lat=30
    res = make_wcs_frame(return_hdr=True, projection=projection,
                         center=(45.0, 30.0))
    assert float(res[-1]["CRVAL2"]) == pytest.approx(45.0)
    # explicit standard parallel drives CRVAL2
    res2 = make_wcs_frame(return_hdr=True, projection=projection,
                          center=45.0, pv2_1=60.0)
    assert float(res2[-1]["CRVAL2"]) == pytest.approx(60.0)
    plt.close("all")  # return_hdr still builds axes — don't leak figures


@pytest.mark.parametrize("projection", ["COD", "COE", "COO", "COP"])
def test_make_wcs_frame_conic_clips_data_to_wedge(projection):
    """All-sky conics now have a reliable wedge boundary, so the data clip
    installs a clip path (data no longer bleeds past the wedge into the
    bbox corners)."""
    from skyplothelper.wcs_frame import clip_to_projection_boundary
    fig = plt.figure()
    ax = make_wcs_frame(111, projection=projection, center=0,
                        frame="galactic", fig=fig)
    ln, = ax.plot([10, 200], [10, -10], transform=ax.get_transform("world"))
    before = ln.get_clip_path()
    clip_to_projection_boundary(ax, ln)
    assert ln.get_clip_path() is not before  # conic wedge clip installed
    plt.close(fig)


@pytest.mark.parametrize("projection", ["COO", "COP"])
def test_conic_overlay_gridlines_stay_within_visible_wedge(projection):
    """COO/COP drop the divergent far-pole cap, so the backfill overlay
    gridlines must not flare past the clipped wedge — no rendered gridline
    sample may sit below the visible latitude bound (previously they leaked
    down to the pole because the pixel-polygon mask mis-clipped the wedge)."""
    import numpy as np

    from skyplothelper.projections._boundaries import (
        _conic_pv2_1,
        conic_visible_lat_range,
    )
    fig = plt.figure()
    ax = make_wcs_frame(111, projection=projection, center=0,
                        frame="galactic", fig=fig, grid=True)
    fig.canvas.draw()
    lat_lo, lat_hi = conic_visible_lat_range(projection, _conic_pv2_1(ax))
    lat_min = 90.0
    for ln in ax.lines:
        if ln.get_linestyle() not in (":", "dotted"):
            continue
        _, yd = ln.get_data()
        yd = np.asarray(yd, dtype=float)
        yd = yd[np.isfinite(yd)]
        if yd.size:
            lat_min = min(lat_min, float(yd.min()))
    # Allow a small margin (one sample step) below the bound.
    assert lat_min >= lat_lo - 2.0
    plt.close(fig)


@pytest.mark.parametrize("projection", ["COD", "COE"])
def test_full_sphere_conic_gridlines_reach_far_latitudes(projection):
    """COD/COE span the full sphere, so the wedge-clip fix must NOT over-clip
    them — their gridlines still reach well past the COO/COP -30 cap."""
    import numpy as np

    fig = plt.figure()
    ax = make_wcs_frame(111, projection=projection, center=0,
                        frame="galactic", fig=fig, grid=True)
    fig.canvas.draw()
    lat_min = 90.0
    for ln in ax.lines:
        if ln.get_linestyle() not in (":", "dotted"):
            continue
        _, yd = ln.get_data()
        yd = np.asarray(yd, dtype=float)
        yd = yd[np.isfinite(yd)]
        if yd.size:
            lat_min = min(lat_min, float(yd.min()))
    assert lat_min < -50.0  # full-sphere wedge, not clipped to -30
    plt.close(fig)


@pytest.mark.parametrize("projection", [
    "robinson", "winkel_tripel", "eckert_iv", "kavrayskiy", "mcbryde",
])
def test_make_wcs_frame_non_fits_oblique_centers_point(projection):
    """Custom (non-FITS) projections gain center_lat via an oblique spherical
    rotation: the requested center (lon0, lat0) must land at the projection
    origin (the panel's data-space center)."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection=projection, center=(0.0, 30.0),
                        frame="galactic", fig=fig)
    # world (0, 30) -> projected origin (0, 0); data-space center is (0, 0)
    # since xlim/ylim are symmetric about 0.
    xc, yc = ax.transData.inverted().transform(
        ax.get_transform("world").transform([[0.0, 30.0]]))[0]
    assert xc == pytest.approx(0.0, abs=1e-6)
    assert yc == pytest.approx(0.0, abs=1e-6)
    plt.close(fig)


# ---- lon_units ----

def _lon_unit(**kw):
    ax = make_wcs_frame(**kw)
    return str(ax.coords[0].get_format_unit())


def test_make_wcs_frame_lon_units_auto_equatorial_is_hours():
    assert _lon_unit(frame="ICRS") == "hourangle"          # sky default


def test_make_wcs_frame_lon_units_degrees_override():
    """Explicit lon_units='degrees' forces degrees even on a sky ICRS frame
    (the wide-field 'RA in degrees' use case)."""
    assert _lon_unit(frame="ICRS", lon_units="degrees") == "deg"
    assert _lon_unit(frame="ICRS", lon_units="deg") == "deg"       # alias


def test_make_wcs_frame_lon_units_hours_override():
    assert _lon_unit(frame="Galactic", lon_units="hours") == "hourangle"


def test_make_wcs_frame_lon_units_invalid_raises():
    with pytest.raises(ValueError, match="lon_units"):
        make_wcs_frame(lon_units="radians")


# ---- SIN globe sizing: full hemisphere, fov_deg zoom, physical clamp ----

def test_sin_globe_fills_frame_full_hemisphere():
    """A SIN globe (no fov_deg) fills the circular frame to the limb — CDELT =
    hemisphere radius (180/pi) / half-npix — and uses a full spacing-based
    graticule, not shrunken field-extent tick values."""
    import numpy as np
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="SIN", center=(60, 45), fig=fig)
    nx = ax.wcs.pixel_shape[0]
    expected = 2 * (180.0 / np.pi) / nx
    assert abs(abs(ax.wcs.wcs.cdelt[1]) - expected) < 1e-9
    # spacing-based ticks (full graticule), not explicit field values
    assert ax.coords[1]._formatter_locator.spacing is not None
    plt.close(fig)


def test_sin_globe_fov_deg_zooms_sin_scaled():
    """fov_deg zooms a SIN globe via the orthographic r=sin(theta) relation, so
    the fov-diameter cap fills the frame (not the linear field-frame scaling)."""
    import numpy as np
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="SIN", center=(60, 45),
                        fov_deg=120, fig=fig)
    nx = ax.wcs.pixel_shape[0]
    expected = 2 * np.sin(np.radians(120 / 2)) * (180.0 / np.pi) / nx
    assert abs(abs(ax.wcs.wcs.cdelt[1]) - expected) < 1e-9
    # a smaller fov is more zoomed in (smaller CDELT) than the full hemisphere
    fig2 = plt.figure()
    ax_full = make_wcs_frame(111, projection="SIN", center=(60, 45),
                             fov_deg=180, fig=fig2)
    assert abs(ax.wcs.wcs.cdelt[1]) < abs(ax_full.wcs.wcs.cdelt[1])
    plt.close(fig)
    plt.close(fig2)


def test_sin_globe_fov_deg_clamps_to_hemisphere():
    """fov_deg beyond the physical hemisphere (180 deg diameter) clamps to 180
    with a warning."""
    import numpy as np
    fig = plt.figure()
    with pytest.warns(UserWarning, match="clamping to 180"):
        ax = make_wcs_frame(111, projection="SIN", center=(60, 45),
                            fov_deg=300, fig=fig)
    nx = ax.wcs.pixel_shape[0]
    hemi = 2 * (180.0 / np.pi) / nx
    assert abs(abs(ax.wcs.wcs.cdelt[1]) - hemi) < 1e-9
    plt.close(fig)


def test_edge_ticks_auto_pins_single_field_spines():
    """Default edge_ticks='auto' keeps lon ticks on bottom/top and lat ticks
    on left/right, removing the stray Dec-on-bottom tick astropy's per-spine
    heuristic otherwise places when meridians converge."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(83.6, 22.0),
                        fov_deg=4.0, fig=fig, grid=False)
    fig.canvas.draw()
    assert set(coord_ticks(ax.coords[0]).get_visible_axes()) == {"b", "t"}
    assert set(coord_ticks(ax.coords[1]).get_visible_axes()) == {"l", "r"}
    plt.close(fig)


def test_edge_ticks_all_keeps_astropy_heuristic():
    """edge_ticks='all' leaves astropy's automatic per-spine assignment — for
    this field that scatters a lat tick onto the bottom spine."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="TAN", center=(83.6, 22.0),
                        fov_deg=4.0, fig=fig, grid=False, edge_ticks="all")
    fig.canvas.draw()
    # The stray the 'auto' default suppresses: lat on the bottom spine.
    assert "b" in coord_ticks(ax.coords[1]).get_visible_axes()
    plt.close(fig)


def test_edge_ticks_does_not_touch_allsky():
    """All-sky frames keep their own tick handling regardless of edge_ticks."""
    fig = plt.figure()
    ax = make_wcs_frame(111, projection="AIT", fig=fig)
    fig.canvas.draw()
    # 'auto' must not pin an all-sky frame to the rectangular b/t/l/r spines.
    assert set(coord_ticks(ax.coords[0]).get_visible_axes()) != {"b", "t"}
    plt.close(fig)


def test_edge_ticks_invalid_raises():
    with pytest.raises(ValueError, match="edge_ticks must be"):
        make_wcs_frame(111, projection="TAN", center=(83.6, 22.0),
                       fov_deg=4.0, edge_ticks="nope")


# ---- curved frames suppress base-style minor ticks ----

@pytest.mark.parametrize("proj,kw", [
    ("SIN", dict(center=(180, 30), tick_style="native")),
    ("ZEA", dict(center=(180, 30), tick_style="native")),
    ("robinson", dict(center=180)),  # non-FITS all-sky path
])
def test_curved_frames_suppress_base_style_minor_ticks(proj, kw):
    """Under a minor-tick base style (structural sets xtick.minor.visible=True,
    which native WCSAxes honor), curved all-sky / globe frames suppress the
    minor ticks — astropy scatters them into a dense central row + all-around
    limb on a curved spine. (conftest restores rcParams after the test.)"""
    sph.set_base_style("structural")
    ax = make_wcs_frame(111, projection=proj, fig=plt.figure(), **kw)
    ax.figure.canvas.draw()
    assert not coord_ticks(ax.coords[0]).get_display_minor_ticks(), proj
    assert not coord_ticks(ax.coords[1]).get_display_minor_ticks(), proj


def test_flat_field_keeps_base_style_minor_ticks():
    """The companion: a flat field frame KEEPS minor ticks under structural —
    there they are useful subdivisions, only the curved spines are cluttered."""
    sph.set_base_style("structural")
    ax = make_wcs_frame(111, projection="TAN", center=(83.6, 22.0),
                        fov_deg=4.0, fig=plt.figure())
    ax.figure.canvas.draw()
    assert coord_ticks(ax.coords[0]).get_display_minor_ticks()
    assert coord_ticks(ax.coords[1]).get_display_minor_ticks()


# --- grid line width / style are reachable on every branch ------------------

def _grid_render(**kw):
    """Pixels: grid lines are not axes children on the WCSAxes paths, so
    inspecting the artist tree finds nothing and a working knob looks broken.
    """
    import numpy as np
    fig = plt.figure(figsize=(4, 4), dpi=100)
    sph.make_wcs_frame(fig=fig, **kw)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return buf


def _grid_diff(a, b):
    import numpy as np
    return int((np.abs(a.astype(int) - b.astype(int)).sum(axis=2) > 8).sum())


# The three grid paths make_wcs_frame can take:
#   AIT -> non-FITS all-sky matplotlib branch  (historical ls=':')
#   TAN -> plain WCSAxes coords branch         (historical: inherits both)
#   HPX -> densified backfill overlay          (historical lw=0.5, ls=':')
_GRID_BRANCHES = [
    pytest.param(dict(projection="AIT"), id="allsky-mpl"),
    pytest.param(dict(projection="TAN", fov_deg=2.0), id="wcsaxes-coords"),
    pytest.param(dict(projection="HPX"), id="backfill-overlay"),
]


@pytest.mark.parametrize("frame_kw", _GRID_BRANCHES)
def test_grid_defaults_are_unchanged_by_the_new_knobs(frame_kw):
    """gridlw/gridls only make the properties reachable. The three branches
    keep their different historical looks on purpose — agreeing on one would
    move existing renders on whichever branch lost."""
    ref = _grid_render(**frame_kw)
    explicit = _grid_render(**frame_kw, gridlw=None, gridls=None)
    assert _grid_diff(ref, explicit) == 0


@pytest.mark.parametrize("frame_kw", _GRID_BRANCHES)
@pytest.mark.parametrize("knob,value", [("gridlw", 3.0), ("gridls", "--")])
def test_grid_knobs_reach_every_branch(frame_kw, knob, value):
    ref = _grid_render(**frame_kw)
    assert _grid_diff(ref, _grid_render(**frame_kw, **{knob: value})) > 50
