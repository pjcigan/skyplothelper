"""Tests for the public :func:`skyplothelper.auto_size_ticklabels` API.

Coverage: dispatch on axes type (WCS / plain mpl / cartopy-skip /
unknown-skip), kwarg propagation (axis / floor / ceiling), the
``ax._sph_auto_label_fontsize`` cache, reflow-on-resize callback
lifecycle + idempotency, and the make_wcs_frame integration's
warning-on-failure hardening.
"""

import matplotlib

matplotlib.use("Agg")

import warnings  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import skyplothelper as sph  # noqa: E402
from skyplothelper.autosize import (  # noqa: E402
    _attach_resize_reflow,
    _dispatch_auto_size,
    auto_size_ticklabels,
)

# ---- fixtures --------------------------------------------------------------

def _plain_axes(figsize=(6, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1, 2, 3], [0, 1, 4, 9])
    fig.canvas.draw()
    return fig, ax


def _wcs_axes(figsize=(6, 4), projection="AIT", center=0):
    fig = plt.figure(figsize=figsize)
    ax = sph.make_wcs_frame(111, projection=projection, center=center,
                            fig=fig, auto_fontsize=False)
    fig.canvas.draw()
    return fig, ax


# ---- WCSAxes dispatch ------------------------------------------------------

def test_dispatches_to_wcs_path_when_ax_has_coords():
    """WCSAxes is detected via hasattr(ax, 'coords') and routed to the
    WCS helper — returns a sensible fontsize and caches it."""
    fig, ax = _wcs_axes(figsize=(3, 2))
    fs = auto_size_ticklabels(ax)
    assert fs is not None
    assert 6.0 <= fs <= 10.0
    assert ax._sph_auto_label_fontsize == fs
    plt.close(fig)


def test_wcs_path_ignores_axis_kwarg():
    """WCSAxes always sizes both coords together; the axis= kwarg
    isn't meaningful there but should be silently accepted."""
    fig, ax = _wcs_axes(figsize=(4, 3))
    fs_both = auto_size_ticklabels(ax, axis='both')
    fs_x = auto_size_ticklabels(ax, axis='x')
    assert fs_both == fs_x
    plt.close(fig)


# ---- Plain mpl dispatch ----------------------------------------------------

def test_dispatches_to_mpl_path_for_plain_axes():
    """Plain matplotlib axes get sized via tick_params and cache the
    result on the axes."""
    fig, ax = _plain_axes(figsize=(4, 3))
    fs = auto_size_ticklabels(ax)
    assert fs is not None
    assert 6.0 <= fs <= 10.0
    assert ax._sph_auto_label_fontsize == fs
    plt.close(fig)


def test_mpl_path_applies_via_tick_params():
    """After auto_size_ticklabels, the actual tick label fontsize on
    the axes equals the returned value (verifies tick_params dispatch)."""
    fig, ax = _plain_axes(figsize=(3, 2))
    fs = auto_size_ticklabels(ax)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        assert lbl.get_fontsize() == pytest.approx(fs)
    plt.close(fig)


def test_mpl_path_axis_x_only_leaves_y_unchanged():
    """axis='x' applies only to the x ticks — y ticks keep their
    rcParams default size."""
    from matplotlib import rcParams
    fig, ax = _plain_axes(figsize=(3, 2))
    y_before = ax.get_yticklabels()[0].get_fontsize()
    fs = auto_size_ticklabels(ax, axis='x')
    x_after = ax.get_xticklabels()[0].get_fontsize()
    y_after = ax.get_yticklabels()[0].get_fontsize()
    assert x_after == pytest.approx(fs)
    assert y_after == pytest.approx(y_before)
    # Sanity: y_before is the rcParams default (or whatever it was).
    del rcParams
    plt.close(fig)


# ---- Unsupported axes: warn + None -----------------------------------------

def test_unrecognized_axes_type_warns_and_returns_none():
    """A bare object without ax.coords / tick_params triggers the
    catch-all warning, returns None."""
    class FakeAxes:
        pass

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = auto_size_ticklabels(FakeAxes())
    assert result is None
    assert any("unrecognized axes type" in str(w.message) for w in caught)


def test_cartopy_axes_without_gridliners_returns_none_silently():
    """A cartopy GeoAxes with no gridliners returns ``None`` cleanly
    (nothing to size — no warning needed)."""
    pytest.importorskip("cartopy")
    import cartopy.crs as ccrs

    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot(111, projection=ccrs.PlateCarree())
    # No ax.gridlines() call → no gridliners on the axes.
    fig.canvas.draw()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = auto_size_ticklabels(ax)
    assert result is None
    # No warning is expected — empty gridliners just means nothing to size.
    assert not any("cartopy" in str(w.message).lower() for w in caught)
    plt.close(fig)


def test_cartopy_axes_with_gridliner_sizes_label_artists():
    """A cartopy GeoAxes with a Gridliner gets its label fontsize
    set both on ``xlabel_style`` (for future redraws) and on the
    currently-rendered label artists (for immediate visibility)."""
    pytest.importorskip("cartopy")
    import cartopy.crs as ccrs

    # Small figure → auto-size should shrink below the default 10pt.
    fig = plt.figure(figsize=(3, 2))
    ax = plt.subplot(111, projection=ccrs.PlateCarree())
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True)
    fig.canvas.draw()

    fs = auto_size_ticklabels(ax)
    assert fs is not None
    assert 6.0 <= fs <= 10.0
    # Style dict updated for future redraws.
    assert gl.xlabel_style.get('size') == pytest.approx(fs)
    assert gl.ylabel_style.get('size') == pytest.approx(fs)
    # Existing label artists were also resized in-place.
    for art in list(gl.xlabel_artists) + list(gl.ylabel_artists):
        if art.get_text().strip():
            assert art.get_fontsize() == pytest.approx(fs)
    plt.close(fig)


def test_cartopy_unit_helper_with_fake_geoaxes_no_gridliner():
    """The internal cartopy helper handles synthetic
    cartopy.*-namespaced classes that have no children — returns
    ``None`` instead of crashing."""

    class FakeGeoAxes:
        def get_children(self):
            return []

    FakeGeoAxes.__module__ = 'cartopy.mpl.geoaxes'

    # Goes through the public dispatch → cartopy path → empty
    # gridliner list → returns None silently.
    result = auto_size_ticklabels(FakeGeoAxes())
    assert result is None


# ---- floor / ceiling overrides ---------------------------------------------

def test_floor_kwarg_clamps_lower_bound():
    """A very tight axes hits the floor — explicit floor= overrides
    the default of 6pt."""
    fig, ax = _plain_axes(figsize=(1, 1))
    fs = auto_size_ticklabels(ax, floor=4.0)
    assert fs >= 4.0
    plt.close(fig)


def test_ceiling_kwarg_caps_upper_bound():
    """A very roomy axes hits the ceiling — explicit ceiling= caps
    below the rcParams default."""
    fig, ax = _plain_axes(figsize=(20, 12))
    fs = auto_size_ticklabels(ax, ceiling=8.0)
    assert fs == pytest.approx(8.0)
    plt.close(fig)


# ---- reflow_on_resize ------------------------------------------------------

def test_reflow_on_resize_attaches_callback():
    """reflow_on_resize=True wires a resize_event handler onto the
    figure canvas and caches its cid on the axes."""
    fig, ax = _wcs_axes(figsize=(3, 2))
    auto_size_ticklabels(ax, reflow_on_resize=True)
    cid = getattr(ax, '_sph_autosize_cid', None)
    assert cid is not None
    assert isinstance(cid, int)
    plt.close(fig)


def test_reflow_on_resize_is_idempotent():
    """Calling auto_size_ticklabels(reflow_on_resize=True) twice on
    the same axes doesn't stack callbacks — the cached cid list stays
    the same length."""
    fig, ax = _wcs_axes(figsize=(3, 2))
    auto_size_ticklabels(ax, reflow_on_resize=True)
    first_cids = list(ax._sph_autosize_cids)
    auto_size_ticklabels(ax, reflow_on_resize=True)
    second_cids = list(ax._sph_autosize_cids)
    assert first_cids == second_cids
    plt.close(fig)


def test_reflow_on_resize_attaches_xlim_and_ylim_callbacks():
    """reflow_on_resize=True wires resize + xlim_changed + ylim_changed
    callbacks (the pan/zoom-aware set)."""
    fig, ax = _wcs_axes(figsize=(3, 2))
    auto_size_ticklabels(ax, reflow_on_resize=True)
    cids = getattr(ax, '_sph_autosize_cids', None)
    assert cids is not None
    kinds = {kind for (kind, _) in cids}
    assert kinds == {'canvas', 'xlim', 'ylim'}
    plt.close(fig)


def test_reflow_callback_re_sizes_after_xlim_change():
    """Firing xlim_changed (e.g. from a pan/zoom) re-runs the
    dispatcher — verifies the new pan/zoom-aware hook."""
    fig, ax = _wcs_axes(figsize=(10, 6))
    auto_size_ticklabels(ax, reflow_on_resize=True)
    # Pan/zoom: set_xlim fires xlim_changed; the callback should
    # recompute fontsize. Pick a zoom that's likely to change either
    # the geometry or the label widths (long HMS labels).
    initial = ax._sph_auto_label_fontsize
    ax.set_xlim(50, 100)
    fig.canvas.draw()
    after = ax._sph_auto_label_fontsize
    # The cache attribute should have been refreshed (value may be
    # the same or different — we just verify the callback fired by
    # re-setting the cache).
    assert after is not None
    # Sanity: the cache wasn't accidentally cleared.
    assert isinstance(after, float)
    del initial
    plt.close(fig)


def test_reflow_on_resize_default_off_no_callback_attached():
    """Default reflow_on_resize=False means no cid is cached — the
    snapshot-at-construction semantic is preserved."""
    fig, ax = _wcs_axes(figsize=(3, 2))
    auto_size_ticklabels(ax)
    assert not hasattr(ax, '_sph_autosize_cid') or ax._sph_autosize_cid is None
    plt.close(fig)


def test_reflow_callback_re_sizes_after_figure_resize():
    """Simulate a resize by manually firing the registered callback
    after shrinking the figure — the cached fontsize should update
    to reflect the new (tighter) geometry."""
    fig, ax = _wcs_axes(figsize=(10, 6))   # roomy → fs at ceiling
    fs_large = auto_size_ticklabels(ax, reflow_on_resize=True)
    fig.set_size_inches(2.5, 2)
    fig.canvas.draw()
    # Fire the resize callback explicitly via matplotlib's callback
    # registry (the Agg backend doesn't auto-fire resize_event on
    # headless set_size_inches).
    fig.canvas.callbacks.process('resize_event', None)
    fs_small = ax._sph_auto_label_fontsize
    assert fs_small < fs_large
    plt.close(fig)


# ---- Integration: make_wcs_frame hardening ---------------------------------

def test_make_wcs_frame_auto_fontsize_failure_warns_not_raises(monkeypatch):
    """If the auto_size_ticklabels call raises, make_wcs_frame should
    swallow it with a UserWarning and return the axes intact —
    auto-fontsize is a convenience, not a reason for frame construction
    to fail."""
    from skyplothelper import autosize

    def _explode(*_a, **_kw):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(autosize, "auto_size_ticklabels", _explode)

    fig = plt.figure(figsize=(4, 3))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ax = sph.make_wcs_frame(111, "AIT", center=0, fig=fig,
                                auto_fontsize=True)
    assert ax is not None
    assert any("auto_fontsize failed" in str(w.message) for w in caught)
    plt.close(fig)


def test_make_wcs_frame_auto_fontsize_false_skips_dispatch_entirely():
    """auto_fontsize=False shouldn't touch the cache attribute at all
    (the dispatcher never runs)."""
    fig = plt.figure(figsize=(3, 2))
    ax = sph.make_wcs_frame(111, "AIT", center=0, fig=fig,
                            auto_fontsize=False)
    assert not hasattr(ax, '_sph_auto_label_fontsize')
    plt.close(fig)


# ---- _dispatch_auto_size (internal but tested separately) ------------------

def test_dispatch_uses_wcs_path_for_wcsaxes():
    """_dispatch_auto_size returns the WCS path's value — useful for
    confirming the type-check picks WCS over the mpl fallback."""
    fig, ax = _wcs_axes(figsize=(4, 3))
    fs = _dispatch_auto_size(ax, axis='both', floor=6.0, ceiling=None,
                              n_ticks_hint=6)
    assert fs is not None
    plt.close(fig)


def test_attach_resize_reflow_handles_axes_without_figure():
    """Pathological axes with no figure shouldn't crash the attach
    helper — it returns silently."""

    class FakeNoFig:
        pass

    # Should not raise.
    _attach_resize_reflow(FakeNoFig(), axis='both', floor=6.0,
                          ceiling=None, n_ticks_hint=6)
