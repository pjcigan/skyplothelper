"""Tests for skyplothelper.channel_map (spectral-cube channel grids)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import skyplothelper as sph  # noqa: E402
from skyplothelper.images.channels import channel_map  # noqa: E402
from skyplothelper.images.cube import DataCube  # noqa: E402
from skyplothelper.wcs_frame import dummy_standard_hdr  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _synthetic_cube(nchan=8, ny=40, nx=40):
    """A blob that drifts across the field with channel + a spectral WCS."""
    yy, xx = np.mgrid[0:ny, 0:nx]
    planes = []
    for k in range(nchan):
        cx = 8 + (nx - 16) * k / (nchan - 1)
        planes.append(np.exp(-((xx - cx) ** 2 + (yy - ny / 2) ** 2) / (2 * 5 ** 2)))
    cube = np.stack(planes).astype(float)
    hdr = dummy_standard_hdr(
        centercoords_deg=(180.0, 30.0),
        cdelts=(-1.0 / 3600, 1.0 / 3600),
        cunit="deg", projection="TAN", naxis_xy=(nx, ny),
    )
    hdr["NAXIS"] = 3
    hdr["NAXIS3"] = nchan
    hdr["CTYPE3"] = "VRAD"
    hdr["CRVAL3"] = 300000.0     # 300 km/s at the reference channel (m/s)
    hdr["CDELT3"] = -1000.0      # -1 km/s per channel
    hdr["CRPIX3"] = 1.0
    hdr["CUNIT3"] = "m/s"
    hdr["BUNIT"] = "Jy/beam"
    return cube, hdr


# ---------------------------------------------------------------------------
# DataCube core (channel_map's shared cube backend)
# ---------------------------------------------------------------------------

def test_datacube_squeezes_degenerate_axis():
    cube, hdr = _synthetic_cube(nchan=5)
    view = DataCube(cube[np.newaxis], hdr)    # (1, nchan, y, x) -> squeezed
    assert view.data.shape == cube.shape
    assert view.nchan == 5


def test_datacube_label_converts_to_kms():
    cube, hdr = _synthetic_cube(nchan=8)
    view = DataCube(cube, hdr)
    assert view.axis_kind == "velocity"
    assert view.spectral_label(0) == "300 km/s"   # channel 0 = CRVAL3 = 300 km/s


def test_datacube_no_header_has_no_spectral():
    cube, _ = _synthetic_cube()
    view = DataCube(cube)
    assert view.spectral is None
    assert view.spectral_label(0) is None


def test_datacube_rejects_non_cube():
    with pytest.raises(ValueError, match="3-D"):
        DataCube(np.zeros((10, 10)))              # 2-D, not a cube


def test_channel_map_accepts_datacube():
    cube, hdr = _synthetic_cube(nchan=6)
    view = DataCube(cube, hdr)
    res = channel_map(view, channels=3)
    assert len(res.channels) == 3
    plt.close(res.fig)


# ---------------------------------------------------------------------------
# channel_map: layout + return object
# ---------------------------------------------------------------------------

def test_returns_result_with_all_fields():
    cube, hdr = _synthetic_cube(nchan=12)
    res = channel_map(cube, header=hdr, channels=9, ncols=3)
    assert res.axes.shape == (3, 3)
    assert len(res.images) == 9
    assert res.colorbar is not None
    assert res.norm is not None
    assert len(res.channels) == 9
    assert len(res.velocities) == 9
    assert len(res.labels) == 9
    assert res.beam is None and res.scalebar is None
    assert res.moment0_image is None and res.moment0_units is None
    # tuple-unpackable
    (fig, axes, images, cb, norm, chans, vels, labels,
     beam, sbar, mom0, mom0_units) = res
    assert fig is res.fig


def test_panel_accessor_returns_channel_axes():
    cube, hdr = _synthetic_cube()
    res = channel_map(cube, header=hdr, channels=[0, 5])
    assert res.panel(5) is res.images[1].axes
    assert hasattr(res.panel(0), "coords")


def test_wcs_panels_by_default():
    cube, hdr = _synthetic_cube()
    res = channel_map(cube, header=hdr, channels=4, ncols=2)
    assert all(hasattr(ax, "coords") for ax in res.axes.ravel())


def test_plain_panels_when_disabled():
    cube, hdr = _synthetic_cube()
    res = channel_map(cube, header=hdr, channels=4, ncols=2, wcs_panels=False)
    assert not any(hasattr(ax, "coords") for ax in res.axes.ravel())
    assert res.axes[0, 0].get_xticks().size == 0


def test_shared_norm_across_panels():
    cube, hdr = _synthetic_cube()
    res = channel_map(cube, header=hdr, channels=6, ncols=3)
    assert all(im.norm is res.norm for im in res.images)


def test_partial_grid_blanks_trailing_cells():
    cube, hdr = _synthetic_cube(nchan=20)
    res = channel_map(cube, header=hdr, channels=5, ncols=3)   # 2x3, 1 blank
    assert res.axes.shape == (2, 3)
    blank = [not ax.get_visible() for ax in res.axes.ravel()]
    assert sum(blank) == 1


def test_explicit_and_negative_channel_indices():
    cube, hdr = _synthetic_cube(nchan=10)
    res = channel_map(cube, header=hdr, channels=[0, 3, -1], ncols=3)
    assert list(res.channels) == [0, 3, 9]


def test_explicit_norm_overrides_stretch():
    cube, hdr = _synthetic_cube()
    my_norm = sph.make_norm(stretch="linear", clip="manual", vmin=0.0, vmax=1.0)
    res = channel_map(cube, header=hdr, channels=4, norm=my_norm)
    assert res.norm is my_norm


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------

def _label_texts(res):
    out = []
    for ax in res.axes.ravel():
        out.extend(t.get_text() for t in ax.texts if t.get_text())
    return out


def test_velocity_labels_have_units_and_stroke():
    cube, hdr = _synthetic_cube()
    res = channel_map(cube, header=hdr, channels=4)
    texts = _label_texts(res)
    assert any("km/s" in t for t in texts)
    # every label carries the readability stroke
    labels = [t for ax in res.axes.ravel() for t in ax.texts if t.get_text()]
    assert all(t.get_path_effects() for t in labels)


def test_channel_label_mode():
    cube, hdr = _synthetic_cube()
    res = channel_map(cube, header=hdr, channels=[0, 2], label="channel")
    assert any(t == "ch 0" for t in _label_texts(res))


def test_callable_label():
    cube, hdr = _synthetic_cube()
    res = channel_map(cube, header=hdr, channels=[0, 1],
                      label=lambda ch, view: f"slice {ch}")
    assert "slice 0" in _label_texts(res)


def test_label_none_draws_no_text():
    cube, hdr = _synthetic_cube()
    res = channel_map(cube, header=hdr, channels=4, label=None)
    assert _label_texts(res) == []


def test_no_header_falls_back_to_channel_labels():
    cube, _ = _synthetic_cube()
    res = channel_map(cube, channels=[0, 1])       # no header -> plain + ch labels
    assert all(not hasattr(ax, "coords") for ax in res.axes.ravel())
    assert any(t.startswith("ch ") for t in _label_texts(res))
    assert all(v is None for v in res.velocities)


# ---------------------------------------------------------------------------
# edge-only coordinate labels on WCS panels
# ---------------------------------------------------------------------------

def _n_label_panels(res):
    return sum(1 for ax in res.axes.ravel()
               if hasattr(ax, "coords") and ax.get_visible()
               and ax.coords[0].ticklabels.get_visible())


@pytest.mark.parametrize("mode,expected", [("plain", 0), ("minimal", 1)])
def test_tick_modes_label_panel_count(mode, expected):
    cube, hdr = _synthetic_cube(nchan=12)
    res = channel_map(cube, header=hdr, channels=6, ncols=3, ticks=mode)
    assert _n_label_panels(res) == expected


def test_ticks_complete_labels_every_panel():
    cube, hdr = _synthetic_cube(nchan=12)
    res = channel_map(cube, header=hdr, channels=6, ncols=3, ticks="complete")
    assert _n_label_panels(res) == 6


def test_minimal_label_panel_is_lower_left():
    cube, hdr = _synthetic_cube(nchan=12)
    res = channel_map(cube, header=hdr, channels=6, ncols=3)  # 2x3, minimal
    # lower-left occupied cell = row 1, col 0 = axes[1, 0]
    labeled = [ax for ax in res.axes.ravel()
               if hasattr(ax, "coords") and ax.coords[0].ticklabels.get_visible()]
    assert labeled == [res.axes[1, 0]]


def test_invalid_ticks_and_coords_raise():
    cube, hdr = _synthetic_cube()
    with pytest.raises(ValueError, match="ticks must be"):
        channel_map(cube, header=hdr, channels=4, ticks="bogus")
    with pytest.raises(ValueError, match="coords must be"):
        channel_map(cube, header=hdr, channels=4, coords="bogus")


# ---------------------------------------------------------------------------
# layout knobs: start_panel, pad, panel_facecolor
# ---------------------------------------------------------------------------

def test_start_panel_blanks_leading_cells():
    cube, hdr = _synthetic_cube(nchan=12)
    res = channel_map(cube, header=hdr, channels=5, ncols=3, start_panel=1)
    assert not res.axes.ravel()[0].get_visible()          # cell 0 blank
    assert res.axes.ravel()[1].get_visible()


def test_panel_facecolor_cmap_min():
    cube, hdr = _synthetic_cube()
    res = channel_map(cube, header=hdr, channels=4, panel_facecolor="cmap_min")
    import matplotlib as mpl
    lo = mpl.colormaps["sph.lagoon"](0.0)
    assert res.axes.ravel()[0].get_facecolor()[:3] == pytest.approx(lo[:3], abs=1e-3)


def test_all_panels_uniform_size():
    cube, hdr = _synthetic_cube(nchan=12)
    res = channel_map(cube, header=hdr, channels=9, ncols=3)
    res.fig.canvas.draw()
    widths = {round(im.axes.get_position().width, 3) for im in res.images}
    heights = {round(im.axes.get_position().height, 3) for im in res.images}
    assert len(widths) == 1 and len(heights) == 1


def test_moment0_panel_same_size_as_channels():
    cube, hdr = _synthetic_cube(nchan=12)
    res = channel_map(cube, header=hdr, channels=8, ncols=3, moment0=True)
    res.fig.canvas.draw()
    m0 = res.moment0_image.axes.get_position()
    ch = res.images[0].axes.get_position()
    assert round(m0.width, 3) == round(ch.width, 3)
    assert round(m0.height, 3) == round(ch.height, 3)


def test_independent_wspace_hspace():
    cube, hdr = _synthetic_cube(nchan=12)
    res = channel_map(cube, header=hdr, channels=6, ncols=3,
                      wspace=0.02, hspace=0.4)
    res.fig.canvas.draw()
    p = [res.axes[r, c].get_position() for r in range(2) for c in range(3)]
    hgap = p[1].x0 - p[0].x1          # within a row
    vgap = p[0].y0 - p[3].y1          # between rows
    assert vgap > hgap                # bigger hspace → bigger vertical gap


def test_complete_mode_widens_default_pad():
    cube, hdr = _synthetic_cube(nchan=12)
    tight = channel_map(cube, header=hdr, channels=6, ncols=3, ticks="minimal")
    wide = channel_map(cube, header=hdr, channels=6, ncols=3, ticks="complete")
    tight.fig.canvas.draw()
    wide.fig.canvas.draw()
    t = [tight.axes[r, c].get_position() for r in range(2) for c in range(3)]
    w = [wide.axes[r, c].get_position() for r in range(2) for c in range(3)]
    assert (w[1].x0 - w[0].x1) > (t[1].x0 - t[0].x1)   # complete pads wider


def test_inward_ticks_default():
    cube, hdr = _synthetic_cube()
    res = channel_map(cube, header=hdr, channels=4, ncols=2)
    ax = res.axes[0, 0]
    # WCSAxes stores tick direction on the coord's ticks object as 'in'
    assert ax.coords[0].ticks.get_tick_out() is False


def test_nice_axis_labels_default():
    cube, hdr = _synthetic_cube()
    res = channel_map(cube, header=hdr, channels=4, ncols=2)  # ICRS TAN
    lp = [ax for ax in res.axes.ravel()
          if hasattr(ax, "coords") and ax.coords[0].ticklabels.get_visible()][0]
    assert "R.A." in lp.coords[0].get_axislabel()
    assert "Decl." in lp.coords[1].get_axislabel()


# ---------------------------------------------------------------------------
# channel preprocessing: every_N / average / smooth / trim_empty / order
# ---------------------------------------------------------------------------

def test_every_N_thins_channels():
    cube, hdr = _synthetic_cube(nchan=12)
    res = channel_map(cube, header=hdr, channels=None, every_N=4)
    assert len(res.images) == 3          # 12 // 4


def test_average_reduces_channel_count():
    cube, hdr = _synthetic_cube(nchan=12)
    res = channel_map(cube, header=hdr, channels=None, average=3)
    assert len(res.images) == 4          # 12 / 3


def test_smooth_hanning_preserves_count():
    cube, hdr = _synthetic_cube(nchan=12)
    res = channel_map(cube, header=hdr, channels=None, smooth="hanning",
                      smooth_width=3)
    assert len(res.images) == 12


def test_trim_empty_drops_nan_channels():
    cube, hdr = _synthetic_cube(nchan=10)
    cube[3] = np.nan
    res = channel_map(cube, header=hdr, channels=None, trim_empty=True)
    assert len(res.images) == 9


def test_order_descending_sorts_by_velocity():
    cube, hdr = _synthetic_cube(nchan=10)   # CDELT3 < 0 → velocity decreases
    res = channel_map(cube, header=hdr, channels=[0, 4, 8], order="descending")
    vels = [v[0] for v in res.velocities]
    assert vels == sorted(vels, reverse=True)


def test_invalid_order_raises():
    cube, hdr = _synthetic_cube()
    with pytest.raises(ValueError, match="order must be"):
        channel_map(cube, header=hdr, channels=4, order="sideways")


# ---------------------------------------------------------------------------
# spectral labels: auto / frequency / units / channel-fallback
# ---------------------------------------------------------------------------

def test_auto_label_velocity_cube():
    cube, hdr = _synthetic_cube()
    res = channel_map(cube, header=hdr, channels=[0], label="auto")
    assert "km/s" in _label_texts(res)[0]


def test_auto_label_frequency_cube():
    cube, hdr = _synthetic_cube()
    hdr["CTYPE3"] = "FREQ"
    hdr["CRVAL3"] = 1.15e11
    hdr["CDELT3"] = 1e6
    hdr["CUNIT3"] = "Hz"
    res = channel_map(cube, header=hdr, channels=[0], label="auto")
    assert "GHz" in _label_texts(res)[0]


def test_label_unit_override():
    cube, hdr = _synthetic_cube()
    res = channel_map(cube, header=hdr, channels=[0], label="velocity",
                      label_unit="m/s")
    assert "m/s" in _label_texts(res)[0]


# ---------------------------------------------------------------------------
# beam / scalebar / moment0
# ---------------------------------------------------------------------------

def _cube_with_beam():
    cube, hdr = _synthetic_cube()
    hdr["BMAJ"] = 2.0 / 3600
    hdr["BMIN"] = 1.0 / 3600
    hdr["BPA"] = 30.0
    return cube, hdr


def test_beam_drawn_from_header():
    cube, hdr = _cube_with_beam()
    res = channel_map(cube, header=hdr, channels=4, beam=True)
    assert res.beam is not None
    # visible by default (a bare Beam is transparent) — face alpha > 0
    assert res.beam.get_facecolor()[3] > 0
    assert res.beam in res.beam.axes.patches


def test_beam_kwargs_override_style():
    cube, hdr = _cube_with_beam()
    res = channel_map(cube, header=hdr, channels=4, beam=True,
                      beam_kwargs={"facecolor": "red"})
    assert res.beam.get_facecolor()[:3] == pytest.approx((1.0, 0.0, 0.0))


def test_beam_absent_header_warns_not_fatal():
    cube, hdr = _synthetic_cube()          # no BMAJ/BMIN
    with pytest.warns(UserWarning, match="beam not drawn"):
        res = channel_map(cube, header=hdr, channels=4, beam=True)
    assert res.beam is None


def test_scalebar_drawn():
    cube, hdr = _synthetic_cube()
    res = channel_map(cube, header=hdr, channels=4, scalebar=2.0)
    assert res.scalebar is not None


def test_moment0_adds_summary_panel():
    cube, hdr = _synthetic_cube(nchan=12)
    res = channel_map(cube, header=hdr, channels=6, ncols=3, moment0=True)
    assert res.moment0_image is not None
    assert len(res.images) == 6                      # channels unchanged
    # 6 channels + 1 moment0 = 7 occupied cells → 3x3 grid
    assert res.axes.shape == (3, 3)


def test_moment0_units_exposed():
    cube, hdr = _synthetic_cube()                    # BUNIT Jy/beam, VRAD m/s
    res = channel_map(cube, header=hdr, channels=4, moment0=True)
    assert res.moment0_units == "Jy/beam m/s"        # tidied from m.s**-1


def test_moment0_custom_cmap():
    cube, hdr = _synthetic_cube()
    res = channel_map(cube, header=hdr, channels=4, moment0=True,
                      moment0_cmap="magma")
    assert res.moment0_image.get_cmap().name == "magma"


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_invalid_channel_count_raises():
    cube, hdr = _synthetic_cube()
    with pytest.raises(ValueError):
        channel_map(cube, header=hdr, channels=0)


def test_invalid_label_mode_raises():
    cube, hdr = _synthetic_cube()
    with pytest.raises(ValueError, match="label must be"):
        channel_map(cube, header=hdr, channels=2, label="bogus")
