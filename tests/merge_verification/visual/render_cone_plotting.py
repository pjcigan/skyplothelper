"""Render the cone plotting helpers for visual eyeballing.

Covers: ``cone_scatter``, ``cone_plot``, ``cone_scatter_z``,
``cone_hexbin``, ``cone_pcolormesh``.

Usage
-----
    python render_cone_plotting.py            # save PNGs to output/
    python render_cone_plotting.py --show     # display interactively
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from _common import banner, save_or_show

from skyplothelper.cone.frame import make_cone_frame
from skyplothelper.cone.plotting import (
    cone_hexbin,
    cone_pcolormesh,
    cone_plot,
    cone_scatter,
    cone_scatter_z,
)

try:
    from astropy.cosmology import Planck18
    _HAS_COSMOLOGY = True
except ImportError:
    _HAS_COSMOLOGY = False

PANELS = {}


def _panel(name):
    def deco(fn):
        PANELS[name] = fn
        return fn
    return deco


def _make_cone(angle_center=180, angle_half_width=30,
               r_min=0.0, r_max=0.15, fig=None, **kw):
    """Build a representative redshift-cone wedge centered on RA=12h."""
    if fig is None:
        fig = plt.figure(figsize=(7, 6))
    ax = make_cone_frame(
        111, angle_center=angle_center,
        angle_half_width=angle_half_width,
        r_min=r_min, r_max=r_max, fig=fig, **kw,
    )
    return fig, ax


@_panel("cone_01_scatter")
def render_scatter():
    """cone_scatter — random galaxies in a redshift wedge."""
    rng = np.random.default_rng(42)
    n = 400
    rs = rng.beta(2.5, 5, n) * 0.15
    angles = rng.normal(180, 15, n) + rng.choice([-10, 0, 10], n)
    angles = np.clip(angles, 155, 205)

    fig, ax = _make_cone()
    cone_scatter(ax, angles, rs, s=10, c="C0", alpha=0.5,
                 edgecolor="none")
    ax.set_title("cone_scatter — 400 mock galaxies in a redshift wedge",
                 pad=20)
    return fig


@_panel("cone_02_scatter_colormapped")
def render_scatter_colormapped():
    """cone_scatter — points colored by a derived value (lookback time)."""
    rng = np.random.default_rng(7)
    n = 400
    rs = rng.beta(2, 4, n) * 0.15
    angles = rng.uniform(155, 205, n)
    lookback = rs * 13.0 + rng.normal(0, 0.2, n)

    fig, ax = _make_cone()
    sc = cone_scatter(ax, angles, rs, s=15, c=lookback,
                      cmap="viridis", alpha=0.8, edgecolor="none")
    fig.colorbar(sc, ax=ax, label="lookback time [Gyr]",
                 shrink=0.7, pad=0.12)
    ax.set_title("cone_scatter — c=lookback colormapped",
                 pad=20)
    return fig


@_panel("cone_03_plot_track")
def render_plot():
    """cone_plot — a track / chained line in cone coordinates."""
    fig, ax = _make_cone()
    track_angles = np.array([180, 178, 175, 172, 170, 173, 178, 183, 187, 190])
    track_rs = np.array([0.02, 0.03, 0.045, 0.06, 0.075,
                         0.085, 0.10, 0.115, 0.13, 0.14])
    cone_plot(ax, track_angles, track_rs,
              color="orange", lw=2.5, marker="o", ms=6,
              markeredgecolor="k", markeredgewidth=0.5)
    ax.set_title("cone_plot — connected track in (angle, r)",
                 pad=20)
    return fig


@_panel("cone_04_scatter_z_redshift")
def render_scatter_z_redshift():
    """cone_scatter_z on a redshift-axis wedge — identity conversion."""
    rng = np.random.default_rng(11)
    n = 250
    zs = rng.beta(2, 4, n) * 0.15
    angles = rng.uniform(155, 205, n)

    fig, ax = _make_cone()
    cone_scatter_z(ax, angles, zs, s=8, c="C2", alpha=0.7,
                   edgecolor="none")
    ax.set_title("cone_scatter_z (r_variable='redshift', identity convert)",
                 pad=20)
    return fig


# Conditionally register the comoving-distance panel — only when
# astropy.cosmology is available. Pytest-mpl tests for it auto-omit
# when the panel isn't in PANELS.
if _HAS_COSMOLOGY:

    @_panel("cone_05_scatter_z_comoving")
    def render_scatter_z_comoving():
        """cone_scatter_z with a comoving-distance wedge — uses
        astropy.cosmology."""
        rng = np.random.default_rng(3)
        n = 250
        zs = rng.beta(2, 4, n) * 0.15
        angles = rng.uniform(155, 205, n)

        fig, ax = _make_cone(
            r_min=0.0, r_max=600,
            r_variable="comoving_distance", r_unit="Mpc",
            cosmology=Planck18,
        )
        cone_scatter_z(ax, angles, zs, cosmology=Planck18,
                       s=8, c="C5", alpha=0.7, edgecolor="none")
        ax.set_title("cone_scatter_z — comoving distance (Planck18)",
                     pad=20)
        return fig


@_panel("cone_06_hexbin")
def render_hexbin():
    """cone_hexbin — density of mock galaxies."""
    rng = np.random.default_rng(0)
    n = 8000
    rs = rng.beta(2, 5, n) * 0.15
    angles = rng.uniform(155, 205, n) + rng.normal(0, 3, n)
    angles = np.clip(angles, 155, 205)

    fig, ax = _make_cone()
    hexes = cone_hexbin(ax, angles, rs, gridsize=25, cmap="viridis",
                        mincnt=1)
    fig.colorbar(hexes, ax=ax, label="counts / hex bin",
                 shrink=0.7, pad=0.12)
    ax.set_title("cone_hexbin — 8000-pt mock catalog",
                 pad=20)
    return fig


@_panel("cone_07_pcolormesh")
def render_pcolormesh():
    """cone_pcolormesh — pre-binned density on the (angle, r) grid."""
    rng = np.random.default_rng(0)
    n = 8000
    rs = rng.beta(2, 5, n) * 0.15
    angles = rng.uniform(155, 205, n) + rng.normal(0, 3, n)
    angles = np.clip(angles, 155, 205)

    angle_edges = np.linspace(155, 205, 26)
    r_edges = np.linspace(0, 0.15, 16)
    H, _, _ = np.histogram2d(angles, rs, bins=[angle_edges, r_edges])
    H_T = H.T

    fig, ax = _make_cone()
    qm = cone_pcolormesh(ax, angle_edges, r_edges, H_T, cmap="plasma")
    fig.colorbar(qm, ax=ax, label="counts / cell",
                 shrink=0.7, pad=0.12)
    ax.set_title("cone_pcolormesh — pre-binned density",
                 pad=20)
    return fig


@_panel("cone_09_bowtie_per_half_overrides")
def render_bowtie_per_half_overrides():
    """Bowtie with per-half cosmetic overrides. Two
    halves get distinct grid colors and angle labels via
    ``top_kwargs`` / ``bot_kwargs`` while sharing all geometry."""
    from skyplothelper.cone.frame import make_bowtie_frame

    rng = np.random.default_rng(11)
    n = 250
    rs_n = rng.beta(2, 5, n) * 0.15
    rs_s = rng.beta(2, 5, n) * 0.15
    ang_n = rng.uniform(150, 210, n)
    ang_s = rng.uniform(150, 210, n)

    fig = plt.figure(figsize=(8, 8))
    top, bot = make_bowtie_frame(
        angle_center=180, angle_half_width=40,
        r_min=0, r_max=0.15, angle_tick_spacing=10,
        gridcolor="0.85",  # default for both
        top_kwargs={"gridcolor": "steelblue", "angle_label": "NGP"},
        bot_kwargs={"gridcolor": "crimson", "angle_label": "SGP"},
        fig=fig,
    )
    cone_scatter(top, ang_n, rs_n, s=8, c="steelblue",
                 alpha=0.6, edgecolor="none")
    cone_scatter(bot, ang_s, rs_s, s=8, c="crimson",
                 alpha=0.6, edgecolor="none")
    fig.suptitle("make_bowtie_frame — per-half overrides via "
                 "top_kwargs / bot_kwargs",
                 y=0.95, fontsize=11)
    return fig


@_panel("cone_08_bowtie_with_scatter")
def render_bowtie_with_scatter():
    """A bowtie (back-to-back wedge) with scatter on each side — combines
    make_bowtie_frame and cone_scatter."""
    from skyplothelper.cone.frame import make_bowtie_frame

    rng = np.random.default_rng(99)
    n = 200
    rs_a = rng.beta(2, 5, n) * 0.15
    rs_b = rng.beta(2, 5, n) * 0.15
    ang_a = rng.uniform(-30, 30, n)
    ang_b = rng.uniform(150, 210, n)

    fig = plt.figure(figsize=(7, 7))
    pair = make_bowtie_frame(
        angle_center=0, angle_half_width=30,
        r_min=0, r_max=0.15, fig=fig,
    )
    top, bot = pair
    cone_scatter(top, ang_a, rs_a, s=8, c="C0", alpha=0.6, edgecolor="none")
    cone_scatter(bot, ang_b - 180, rs_b, s=8, c="C3", alpha=0.6, edgecolor="none")
    fig.suptitle("make_bowtie_frame + cone_scatter on each half",
                 y=0.93)
    return fig


def main():
    banner("cone.plotting — merge-verification visual gallery")
    for name, builder in PANELS.items():
        save_or_show(builder(), name)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
