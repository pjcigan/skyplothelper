"""Co-visibility regions (skyplothelper.visibility) — visual gallery.

Run:  python render_covisibility.py   (writes to output/, gitignored)

Panels:
  - covis_01_kokee_wettzell     : 2-station instantaneous co-visible sky
                                  (canonical Kokee-Wettzell baseline). The two
                                  individual station caps are underlaid faintly
                                  so the intersection is clearly their overlap;
                                  an inset Earth globe shows the stations.
  - covis_02_multi_station_mask : 4 stations (one with a non-circular azimuth
                                  horizon mask). Each station's visible cap is
                                  underlaid in its own light color (centers as
                                  matching dots) with the all-4 intersection
                                  overlaid, so the compound region can be matched
                                  to its contributions.
  - covis_03_duration_bands     : a global 6-station array — declination bands
                                  co-visible >=4 h/day by >=2 / >=3 / >=4 of the
                                  stations (the min_stations "k of N" feature).

A fixed observation time is used so the baselines are deterministic.
"""

import sys

import matplotlib.pyplot as plt
from _common import banner, save_or_show

import skyplothelper as sph
from skyplothelper.globe import make_planet_frame, plot_baselines

PANELS = {}


def _panel(name):
    def deco(fn):
        PANELS[name] = fn
        return fn
    return deco


# Fixed instant so the rendered baselines are reproducible.
_TIME = "2026-06-05T06:00:00"

# Canonical / illustrative geodetic-VLBI stations (deg).
_KOKEE = {"lat": 22.13, "lon": -159.665}
_WETTZELL = {"lat": 49.145, "lon": 12.878}
_GBT = {"lat": 38.43, "lon": -79.84}
_VLA = {"lat": 34.08, "lon": -107.62}

_CAP_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd",
               "#8c564b", "#17becf"]


def _baseline_sites(stations):
    """Covisibility station dict -> the (lon, lat) form plot_baselines takes."""
    return {name: (st["lon"], st["lat"]) for name, st in stations.items()}


def _place_inset(ax, rect):
    """Shrink a freshly-built frame into an inset ``rect`` and re-fit its tick
    labels to the smaller size (the frame's own auto-fit ran at full size, so
    re-run it after repositioning)."""
    from skyplothelper.autosize import auto_size_ticklabels
    ax.set_position(rect)
    ax.figure.canvas.draw()
    try:
        auto_size_ticklabels(ax)
    except Exception:
        pass


def _underlay_individual(ax, stations, time, el_min=15):
    """Render each station's individual visible cap faintly (own color, dotted
    edge) + its overhead point, so the final compound region can be matched to
    its per-station contributions. Returns the name→color map used."""
    cmap = {}
    for i, (name, st) in enumerate(stations.items()):
        color = _CAP_COLORS[i % len(_CAP_COLORS)]
        cmap[name] = color
        reg = sph.covisibility_region(ax, {name: st}, time, el_min=el_min)
        if not reg.is_empty:
            reg.render(facecolor=color, alpha=0.13, edgecolor=color,
                       lw=0.8, linestyle=":")
        cap = sph.covisibility_circles({name: st}, time, el_min=el_min)[0]
        ax.scatter(cap["center"].ra.deg, cap["center"].dec.deg,
                   transform=ax.get_transform("world"), s=22, color=color,
                   edgecolor="k", linewidth=0.5, zorder=7)
    return cmap


@_panel("covis_01_kokee_wettzell")
def render_two_station():
    stations = {"Kokee": _KOKEE, "Wettzell": _WETTZELL}
    fig = plt.figure(figsize=(10, 5.5))
    ax = sph.make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    _underlay_individual(ax, stations, _TIME)
    reg = sph.covisibility_region(ax, stations, _TIME, el_min=15)
    reg.render(facecolor="cyan", alpha=0.45, edgecolor="teal", lw=1.6)
    ax.set_title("Kokee-Wettzell co-visible sky (el > 15°): intersection (cyan) "
                 "of the two station caps", fontsize=10.5)

    # Inset Earth globe (lower-left) showing the two station locations.
    # Tick labels are auto-rescaled to the small inset size (see _place_inset).
    plt.figure(fig.number)
    gax = make_planet_frame(111, center_LONdeg=-70, center_LATdeg=40,
                            lon_deg_spacing=30, lat_deg_spacing=30,
                            gridcolor="0.6", auto_fontsize=False)
    _place_inset(gax, [0.015, 0.04, 0.26, 0.26])
    plot_baselines(gax, _baseline_sites(stations), pairs="all", color="red",
                   linewidth=1.0, marker_color="red", marker_size=18,
                   site_label_fontsize=6)
    return fig


@_panel("covis_02_multi_station_mask")
def render_multi_station_mask():
    # One station carries a non-circular horizon mask: a ridge to the north
    # (high elevation block around az 0) so its cap boundary is visibly dented.
    masked_vla = dict(_VLA)
    masked_vla["hor_mask"] = [[0, 45, 90, 135, 180, 225, 270, 315],
                              [45, 30, 12, 12, 12, 12, 12, 30]]
    stations = {"Kokee": _KOKEE, "GBT": _GBT,
                "VLA(mask)": masked_vla, "Wettzell": _WETTZELL}

    fig = plt.figure(figsize=(10, 5.5))
    ax = sph.make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    _underlay_individual(ax, stations, _TIME)
    reg_all = sph.covisibility_region(ax, stations, _TIME, el_min=15)
    reg_all.render(facecolor="none", edgecolor="crimson", lw=2.2, hatch="////")
    ax.set_title("4 stations (VLA az-masked): per-station caps (light) and the "
                 "all-4 intersection (red hatch)", fontsize=10.5)
    return fig


@_panel("covis_03_duration_bands")
def render_duration_bands():
    # A globe-spanning array: no source is visible to ALL of them at once, but
    # the "k of N" duration band shrinks toward the pole as k rises — the
    # schedulable declination window for a chosen number of simultaneous
    # stations.
    glob = {
        "VLA": {"lat": 34.1, "lon": -107.6},
        "Effelsberg": {"lat": 50.5, "lon": 6.9},
        "ATCA": {"lat": -30.3, "lon": 149.6},
        "Hartebeesthoek": {"lat": -25.9, "lon": 27.7},
        "Kokee": {"lat": 22.1, "lon": -159.7},
        "Sheshan": {"lat": 31.1, "lon": 121.2},
    }
    fig = plt.figure(figsize=(10, 5.5))
    ax = sph.make_wcs_frame(111, projection="AIT", center=180, fig=fig)
    for k, color, alpha in [(2, "C0", 0.22), (3, "C2", 0.32), (4, "C3", 0.45)]:
        band = sph.covisibility_duration_band(ax, glob, min_hours=4.0,
                                              el_min=15, min_stations=k)
        if not band.is_empty:
            band.render(facecolor=color, alpha=alpha, edgecolor="none")
    ax.set_title("Global 6-station array: sky co-visible >=4 h/day by "
                 ">=2 (blue) / >=3 (green) / >=4 (red) stations", fontsize=10)

    # Plate-carrée locator inset (lower-left): the global station spread that
    # makes the all-6 set never simultaneously co-visible. Auto-rescaled ticks.
    plt.figure(fig.number)
    cax = sph.make_wcs_frame(111, projection="CAR", center=0, direction="geo",
                             fig=fig, gridcolor="0.6", auto_fontsize=False)
    _place_inset(cax, [0.015, 0.05, 0.30, 0.15])
    plot_baselines(cax, _baseline_sites(glob), pairs="all", color="C3",
                   linewidth=0.4, alpha=0.5, marker_color="C3", marker_size=12,
                   site_label_fontsize=5)
    return fig


def main():
    banner("co-visibility regions — gallery")
    for name, builder in PANELS.items():
        save_or_show(builder(), name)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
