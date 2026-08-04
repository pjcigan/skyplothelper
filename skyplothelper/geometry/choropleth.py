"""Choropleth fills on the sphere — color a set of polygon rings by a value.

A single public entry point, :func:`choropleth`, that fills a collection of
spherical-polygon rings (plates, time zones, countries, survey tiles, HEALPix
cells, …) each colored by its own data value, through the same region-fill
pipeline the rest of the package uses (antimeridian + limb clipping, holes).
The filled-geographic helpers (``plot_tectonic_plates(fill=True, values=…)`` and
``plot_time_zones(fill=True, values=…)``) route their ``values=`` path through
here, and it is exposed as ``sph.choropleth`` for arbitrary ring sets.
"""

from __future__ import annotations

from typing import Any

import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .shapes import add_spherical_polygon

__all__ = ['choropleth']


def choropleth(ax: Any, rings: Any, values: npt.ArrayLike, *,
               cmap: Any = 'viridis', vmin: float | None = None,
               vmax: float | None = None, norm: Any = None,
               edgecolor: Any = 'none', linewidth: float | None = None,
               alpha: float | None = None, zorder: float = 1,
               missing_color: Any = None,
               stroke_color: Any = None, stroke_lw: float | None = None,
               resolution: int = 0, **kwargs: Any) -> Any:
    """Fill spherical-polygon rings colored by a per-ring data value.

    A choropleth on the sphere: each ring in *rings* is filled with the color
    its *values* entry maps to through *cmap* and a :class:`~matplotlib.colors.
    Normalize`. Built on :func:`~skyplothelper.geometry.add_spherical_polygon`,
    so the fills inherit antimeridian + frame-edge clipping and work on any
    FITS-projection frame — all-sky, globe, or planet map.

    Parameters
    ----------
    ax : WCSAxes
        A FITS-projection sky/planet frame.
    rings : sequence of (lons, lats)
        The polygon rings, each a pair of degree array-likes (auto-closed).
    values : array_like
        One value per ring (same length / order as *rings*). A ``NaN`` value
        leaves that ring unfilled unless *missing_color* is given.
    cmap : str or Colormap
        Colormap for the values (default ``'viridis'``).
    vmin, vmax : float, optional
        Value range for the color scale. Default: the finite data min / max.
        Ignored if an explicit *norm* is passed.
    norm : matplotlib Normalize, optional
        A ready-made norm (e.g. ``LogNorm``); overrides *vmin* / *vmax*.
    edgecolor : color
        Ring outline color (default ``'none'`` — fills read as solid regions).
    linewidth : float, optional
        Ring outline width (only meaningful with a non-``'none'`` *edgecolor*).
    alpha : float, optional
        Fill transparency.
    zorder : float
        Draw order (default 1 — under typical overlays).
    missing_color : color, optional
        Color for rings whose value is ``NaN`` (default: skip them entirely).
    stroke_color, stroke_lw : color / float, optional
        Optional legibility stroke around each ring (shared stroke helper).
    resolution : int
        Great-circle densification per edge (default 0 — the bundled datasets
        are already densified; raise it for sparse hand-built rings).
    **kwargs
        Forwarded to :func:`~skyplothelper.geometry.add_spherical_polygon`.

    Returns
    -------
    matplotlib.cm.ScalarMappable
        A mappable already carrying the data array, ready for a colorbar:
        ``sph.add_colorbar(ax, mappable=sm)`` or ``fig.colorbar(sm, ax=ax)``.
        The filled patches are already on *ax*.
    """
    rings = list(rings)
    values = np.asarray(values, dtype=float)
    if len(values) != len(rings):
        raise ValueError(
            f"choropleth: got {len(values)} values for {len(rings)} rings — "
            "they must be the same length (one value per ring).")

    cmap_obj = plt.get_cmap(cmap)
    if norm is None:
        finite = values[np.isfinite(values)]
        lo = (float(np.min(finite)) if finite.size else 0.0) if vmin is None else vmin
        hi = (float(np.max(finite)) if finite.size else 1.0) if vmax is None else vmax
        if lo == hi:                       # degenerate: widen so colors resolve
            lo, hi = lo - 0.5, hi + 0.5
        norm = mcolors.Normalize(vmin=lo, vmax=hi)

    for (lons, lats), val in zip(rings, values):
        if not np.isfinite(val):
            if missing_color is None:
                continue
            color: Any = missing_color
        else:
            color = cmap_obj(norm(val))
        extra: dict[str, Any] = dict(kwargs)
        if linewidth is not None:
            extra['linewidth'] = linewidth
        add_spherical_polygon(
            ax, lons, lats, resolution=resolution, facecolor=color,
            edgecolor=edgecolor, alpha=alpha, zorder=zorder,
            stroke_color=stroke_color, stroke_lw=stroke_lw, **extra)

    sm = mcm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array(values)
    return sm
