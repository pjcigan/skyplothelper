"""High-level figure constructors.

``allsky_figure`` is a 1-call wrapper around ``make_wcs_frame`` for
typical all-sky plots; ``offset_figure`` is the same for offset / TAN
field plots; ``projection_gallery`` renders the same data side-by-side
across multiple projections.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np  # noqa: F401  (used by some allsky_figure paths)
import numpy.typing as npt
from astropy.coordinates import SkyCoord  # noqa: F401

try:
    import healpy as hp  # noqa: F401
except ImportError:
    hp = None

from .geometry._parsing import _coords_to_frame_deg
from .healpix import (
    _require_healpy,
    healpix_to_celestial,
    mask_seam_crossing_quads,
)
from .projections.registry import _resolve_projection
from .ticks import format_ticklabels
from .wcs_frame import clip_to_projection_boundary, make_wcs_frame

# Annotations are strings (PEP 563 / `from __future__ import annotations`),
# so this import costs nothing at run time.
if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord


def allsky_figure(projection: str = 'AIT', center: SkyCoord | float | tuple[float, float] = 180,
                  frame: str = 'ICRS',
                  figsize: tuple[float, float] | None = None, dpi: int = 100,
                  style: str = 'publication',
                  grid: bool = True, **wcs_kwargs: Any) -> tuple[Any, Any]:
    """
    Create an all-sky figure with one call, returning (fig, ax).

    Parameters
    ----------
    projection : str
    center : float, (lon, lat) pair, or SkyCoord
        Center longitude in degrees, or a full position. A SkyCoord is
        converted into ``frame`` (not blindly to ICRS), so a galactic
        SkyCoord on a galactic frame centers where you asked.
    frame : str
    figsize : tuple, optional
        Defaults to (14, 7)
    dpi : int
    style : str
        Tick label style preset
    grid : bool
    **wcs_kwargs
        Additional kwargs passed to make_wcs_frame

    Returns
    -------
    fig : Figure
    ax : WCSAxes

    Examples
    --------
    >>> fig, ax = sph.allsky_figure('mollweide', frame='Galactic')
    >>> ax.scatter(lons, lats, transform=ax.get_transform('world'), s=1)
    """
    if figsize is None:
        figsize = (14, 7)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = make_wcs_frame(111, projection, center=center, frame=frame,
                        grid=grid, fig=fig, **wcs_kwargs)
    format_ticklabels(ax, style=style)
    return fig, ax


def offset_figure(center: SkyCoord | tuple[float, float], fov_deg: float = 1.0, projection: str = 'TAN',
                  frame: str = 'ICRS',
                  figsize: tuple[float, float] | None = None, dpi: int = 100,
                  style: str = 'publication',
                  grid: bool = True, npix: int = 500,
                  **wcs_kwargs: Any) -> tuple[Any, Any]:
    """
    Create a zoomed field-of-view figure centered on a sky position.

    Parameters
    ----------
    center : tuple or SkyCoord
        (lon, lat) in degrees, or an astropy SkyCoord
    fov_deg : float
        Field of view in degrees
    projection : str
        Typically 'TAN' (gnomonic) for small fields
    frame : str
    figsize : tuple, optional
    dpi : int
    style : str
    grid : bool
    npix : int
        Number of pixels per axis
    **wcs_kwargs
        Additional kwargs passed to make_wcs_frame

    Returns
    -------
    fig : Figure
    ax : WCSAxes

    Examples
    --------
    >>> fig, ax = sph.offset_figure((83.6, 22.0), fov_deg=0.5)
    """
    if isinstance(center, SkyCoord):
        # Convert into the frame being built, not unconditionally to ICRS:
        # offset_figure(center=<galactic coord>, frame='galactic') must center
        # on the galactic position, not on its ICRS equivalent.
        lon, lat = _coords_to_frame_deg(center, frame)
    else:
        lon, lat = center

    if figsize is None:
        figsize = (8, 8)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # ``fov_deg`` is the convenience entry on ``make_wcs_frame``
    # (it computes ``cdelt = fov_deg / npix`` internally).
    ax = make_wcs_frame(111, projection, center=(lon, lat), frame=frame,
                        grid=grid, fig=fig, npix=(npix, npix),
                        fov_deg=fov_deg,
                        **wcs_kwargs)
    format_ticklabels(ax, style=style)
    return fig, ax


# ===== Multi-panel comparison =====

def projection_gallery(data: npt.ArrayLike | None = None,
                       nside: int | None = None,
                       projections: Sequence[str] | None = None,
                       center: float | tuple[float, float] | None = None,
                       center_lon: float | None = None,
                       center_lat: float | None = None,
                       frame: str = 'ICRS', ncols: int = 4,
                       figsize: tuple[float, float] | None = None,
                       cmap: str = 'viridis', title: str | None = None,
                       savepath: str | None = None, dpi: int = 150,
                       **pcolor_kwargs: Any) -> tuple[Any, list[Any]]:
    """
    Render the same data in multiple projections as a multi-panel figure.
    Currently only supports all-sky projections and (optionally) HEALPix data.

    Useful for choosing a projection or for presentation/comparison figures.

    Parameters
    ----------
    data : array-like, optional
        HEALPix map to render. If None, generates a random smoothed field.
    nside : int, optional
        HEALPix nside for the synthetic field when ``data`` is None
        (default 32). Ignored when ``data`` is supplied.
    projections : list of str, optional
        Projection names to show. Defaults to a curated set of common
        all-sky projections.
    center : float or tuple, optional
        Center of the projection. For all-sky projections, a single float
        gives the center longitude in degrees. For zenithal/field projections,
        a (lon, lat) tuple gives the center coordinates in degrees.
        Defaults to 180. Overridden by ``center_lon``/``center_lat``.
    center_lon : float, optional
        Center longitude in degrees. Takes precedence over ``center``.
    center_lat : float, optional
        Center latitude in degrees. Defaults to 0 if only ``center_lon``
        is given.
    frame : str
    ncols : int
        Number of columns in the gallery grid
    figsize : tuple, optional
    cmap : str
    title : str, optional
        Overall figure title
    savepath : str, optional
    dpi : int
    **pcolor_kwargs
        Kwargs passed to pcolormesh

    Returns
    -------
    fig : Figure
    axes : list of WCSAxes

    Examples
    --------
    >>> fig, axes = sph.projection_gallery()  # random field, default projections
    >>> fig, axes = sph.projection_gallery(my_hpxmap, nside=64, cmap='inferno')
    >>> fig, axes = sph.projection_gallery(
    ...     projections=['AIT', 'MOL', 'SFL', 'CAR', 'PAR', 'PCO'])
    """
    _require_healpy('projection_gallery')

    # Resolve center from center_lon/center_lat or center parameter
    if center_lon is not None:
        if center_lat is not None:
            center = (center_lon, center_lat)
        else:
            center = center_lon
    elif center is None:
        center = 180

    if projections is None:
        projections = ['AIT', 'MOL', 'SFL', 'CAR', 'PAR', 'PCO']

    if data is None:
        nside = nside if nside is not None else 32
        # Local RandomState (not the global np.random) so the synthetic
        # field is reproducible without perturbing the caller's RNG. Seed
        # 0 via RandomState matches the legacy global-seed sequence.
        rng = np.random.RandomState(0)
        data = hp.smoothing(rng.normal(0, 1, hp.nside2npix(nside)),
                            sigma=np.radians(7))

    n = len(projections)
    nrows = int(np.ceil(n / ncols))

    if figsize is None:
        figsize = (5 * ncols, 3 * nrows)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    axes: list[Any] = []

    # ``center`` may be a float or a (lon, lat) tuple (when center_lat was
    # given). The all-sky HEALPix grid only needs the longitude center —
    # pass that scalar so a tuple center doesn't break ``center_deg - 180``.
    center_lon_deg = center[0] if isinstance(center, (tuple, list)) else center
    plonc, platc, pvals = healpix_to_celestial(
        data, 'allsky', center_lon_deg, (1000, 500), np.nan)

    for i, proj in enumerate(projections):
        try:
            ax = make_wcs_frame((nrows, ncols, i + 1),
                                proj, center=center, frame=frame, fig=fig)
            # Blank cells that bridge a projection seam so interrupted
            # projections (HPX/XPH/quadcube) don't smear across boundaries;
            # no-op on continuous projections.
            panel_vals = mask_seam_crossing_quads(ax, plonc, platc, pvals)
            qm = ax.pcolormesh(plonc, platc, panel_vals,
                               transform=ax.get_transform('world'),
                               cmap=cmap, **pcolor_kwargs)
            clip_to_projection_boundary(ax, qm)

            # Resolve name for display
            try:
                _, info = _resolve_projection(proj)
                ax.set_title(f"{info.fits_code or proj}\n{info.description}",
                             fontsize=9)
            except ValueError:
                ax.set_title(proj, fontsize=9)

            # Tick-label fontsize is set by make_wcs_frame's
            # ``auto_fontsize=True`` default — sized to the per-panel
            # axes width, so we don't override here.
            axes.append(ax)
        except Exception:
            # Skip non-FITS or broken projections gracefully
            ax_plain = fig.add_subplot(nrows, ncols, i + 1)
            ax_plain.text(0.5, 0.5, f"{proj}\n(not available)",
                          ha='center', va='center', fontsize=10,
                          color='0.5', transform=ax_plain.transAxes)
            ax_plain.set_xticks([])
            ax_plain.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches='tight')

    return fig, axes
