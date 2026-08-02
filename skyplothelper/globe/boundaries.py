"""Coastlines, tectonic plates, and time-zone overlays.

Boundary data is loaded from ``.npz`` files. ``prepare_earth_data()``
regenerates the coastline / tectonic-plate / time-zone files locally from
their public sources (Natural Earth + Bird 2003) — the supported way to
obtain them. ``fetch_boundary_data()`` is a download-convenience layer for
pulling pre-built files from a host; the canonical hosted copies aren't
published yet, so it currently directs callers to ``prepare_earth_data()``
unless a ``base_url=`` mirror is supplied.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import urllib.error
import urllib.request
import warnings
from collections.abc import Callable, Sequence
from importlib import resources
from typing import Any, cast

import matplotlib.pyplot as plt  # noqa: F401
import numpy as np
import numpy.typing as npt

from .._stroke import _stroke_path_effects
from .frame import make_globe_frame  # noqa: F401  (used by demo paths)
from .plotting import _is_globe_axes  # noqa: F401
from .spherical import (
    great_circle_arc,
    orthographic_forward,
    orthographic_visibility,
)

# Module directory for locating bundled data files (used by
# prepare_earth_data() when no explicit output_dir is given).
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Search paths ----
# Order: skyplothelper/data (package-bundled) -> ~/.skyplothelper/data (user
# cache populated by fetch_boundary_data) -> module-local fallbacks.
_PACKAGE_DATA_DIR = str(resources.files("skyplothelper").joinpath("data"))
_USER_CACHE_DIR = os.path.expanduser("~/.skyplothelper/data")
_DATA_SEARCH_PATHS = [
    _PACKAGE_DATA_DIR,
    _USER_CACHE_DIR,
]


# ---- GitHub-hosted boundary data ----
# Canonical raw-content location of the bundled boundary .npz files once
# the package is published. Override per-call via
# ``fetch_boundary_data(base_url=...)`` (an http(s) prefix or a local
# offline mirror). The ``raw.githubusercontent.com`` host serves file
# contents directly (the ``github.com/.../raw/...`` form just 302s here).
_BOUNDARY_DATA_BASE_URL = (
    "https://raw.githubusercontent.com/pjcigan/skyplothelper/main/data")

# The Earth-boundary .npz files aren't hosted in the repo yet, so the
# default download source has nothing to serve. Flip to True (and pin the
# SHA-256 hashes below) once the files are published. Until then
# ``fetch_boundary_data()`` fails fast with a pointer to
# ``prepare_earth_data()`` rather than emitting confusing 404s — a caller
# who passes ``base_url=`` (their own mirror) bypasses this gate.
_BOUNDARY_DATA_PUBLISHED = False

# SHA-256 hashes stay ``None`` until the files are published and pinned;
# ``verify=`` is a no-op while a hash is None.
# Filenames match the bundled artifacts in ``skyplothelper/data`` (and
# what the loaders / ``prepare_earth_data`` read and write). ``coastlines.npz``
# holds both resolutions as the ``coast_110m`` / ``coast_50m`` keys.
_BOUNDARY_DATA_FILES = (
    "coastlines.npz",
    "tectonic_plates.npz",
    "time_zones.npz",
)
_BOUNDARY_DATA_URLS = {
    # filename: (url, sha256_hash_or_None)
    name: (f"{_BOUNDARY_DATA_BASE_URL}/{name}", None)
    for name in _BOUNDARY_DATA_FILES
}


def _progress_hook(name: str) -> Callable[[int, int, int], None]:
    """Build a ``urlretrieve`` reporthook that draws a simple text bar.

    Kept dependency-free (no tqdm): writes a carriage-return-updated bar
    to stderr so it doesn't pollute stdout / notebook output streams.
    """
    def hook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            frac = min(downloaded / total_size, 1.0)
            filled = int(40 * frac)
            bar = "#" * filled + "-" * (40 - filled)
            sys.stderr.write(
                f"\r  {name}: [{bar}] {frac * 100:5.1f}% "
                f"({downloaded / 1024:.0f}/{total_size / 1024:.0f} KB)")
        else:
            # Unknown length (no Content-Length header): show bytes only.
            sys.stderr.write(f"\r  {name}: {downloaded / 1024:.0f} KB")
        sys.stderr.flush()
    return hook


def _sha256(path: str) -> str:
    """SHA-256 of a file, read in chunks to bound memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_boundary_data(filename: str | None = None, dest: str | None = None,
                        force: bool = False, verify: bool = True,
                        base_url: str | None = None, retries: int = 2,
                        progress: bool = True) -> list[str]:
    """Download Earth-boundary data files from the GitHub repo on first use.

    .. note::

       The hosted ``.npz`` files are not published yet, so the default
       source (``base_url=None``) warns and returns an empty list,
       directing you to :func:`prepare_earth_data`, which regenerates the
       same files locally from their public sources. Pass ``base_url=`` to
       fetch from your own mirror in the meantime.

    Parameters
    ----------
    filename : str or None
        A specific filename (e.g. ``"coastlines.npz"``) or None to
        fetch all known files.
    dest : str or None
        Destination directory. Defaults to the package's ``data/``
        directory if writable, else ``~/.skyplothelper/data/``.
    force : bool
        If False (default), skips files that already exist locally.
    verify : bool
        If True and a SHA-256 hash is registered for the file, verify
        the download and discard it on mismatch.
    base_url : str or None
        Override the download source. May be an ``http(s)://`` URL prefix
        (the filename is appended) or a path to a local directory mirror
        (files are copied from there). Useful behind a firewall or for
        offline installs from a pre-downloaded cache.
    retries : int
        Number of additional attempts on network failure or hash mismatch
        (default 2, i.e. up to 3 tries total per file).
    progress : bool
        If True (default), draw a text progress bar to stderr during each
        download. Ignored for local-mirror copies.

    Returns
    -------
    fetched : list of str
        Absolute paths of files now present on disk.
    """
    if filename is None:
        targets = list(_BOUNDARY_DATA_URLS.keys())
    elif filename in _BOUNDARY_DATA_URLS:
        targets = [filename]
    else:
        raise ValueError(
            f"Unknown boundary data file '{filename}'. "
            f"Known: {list(_BOUNDARY_DATA_URLS.keys())}")

    # The default hosted source isn't populated yet. Rather than let every
    # download 404, warn and return empty — the caller can fall back to the
    # local generator. A caller who supplies their own ``base_url`` knows
    # where the data lives, so let that path through.
    if base_url is None and not _BOUNDARY_DATA_PUBLISHED:
        warnings.warn(
            "Hosted Earth-boundary data is not published yet, so there is "
            "nothing to download from the default source. Generate the "
            "files locally with skyplothelper.prepare_earth_data() "
            "(coastlines and time zones require cartopy; tectonic plates "
            "need only the standard library), or pass base_url= to fetch "
            "from your own mirror.")
        return []

    if dest is None:
        # Try package data dir first; if not writable, fall back to user cache
        try:
            os.makedirs(_PACKAGE_DATA_DIR, exist_ok=True)
            test_path = os.path.join(_PACKAGE_DATA_DIR, ".write_test")
            with open(test_path, "w") as f:
                f.write("")
            os.remove(test_path)
            dest = _PACKAGE_DATA_DIR
        except (OSError, PermissionError):
            os.makedirs(_USER_CACHE_DIR, exist_ok=True)
            dest = _USER_CACHE_DIR
    else:
        os.makedirs(dest, exist_ok=True)

    # A local directory passed as base_url is treated as an offline mirror.
    local_mirror = (base_url is not None and os.path.isdir(base_url))

    fetched = []
    for name in targets:
        out_path = os.path.join(dest, name)
        if os.path.isfile(out_path) and not force:
            fetched.append(out_path)
            continue

        default_url, expected_sha = _BOUNDARY_DATA_URLS[name]
        if local_mirror and base_url is not None:
            source = os.path.join(base_url, name)
        elif base_url is not None:
            source = base_url.rstrip("/") + "/" + name
        else:
            source = default_url

        ok = False
        for attempt in range(retries + 1):
            try:
                if local_mirror:
                    if not os.path.isfile(source):
                        raise FileNotFoundError(
                            f"{name} not found in mirror {base_url}")
                    shutil.copyfile(source, out_path)
                else:
                    hook = _progress_hook(name) if progress else None
                    urllib.request.urlretrieve(source, out_path, hook)
                    if progress:
                        sys.stderr.write("\n")
                        sys.stderr.flush()
            except (urllib.error.URLError, OSError) as e:
                if attempt < retries:
                    continue
                warnings.warn(
                    f"Could not fetch {name} from {source} after "
                    f"{retries + 1} attempt(s): {e}")
                break

            if verify and expected_sha is not None:
                actual = _sha256(out_path)
                if actual != expected_sha:
                    # Discard the bad file so a later load() can't pick up
                    # corrupt data, then retry if attempts remain.
                    try:
                        os.remove(out_path)
                    except OSError:
                        pass
                    if attempt < retries:
                        continue
                    warnings.warn(
                        f"Hash mismatch for {name}: expected {expected_sha}, "
                        f"got {actual}. File discarded; re-run to retry.")
                    break

            ok = True
            break

        if ok:
            fetched.append(out_path)
    return fetched


def _find_data_file(filename: str) -> str | None:
    """Search for a data file in standard locations.

    If not found locally, suggests calling ``fetch_boundary_data()``.
    """
    for d in _DATA_SEARCH_PATHS:
        path = os.path.join(d, filename)
        if os.path.isfile(path):
            return path
    return None


def load_boundary_data(filename: str, key: str | None = None) -> np.ndarray:
    """
    Load boundary data from a .npz file.

    The file should contain NaN-separated (lon, lat) segment arrays with
    shape (N, 2). NaN rows separate distinct line segments.

    Parameters
    ----------
    filename : str
        Filename (searched in standard data paths) or full path.
    key : str or None
        Key within the .npz file. If None, uses the first key.

    Returns
    -------
    data : ndarray, shape (N, 2)
        Columns are [longitude, latitude] in degrees. NaN rows separate
        segments.
    """
    path = filename if os.path.isfile(filename) else _find_data_file(filename)
    if path is None:
        raise FileNotFoundError(
            f"Data file '{filename}' not found. Searched: {_DATA_SEARCH_PATHS}. "
            f"Run prepare_earth_data() to download and process boundary data.")
    d = np.load(path, allow_pickle=False)
    if key is None:
        key = list(d.keys())[0]
    return d[key]


def split_segments(data: npt.ArrayLike) -> list[np.ndarray]:
    """
    Split a NaN-separated (N, 2) array into a list of segment arrays.

    Parameters
    ----------
    data : ndarray, shape (N, 2)
        NaN rows separate segments.

    Returns
    -------
    segments : list of ndarray
        Each element has shape (M, 2).
    """
    data = np.asarray(data)
    nan_mask = np.isnan(data[:, 0])
    segments = []
    start = 0
    for i in range(len(data)):
        if nan_mask[i]:
            if i > start:
                segments.append(data[start:i])
            start = i + 1
    if start < len(data):
        segments.append(data[start:])
    return segments


def plot_boundaries_globe(ax: Any, data: npt.ArrayLike, wcs: Any = None,
                          hemisphere_only: bool | None = None,
                          center_lon: float | None = None,
                          center_lat: float | None = None,
                          densify: bool = True, n_interp: int = 3,
                          stroke_color: Any = None,
                          stroke_lw: float | None = None,
                          **kwargs: Any) -> list[Any]:
    """
    Plot boundary line segments (coastlines, tectonic plates, etc.) on a
    WCSAxes globe.

    Parameters
    ----------
    ax : WCSAxes
        Axes from make_globe_frame().
    data : ndarray, shape (N, 2)
        NaN-separated [lon, lat] boundary data (degrees).
    wcs : WCS or None
        If None, extracted from ax.wcs (None for non-FITS projections).
    hemisphere_only : bool or None
        Cull back-hemisphere segments. Default None auto-detects: cull on a
        real globe (orthographic/zenithal), show the whole surface on a flat
        planet map (CAR, Robinson, …).
    center_lon, center_lat : float or None
        Projection center.
    densify : bool
        Interpolate between points for smoother curves.
    n_interp : int
        Interpolation points per segment pair.
    **kwargs
        Passed to ax.plot() for each segment.

    Returns
    -------
    lines : list of Line2D
    """
    segments = split_segments(data)
    if hemisphere_only is None:
        # Only a real globe (orthographic/zenithal) has a hidden far side to
        # cull; a flat planet map (CAR, Robinson, …) shows the whole surface.
        hemisphere_only = _is_globe_axes(ax)
    if wcs is None:
        wcs = getattr(ax, 'wcs', None)
    # The projection center is only needed to cull the back hemisphere on a
    # globe. Non-FITS projections (Robinson & co.) have no WCS object, but they
    # are never globes, so the crval lookup is never reached for them.
    if hemisphere_only and wcs is not None:
        if center_lon is None:
            center_lon = wcs.wcs.crval[0]
        if center_lat is None:
            center_lat = wcs.wcs.crval[1]
    elif hemisphere_only and wcs is None:
        hemisphere_only = False  # no WCS to derive a center from; nothing to cull

    kwargs.setdefault('transform', ax.get_transform('world'))
    _pe = _stroke_path_effects(stroke_color, stroke_lw)
    if _pe is not None:
        kwargs.setdefault('path_effects', _pe)
    all_lines = []

    for seg in segments:
        lons, lats = seg[:, 0], seg[:, 1]

        # Densify by great-circle interpolation
        if densify and len(lons) > 1:
            new_lons, new_lats = [lons[0]], [lats[0]]
            for i in range(len(lons) - 1):
                sl, sa = great_circle_arc(
                    lons[i], lats[i], lons[i+1], lats[i+1],
                    n_pts=n_interp + 2)
                new_lons.extend(sl[1:])
                new_lats.extend(sa[1:])
            lons = np.array(new_lons)
            lats = np.array(new_lats)

        if hemisphere_only:
            vis = orthographic_visibility(lons, lats, center_lon, center_lat)
            lons = np.where(vis, lons, np.nan)
            lats = np.where(vis, lats, np.nan)
        else:
            # Flat planet map: break the line where it wraps across the
            # projection seam so it doesn't streak across the frame (a globe
            # never reaches its own seam — the far side is culled above). Reuse
            # the same display-space seam splitter the celestial line verbs use
            # (``sph.plot`` & co.) rather than a bespoke one.
            from ..plotting import _split_at_seam
            lons, lats = _split_at_seam(ax, np.asarray(lons), np.asarray(lats))

        lines = ax.plot(lons, lats, **kwargs)
        all_lines.extend(lines)

    return all_lines


def plot_boundaries_ortho(ax: Any, data: npt.ArrayLike, lon_0: float = 0.,
                          lat_0: float = 0., R: float = 1.,
                          show_back: bool = False,
                          front_kwargs: dict[str, Any] | None = None,
                          back_kwargs: dict[str, Any] | None = None,
                          densify: bool = True,
                          n_interp: int = 3) -> list[Any]:
    """
    Plot boundary data on a plain (non-WCS) orthographic axes, with
    optional front/back hemisphere styling.

    Parameters
    ----------
    ax : matplotlib Axes
        Regular matplotlib axes (not WCSAxes). Should have equal aspect.
    data : ndarray, shape (N, 2)
        NaN-separated [lon, lat] boundary data.
    lon_0, lat_0 : float
        Projection center in degrees.
    R : float
        Sphere radius in plot units.
    show_back : bool
        Whether to show back-hemisphere lines.
    front_kwargs, back_kwargs : dict or None
        Plot styling for front/back hemisphere segments.
    densify : bool
        Interpolate between points.
    n_interp : int
        Points per interpolation segment.

    Returns
    -------
    lines : list of Line2D
    """
    # Front/back tones are deliberately muted and deliberately *paired* —
    # the back-hemisphere line is the fainter of the two. Mirrored together
    # so that relationship survives on a dark canvas, where a fixed '0.3'
    # front line disappears and a '0.8' back line shouts.
    from ..style import muted_ink
    if front_kwargs is None:
        front_kwargs = dict(color=muted_ink(ax, light='0.3'), lw=0.6)
    if back_kwargs is None:
        back_kwargs = dict(color=muted_ink(ax, light='0.8'), lw=0.3,
                           ls='--', alpha=0.4)

    segments = split_segments(data)
    all_lines = []

    for seg in segments:
        lons, lats = seg[:, 0], seg[:, 1]

        if densify and len(lons) > 1:
            new_lons, new_lats = [lons[0]], [lats[0]]
            for i in range(len(lons) - 1):
                sl, sa = great_circle_arc(
                    lons[i], lats[i], lons[i+1], lats[i+1],
                    n_pts=n_interp + 2)
                new_lons.extend(sl[1:])
                new_lats.extend(sa[1:])
            lons = np.array(new_lons)
            lats = np.array(new_lats)

        xy = orthographic_forward(lons, lats, lon_0, lat_0, R)
        x, y = xy[:, 0], xy[:, 1]
        vis = orthographic_visibility(lons, lats, lon_0, lat_0)

        # Front hemisphere
        xf, yf = x.copy(), y.copy()
        xf[~vis] = np.nan
        yf[~vis] = np.nan
        lines = ax.plot(xf, yf, **front_kwargs)
        all_lines.extend(lines)

        # Back hemisphere
        if show_back:
            xb, yb = x.copy(), y.copy()
            xb[vis] = np.nan
            yb[vis] = np.nan
            lines = ax.plot(xb, yb, **back_kwargs)
            all_lines.extend(lines)

    return all_lines


def fill_boundaries_globe(ax: Any, data: npt.ArrayLike,
                          facecolor: Any = '0.85', edgecolor: Any = 'none',
                          alpha: float | None = None, zorder: float = 1,
                          stroke_color: Any = None,
                          stroke_lw: float | None = None,
                          clip: str = 'auto', min_ring_points: int = 4,
                          **kwargs: Any) -> list[Any]:
    """Fill the CLOSED rings of NaN-separated ``[lon, lat]`` boundary data
    using the spherical-polygon region machinery.

    Each closed ring is routed through
    :func:`~skyplothelper.geometry.add_spherical_polygon`, so the fills inherit
    the region engine's antimeridian clipping, on-globe limb clipping, and hole
    handling — the same pipeline as the celestial region tools. Open segments
    (coastline pieces clipped at the data edge, or tectonic-plate boundary arcs)
    cannot bound a fill and are skipped.

    Parameters
    ----------
    ax : WCSAxes
        Target axes (a globe or a flat planet frame).
    data : ndarray, shape (N, 2)
        NaN-separated ``[lon, lat]`` boundary data (degrees).
    facecolor, edgecolor, alpha, zorder :
        Patch style. ``edgecolor='none'`` by default so filled areas read as
        solid regions rather than outlined ones; ``zorder=1`` keeps them under
        typical overlays.
    clip : str
        Seam-handling pipeline forwarded to ``add_spherical_polygon`` (default
        ``'auto'`` → the d3 antimeridian clipper for closed regions).
    min_ring_points : int
        Skip rings with fewer than this many vertices (default 4).
    **kwargs
        Forwarded to ``add_spherical_polygon`` (e.g. ``hatch``).

    Returns
    -------
    patches : list
        The artist(s) returned by ``add_spherical_polygon`` for each ring.
    """
    from ..geometry.shapes import add_spherical_polygon

    if getattr(ax, 'wcs', None) is None:
        raise NotImplementedError(
            "Filling geographic overlays on a non-FITS projection (Robinson, "
            "Eckert, Winkel Tripel, ...) is not supported yet: the region fill "
            "pipeline currently requires a FITS WCS. Use a FITS projection "
            "(e.g. 'CAR', 'MOL', 'MER') for filled maps, or the default outline "
            "rendering (non-FITS fill is planned via the region-projector "
            "unification).")

    _pe = _stroke_path_effects(stroke_color, stroke_lw)
    if _pe is not None:
        kwargs.setdefault('path_effects', _pe)

    segments = split_segments(data)
    patches = []
    for seg in segments:
        lons = np.asarray(seg[:, 0], dtype=float)
        lats = np.asarray(seg[:, 1], dtype=float)
        if len(lons) < min_ring_points:
            continue
        # Only closed rings (first vertex == last) can bound a fill. The
        # bundled polygon data is already densified, so we pass resolution=0
        # (no extra edge subdivision) and let the region pipeline handle
        # seam + limb clipping.
        if not (abs(lons[0] - lons[-1]) < 1e-6
                and abs(lats[0] - lats[-1]) < 1e-6):
            continue
        patches.append(add_spherical_polygon(
            ax, lons, lats, resolution=0, clip=clip,
            facecolor=facecolor, edgecolor=edgecolor, alpha=alpha,
            zorder=zorder, **kwargs))
    return patches


def plot_coastlines(ax: Any, resolution: str = '110m', wcs_mode: bool = True,
                    **kwargs: Any) -> list[Any]:
    """
    Draw coastlines as outlines on a globe or flat planet frame.

    Loads coastline LINE data (Natural Earth, public domain). For FILLED land
    use :func:`plot_land` — coastline lines are clipped at the antimeridian and
    so don't bound continent fills; the land-polygon dataset does.

    Parameters
    ----------
    ax : Axes
        WCSAxes (if wcs_mode=True) or regular Axes.
    resolution : str
        '110m' or '50m'. Maps to data file key.
    wcs_mode : bool
        If True, uses plot_boundaries_globe. If False, uses plot_boundaries_ortho.
    **kwargs
        Passed to the underlying plot function.

    Returns
    -------
    lines : list of Line2D
    """
    fname = 'coastlines.npz'
    key = f'coast_{resolution}'
    data = load_boundary_data(fname, key=key)

    kwargs.setdefault('color', '0.25')
    kwargs.setdefault('lw', 0.6)

    if wcs_mode:
        return plot_boundaries_globe(ax, data, **kwargs)
    else:
        fk = {k: kwargs.pop(k) for k in list(kwargs) if k in
              ['lon_0', 'lat_0', 'R', 'show_back', 'front_kwargs', 'back_kwargs',
               'densify', 'n_interp']}
        fk['front_kwargs'] = dict(color=kwargs.get('color', '0.25'),
                                   lw=kwargs.get('lw', 0.6))
        return plot_boundaries_ortho(ax, data, **fk)


def plot_land(ax: Any, resolution: str = '110m', facecolor: Any = '0.85',
              **kwargs: Any) -> list[Any]:
    """Fill continents, islands, and other land areas as solid land.

    Uses the bundled Natural Earth LAND polygons (public domain) — proper
    closed, antimeridian-safe polygons — filled via the spherical-region
    machinery (:func:`fill_boundaries_globe`). Inland lakes (Great Lakes, etc.)
    are treated as land here, matching the Natural Earth land product; overlay
    them as water with :func:`plot_lakes`. Add a ``plot_coastlines(ax)`` call
    for a coastline stroke on top.

    Parameters
    ----------
    ax : WCSAxes
        A globe or flat planet frame on a FITS projection. (Non-FITS projections
        like Robinson are not yet supported for fills.)
    resolution : str
        Data resolution. Only ``'110m'`` is bundled; heavier resolutions can be
        generated into the user cache with :func:`prepare_earth_data` (or reach
        for cartopy for cartography-grade detail).
    facecolor : color
        Land fill color (default ``'0.85'``).
    **kwargs
        Forwarded to :func:`fill_boundaries_globe`.

    Returns
    -------
    patches : list
    """
    data = load_boundary_data('land.npz', key=f'land_{resolution}')
    return fill_boundaries_globe(ax, data, facecolor=facecolor, **kwargs)


def plot_lakes(ax: Any, resolution: str = '110m', facecolor: Any = '#a6cee3',
               **kwargs: Any) -> list[Any]:
    """Fill lakes (Great Lakes, Caspian, Baikal, ...) as water.

    Uses the bundled Natural Earth LAKES polygons (public domain) — a layer
    separate from :func:`plot_land` (in the Natural Earth model, LAND treats
    inland lakes as land, so lakes are overlaid here). The default ``facecolor``
    is a light water blue; set it to your ocean / axes background color to make
    the lakes read as holes punched in the land underneath.

    Parameters
    ----------
    ax : WCSAxes
        A globe or flat planet frame on a FITS projection.
    resolution : str
        Only ``'110m'`` is bundled (see :func:`prepare_earth_data` for more).
    facecolor : color
        Lake fill color (default light water blue ``'#a6cee3'``).
    **kwargs
        Forwarded to :func:`fill_boundaries_globe`.

    Returns
    -------
    patches : list
    """
    data = load_boundary_data('lakes.npz', key=f'lakes_{resolution}')
    return fill_boundaries_globe(ax, data, facecolor=facecolor, **kwargs)


def plot_rivers(ax: Any, resolution: str = '110m', wcs_mode: bool = True,
                **kwargs: Any) -> list[Any]:
    """Draw major rivers (Nile, Amazon, Mississippi, ...) as centerlines.

    Uses the bundled Natural Earth ``rivers_lake_centerlines`` (public domain).
    This is a line overlay (like :func:`plot_coastlines`), not a fill. At the
    bundled ``'110m'`` resolution only the ~13 major rivers are present; finer
    detail can be generated with :func:`prepare_earth_data` (or via cartopy).

    Parameters
    ----------
    ax : Axes
        WCSAxes (if wcs_mode=True) or regular Axes.
    resolution : str
        Data resolution (only ``'110m'`` is bundled).
    wcs_mode : bool
        If True, uses plot_boundaries_globe. If False, plot_boundaries_ortho.
    **kwargs
        Passed to the underlying plot function (``color``, ``lw``, ...).

    Returns
    -------
    lines : list of Line2D
    """
    data = load_boundary_data('rivers.npz', key=f'rivers_{resolution}')

    kwargs.setdefault('color', '#4a90d9')
    kwargs.setdefault('lw', 0.5)

    if wcs_mode:
        return plot_boundaries_globe(ax, data, **kwargs)
    else:
        fk = {k: kwargs.pop(k) for k in list(kwargs) if k in
              ['lon_0', 'lat_0', 'R', 'show_back', 'front_kwargs', 'back_kwargs',
               'densify', 'n_interp']}
        fk['front_kwargs'] = dict(color=kwargs.get('color', '#4a90d9'),
                                   lw=kwargs.get('lw', 0.5))
        return plot_boundaries_ortho(ax, data, **fk)


def _earth_clip_path(ax: Any, resolution: str = '110m',
                     ocean: bool = False) -> Any:
    """Build a matplotlib clip ``Path`` (in ``ax.transData``) covering land — or
    the ocean complement — from the bundled land polygons via the region
    machinery.  Shared by :func:`clip_to_land` / :func:`clip_to_ocean`.
    """
    if getattr(ax, 'wcs', None) is None:
        raise NotImplementedError(
            "clip_to_land / clip_to_ocean need a FITS-projection frame (ax.wcs "
            "is None on non-FITS projections like Robinson). Use a FITS "
            "projection (CAR/MOL/AIT/...) or a globe.")

    import shapely

    from ..geometry._frame_geom import _geom_to_clip_path
    from ..geometry._projector import WCSAxesProjector

    # Batch-project each land ring and union ONCE (shapely.unary_union) — much
    # faster than adding 127 polygons to a CompoundRegion, which unions
    # incrementally. The resulting path drops into the same clip machinery as
    # CompoundRegion.clip.
    proj = WCSAxesProjector(ax)
    data = load_boundary_data('land.npz', key=f'land_{resolution}')
    geoms = []
    for seg in split_segments(data):
        lons = np.asarray(seg[:, 0], dtype=float)
        lats = np.asarray(seg[:, 1], dtype=float)
        if len(lons) < 4 or not (abs(lons[0] - lons[-1]) < 1e-6
                                 and abs(lats[0] - lats[-1]) < 1e-6):
            continue
        g = proj.project_polygon(lons, lats, clip='d3', min_piece_area=0.0)
        if g is not None and not g.is_empty:
            geoms.append(g)
    land = shapely.unary_union(geoms)
    return _geom_to_clip_path(land, proj.frame_polygon, complement=ocean)


def _apply_earth_clip(ax: Any, artists: Any, resolution: str,
                      ocean: bool) -> Any:
    path = _earth_clip_path(ax, resolution=resolution, ocean=ocean)
    if artists is not None:
        seq = artists if isinstance(artists, (list, tuple)) else [artists]
        for art in seq:
            art.set_clip_path(path, ax.transData)
    return path


def clip_to_land(ax: Any, artists: Any = None, *,
                 resolution: str = '110m') -> Any:
    """Clip matplotlib artist(s) to land, using the bundled land polygons.

    Masks overlays drawn on a planet frame — location markers, quiver / vector
    arrows, a geographic raster field — so they appear only over **land**. The
    clip region is the union of the Natural Earth land polygons, built through
    the same spherical-region pipeline as the celestial region tools (this is
    the region ``contains``/set-algebra machinery used as a matplotlib clip
    path). See :func:`clip_to_ocean` for the complement.

    Parameters
    ----------
    ax : WCSAxes
        A FITS-projection planet frame (non-FITS projections are not yet
        supported — the projector needs a FITS WCS).
    artists : Artist or list of Artist, optional
        Artist(s) to clip in place (e.g. the return of ``ax.scatter`` /
        ``ax.quiver`` / ``ax.imshow``). If omitted, nothing is clipped and only
        the clip path is returned.
    resolution : str
        Land-polygon resolution (only ``'110m'`` is bundled).

    Returns
    -------
    path : matplotlib.path.Path
        The land clip path (in ``ax.transData``); apply to more artists with
        ``artist.set_clip_path(path, ax.transData)``.
    """
    return _apply_earth_clip(ax, artists, resolution, ocean=False)


def clip_to_ocean(ax: Any, artists: Any = None, *,
                  resolution: str = '110m') -> Any:
    """Clip matplotlib artist(s) to ocean (the complement of land).

    The mirror of :func:`clip_to_land` — masks overlays so they appear only
    over **water** (the frame minus the land polygons). Same parameters and
    return.
    """
    return _apply_earth_clip(ax, artists, resolution, ocean=True)


def plot_tectonic_plates(ax: Any, wcs_mode: bool = True, *,
                         fill: bool = False,
                         **kwargs: Any) -> list[Any]:
    """
    Convenience function to plot tectonic plate boundaries on a globe.

    Parameters
    ----------
    ax : Axes
        WCSAxes or regular Axes.
    wcs_mode : bool
        If True, uses plot_boundaries_globe.
    fill : bool
        Not supported — the bundled tectonic data is plate *boundary arcs*,
        not closed plate polygons, so there is nothing to fill. Passing
        ``fill=True`` raises ``NotImplementedError`` (closed-plate fill is
        planned for a future release).
    **kwargs
        Passed to the underlying plot function.

    Returns
    -------
    lines : list of Line2D
    """
    if fill:
        raise NotImplementedError(
            "plot_tectonic_plates fill is not supported: the bundled tectonic "
            "data is plate boundary arcs, not closed plate polygons, so there "
            "is nothing to fill. Use the default line rendering; closed-plate "
            "fill is planned for a future release.")

    fname = 'tectonic_plates.npz'
    data = load_boundary_data(fname)

    kwargs.setdefault('color', '#CC4444')
    kwargs.setdefault('lw', 0.8)

    if wcs_mode:
        return plot_boundaries_globe(ax, data, **kwargs)
    else:
        fk = {k: kwargs.pop(k) for k in list(kwargs) if k in
              ['lon_0', 'lat_0', 'R', 'show_back', 'front_kwargs', 'back_kwargs',
               'densify', 'n_interp']}
        fk['front_kwargs'] = dict(color=kwargs.get('color', '#CC4444'),
                                   lw=kwargs.get('lw', 0.8))
        return plot_boundaries_ortho(ax, data, **fk)


def plot_time_zones(ax: Any, wcs_mode: bool = True, *,
                    fill: bool = False, facecolor: Any = None,
                    edgecolor: Any = None, alpha: float | None = None,
                    **kwargs: Any) -> list[Any]:
    """
    Convenience function to plot world time-zone boundaries on a map.

    Uses the Natural Earth 10m time zones cultural dataset (public domain),
    which supplies time-zone boundaries as polygons. This function plots the
    polygon edges as light reference lines — a neat companion to coastlines
    for "where on Earth was this taken" overlays on astronomical plots, or
    for planning observation timing across sites.

    Parameters
    ----------
    ax : Axes
        WCSAxes (if ``wcs_mode=True``) or regular Axes with lon/lat limits.
    wcs_mode : bool
        If True, uses :func:`plot_boundaries_globe`. If False, uses
        :func:`plot_boundaries_ortho` (for mplot3d / plain matplotlib).
    **kwargs
        Passed to the underlying plot function. Common overrides:
        ``color`` (default ``'0.55'``) and ``lw`` (default ``0.4``).

    Returns
    -------
    lines : list of Line2D

    Notes
    -----
    Requires the file ``time_zones.npz`` to be present in the package data
    directory. Run :func:`prepare_earth_data` with ``include_timezones=True``
    to download and cache it. Attribution: Natural Earth (public domain).
    """
    fname = 'time_zones.npz'
    data = load_boundary_data(fname)

    if fill or facecolor is not None:
        if not wcs_mode:
            raise ValueError(
                "plot_time_zones(fill=True) needs wcs_mode=True: the fill "
                "routes through the spherical-region machinery, which requires "
                "a WCSAxes projection.")
        return fill_boundaries_globe(
            ax, data,
            facecolor='0.85' if facecolor is None else facecolor,
            edgecolor='none' if edgecolor is None else edgecolor,
            alpha=alpha, **kwargs)

    kwargs.setdefault('color', '0.55')
    kwargs.setdefault('lw', 0.4)
    kwargs.setdefault('linestyle', '--')

    if wcs_mode:
        return plot_boundaries_globe(ax, data, **kwargs)
    else:
        fk = {k: kwargs.pop(k) for k in list(kwargs) if k in
              ['lon_0', 'lat_0', 'R', 'show_back', 'front_kwargs', 'back_kwargs',
               'densify', 'n_interp']}
        fk['front_kwargs'] = dict(color=kwargs.get('color', '0.55'),
                                   lw=kwargs.get('lw', 0.4),
                                   linestyle=kwargs.get('linestyle', '--'))
        return plot_boundaries_ortho(ax, data, **fk)


def prepare_earth_data(output_dir: str | None = None,
                       include_tectonic: bool = True,
                       include_coastlines: bool = True,
                       include_timezones: bool = True,
                       include_land: bool = True,
                       include_lakes: bool = True,
                       include_rivers: bool = True,
                       resolutions: Sequence[str] = ('110m', '50m')) -> None:
    """
    Download and prepare Earth boundary data files: coastlines and time
    zones from Natural Earth, plate boundaries from Bird (2003) via the
    fraxen/tectonicplates GeoJSON distribution.

    Requires: cartopy (for Natural Earth download), stdlib urllib (for
    tectonic plate GeoJSON).

    Parameters
    ----------
    output_dir : str or None
        Directory to save .npz files. Default: 'data/' next to this module.
    include_tectonic : bool
        Download tectonic plate boundaries (Bird 2003).
    include_coastlines : bool
        Download Natural Earth coastlines.
    include_timezones : bool
        Download Natural Earth time zones (10m only; it's the only
        resolution available for this dataset).
    include_land, include_lakes : bool
        Download Natural Earth land / lakes POLYGONS (for the filled
        overlays ``plot_land`` / ``plot_lakes``). Antimeridian-touching
        vertices are nudged inward so the region fill's seam clipper reads
        them correctly.
    include_rivers : bool
        Download Natural Earth river / lake centerlines (for ``plot_rivers``).
    resolutions : tuple of str
        Resolutions to include: any of ``'110m'``, ``'50m'``, ``'10m'``. The
        files bundled in the package are ``'110m'`` only for land / lakes /
        rivers (regenerate here for finer detail, or reach for cartopy).

    Notes
    -----
    Sources and licenses (see the header comment at the top of the Boundary
    Data section of this module for full details):

    * **Natural Earth** (coastlines, time zones) — Public Domain. Website:
      https://www.naturalearthdata.com/. Time zones donated to Natural
      Earth by International Mapping Associates, Inc.
    * **Tectonic plates** — Bird, P. (2003),
      doi:10.1029/2001GC000252. GeoJSON re-distribution:
      https://github.com/fraxen/tectonicplates

    This only needs to be run once. The resulting .npz files are small
    (~50 KB to ~5 MB depending on resolution) and can be committed to the
    package data directory for offline use.
    """
    # Print attribution on every run so it's easy to carry over into
    # figure credits. Also nice for reminding users what they're using.
    print("Data sources for prepare_earth_data():")
    print("  * Natural Earth (coastlines, land, lakes, rivers, time zones)"
          " — Public Domain")
    print("      https://www.naturalearthdata.com/")
    print("  * Plate boundaries — Bird (2003), doi:10.1029/2001GC000252")
    print("      via https://github.com/fraxen/tectonicplates")
    print("  Suggested figure credit:")
    print("    'Coastlines, land, lakes, rivers, and time zones: Natural "
          "Earth (public domain).'")
    print("    'Plate boundaries: Bird (2003), doi:10.1029/2001GC000252.'")
    print()

    if output_dir is None:
        # Write to the same location the loader and fetch_boundary_data use
        # (skyplothelper/data), so generate-then-plot works without a manual
        # move. _PACKAGE_DATA_DIR is the first entry in _DATA_SEARCH_PATHS.
        output_dir = _PACKAGE_DATA_DIR
    os.makedirs(output_dir, exist_ok=True)

    # -- Coastlines (Natural Earth physical, line geometries) ------------
    if include_coastlines:
        try:
            import cartopy.io.shapereader as shpreader
        except ImportError:
            raise ImportError("cartopy is required to download Natural Earth data. "
                              "Install with: pip install cartopy")

        coast_arrays = {}
        for res in resolutions:
            shp_path = shpreader.natural_earth(resolution=res, category='physical',
                                                name='coastline')
            reader = shpreader.Reader(shp_path)
            segments = []
            for geom in reader.geometries():
                if geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        segments.append(np.array(line.coords))
                elif geom.geom_type == 'LineString':
                    segments.append(np.array(geom.coords))

            all_pts = []
            for seg in segments:
                all_pts.append(seg[:, :2])  # lon, lat only
                all_pts.append(np.array([[np.nan, np.nan]]))
            coast_arrays[f'coast_{res}'] = np.vstack(all_pts).astype(np.float32)
            print(f"  Coastline {res}: {len(segments)} segments, "
                  f"{coast_arrays[f'coast_{res}'].shape[0]} points")

        out_path = os.path.join(output_dir, 'coastlines.npz')
        np.savez_compressed(out_path, **cast("dict[str, Any]", coast_arrays))
        print(f"  Saved: {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")

    # -- Land / lakes (polygons) + rivers (lines), Natural Earth physical --
    # Filled-area overlays (plot_land / plot_lakes) need CLOSED polygons, not
    # the coastline LINES (which Natural Earth clips at the antimeridian,
    # leaving continents as open arcs). We store each polygon's exterior ring;
    # a vertex touching exactly one antimeridian edge is nudged inward so the
    # region fill's d3 clipper doesn't read a seam-touch as a seam-crossing
    # (which would close the polygon across the whole map).
    if include_land or include_lakes or include_rivers:
        try:
            import cartopy.io.shapereader as shpreader
        except ImportError:
            raise ImportError("cartopy is required to download Natural Earth "
                              "data. Install with: pip install cartopy")

        def _poly_rings(res: str, name: str) -> tuple[Any, int]:
            shp = shpreader.natural_earth(resolution=res, category='physical',
                                          name=name)
            pts, n = [], 0
            for geom in shpreader.Reader(shp).geometries():
                parts = (geom.geoms if geom.geom_type == 'MultiPolygon'
                         else [geom])
                for p in parts:
                    n += 1
                    ring = np.asarray(p.exterior.coords)[:, :2].copy()
                    lo = ring[:, 0]
                    tp, tn = np.any(lo > 179.99), np.any(lo < -179.99)
                    if tp != tn:  # touches one seam edge only → nudge inward
                        lo[lo > 179.99] = 179.9
                        lo[lo < -179.99] = -179.9
                    pts.append(ring)
                    pts.append(np.array([[np.nan, np.nan]]))
            return np.vstack(pts).astype(np.float32), n

        for flag, name, fname, key in (
                (include_land, 'land', 'land.npz', 'land'),
                (include_lakes, 'lakes', 'lakes.npz', 'lakes')):
            if not flag:
                continue
            arrs = {}
            for res in resolutions:
                arr, n = _poly_rings(res, name)
                arrs[f'{key}_{res}'] = arr
                print(f"  {name.capitalize()} {res}: {n} polygons, "
                      f"{arr.shape[0]} points")
            out_path = os.path.join(output_dir, fname)
            np.savez_compressed(out_path, **cast("dict[str, Any]", arrs))
            print(f"  Saved: {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")

        if include_rivers:
            arrs = {}
            for res in resolutions:
                shp = shpreader.natural_earth(
                    resolution=res, category='physical',
                    name='rivers_lake_centerlines')
                segs = []
                for geom in shpreader.Reader(shp).geometries():
                    if geom.geom_type == 'MultiLineString':
                        for ln in geom.geoms:
                            segs.append(np.array(ln.coords))
                    elif geom.geom_type == 'LineString':
                        segs.append(np.array(geom.coords))
                all_pts = []
                for seg in segs:
                    all_pts.append(seg[:, :2])
                    all_pts.append(np.array([[np.nan, np.nan]]))
                arrs[f'rivers_{res}'] = np.vstack(all_pts).astype(np.float32)
                print(f"  Rivers {res}: {len(segs)} segments, "
                      f"{arrs[f'rivers_{res}'].shape[0]} points")
            out_path = os.path.join(output_dir, 'rivers.npz')
            np.savez_compressed(out_path, **cast("dict[str, Any]", arrs))
            print(f"  Saved: {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")

    # -- Time zones (Natural Earth cultural, polygon geometries) ----------
    # Natural Earth provides time zones only at 10m, as polygons. We extract
    # the exterior ring of each polygon and save as NaN-separated line
    # segments (same format as coastlines/tectonics for uniform loading).
    # Adjacent polygons share boundaries; the redundant edges render
    # on top of each other and look identical — no special dedup needed.
    if include_timezones:
        try:
            import cartopy.io.shapereader as shpreader
        except ImportError:
            raise ImportError("cartopy is required to download Natural Earth data. "
                              "Install with: pip install cartopy")

        try:
            shp_path = shpreader.natural_earth(resolution='10m', category='cultural',
                                                name='time_zones')
        except Exception as e:
            print(f"  Could not download time-zones data: {e}")
            print("  Skipping time zones.")
            shp_path = None

        if shp_path is not None:
            reader = shpreader.Reader(shp_path)
            tz_segments = []
            tz_labels = []  # optional: UTC-offset label per polygon
            for rec in reader.records():
                geom = rec.geometry
                # Try common NE attribute fields for the UTC offset; fall
                # back to any of the candidate names. Different NE
                # releases have used different field names over time.
                attrs = rec.attributes
                label = (attrs.get('time_zone')
                         or attrs.get('tz_name1st')
                         or attrs.get('name')
                         or attrs.get('zone'))
                if geom.geom_type == 'Polygon':
                    tz_segments.append(np.array(geom.exterior.coords))
                    tz_labels.append(str(label))
                elif geom.geom_type == 'MultiPolygon':
                    for poly in geom.geoms:
                        tz_segments.append(np.array(poly.exterior.coords))
                        tz_labels.append(str(label))

            all_pts = []
            for seg in tz_segments:
                all_pts.append(seg[:, :2])
                all_pts.append(np.array([[np.nan, np.nan]]))
            if all_pts:
                tz_data = np.vstack(all_pts).astype(np.float32)
                out_path = os.path.join(output_dir, 'time_zones.npz')
                np.savez_compressed(out_path, boundaries=tz_data,
                                     labels=np.array(tz_labels))
                print(f"  Time zones (10m): {len(tz_segments)} polygons, "
                      f"{tz_data.shape[0]} points")
                print(f"  Saved: {out_path} "
                      f"({os.path.getsize(out_path)/1024:.0f} KB)")

    # -- Tectonic plates (Bird 2003 via fraxen/tectonicplates GeoJSON) ----
    if include_tectonic:
        import json
        try:
            from urllib.request import urlopen
            url = ("https://raw.githubusercontent.com/fraxen/tectonicplates/"
                   "master/GeoJSON/PB2002_boundaries.json")
            print(f"  Downloading tectonic plate data from {url}...")
            resp = urlopen(url)
            geojson = json.loads(resp.read().decode())
        except Exception as e:
            print(f"  Could not download tectonic data: {e}")
            print("  You can manually download PB2002_boundaries.json from:")
            print("  https://github.com/fraxen/tectonicplates/tree/master/GeoJSON")
            print("  and place it in the data/ directory.")
            return

        segments = []
        for feature in geojson['features']:
            geom = feature['geometry']
            if geom['type'] == 'LineString':
                segments.append(np.array(geom['coordinates'])[:, :2])
            elif geom['type'] == 'MultiLineString':
                for line in geom['coordinates']:
                    segments.append(np.array(line)[:, :2])

        all_pts = []
        for seg in segments:
            all_pts.append(seg)
            all_pts.append(np.array([[np.nan, np.nan]]))
        tect_data = np.vstack(all_pts).astype(np.float32)

        out_path = os.path.join(output_dir, 'tectonic_plates.npz')
        np.savez_compressed(out_path, boundaries=tect_data)
        print(f"  Tectonic plates: {len(segments)} segments, "
              f"{tect_data.shape[0]} points")
        print(f"  Saved: {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")


