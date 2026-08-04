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
# The Earth-boundary .npz files the loaders / ``prepare_earth_data`` read and
# write. All are generate-on-demand (NOT shipped in the wheel/repo — see the
# .gitignore); ``coastlines.npz`` holds both resolutions as the
# ``coast_110m`` / ``coast_50m`` keys. Only the constellation_*.npz are bundled.
_BOUNDARY_DATA_FILES = (
    "coastlines.npz",
    "tectonic_plates.npz",
    "time_zones.npz",
    "land.npz",
    "lakes.npz",
    "rivers.npz",
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


def _require_data_file(filename: str) -> str:
    """:func:`_find_data_file` but raise if the file is absent (so the path is
    a plain ``str`` for ``np.load`` etc.)."""
    path = _find_data_file(filename)
    if path is None:
        raise FileNotFoundError(
            f"{filename} not found in the data search paths; run "
            "skyplothelper.prepare_earth_data() to fetch it.")
    return path


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


def _densify_seam_runs(lons: npt.ArrayLike, lats: npt.ArrayLike,
                       seam_tol: float = 178.0,
                       max_step: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Insert vertices along polygon edges that run **along** the antimeridian.

    The split pieces of an antimeridian-cut MultiPolygon (e.g. the Pacific
    plate) carry a long, straight edge hugging ``±180``. Projected on a curved
    frame (AIT / MOL) that two-vertex edge chords across the frame silhouette
    instead of tracing it. This inserts intermediate vertices (≤ ``max_step``°
    of latitude apart) along such edges so the projected edge follows the curve.

    Only edges whose *both* endpoints sit at ``|lon| ≥ seam_tol`` on the *same*
    side (a small Δlon — an edge along the seam, NOT a seam *crossing*, which
    the d3 clipper handles) are touched, so it stays cheap.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    out_lo = [float(lons[0])]
    out_la = [float(lats[0])]
    for i in range(1, len(lons)):
        l0, l1 = float(lons[i - 1]), float(lons[i])
        a0, a1 = float(lats[i - 1]), float(lats[i])
        if (abs(l0) >= seam_tol and abs(l1) >= seam_tol
                and abs(l1 - l0) < 10.0 and abs(a1 - a0) > max_step):
            n = int(np.ceil(abs(a1 - a0) / max_step))
            for k in range(1, n):
                t = k / n
                out_lo.append(l0 + t * (l1 - l0))
                out_la.append(a0 + t * (a1 - a0))
        out_lo.append(l1)
        out_la.append(a1)
    return np.asarray(out_lo), np.asarray(out_la)


def _closed_rings(data: npt.ArrayLike,
                  min_points: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract the CLOSED rings (first vertex == last) from NaN-separated
    ``[lon, lat]`` polygon data as ``(lons, lats)`` pairs — the fillable rings,
    dropping open arcs. Shared by the region-set-algebra fills."""
    rings = []
    for seg in split_segments(data):
        lons = np.asarray(seg[:, 0], dtype=float)
        lats = np.asarray(seg[:, 1], dtype=float)
        if (len(lons) >= min_points
                and abs(lons[0] - lons[-1]) < 1e-6
                and abs(lats[0] - lats[-1]) < 1e-6):
            rings.append((lons, lats))
    return rings


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
            # hemisphere_only is only left True above when a WCS supplied a
            # center (else it is flipped to False), so both are set here.
            assert center_lon is not None and center_lat is not None
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

    # Each ring routes through add_spherical_polygon, which the G4 region-
    # projector unification made backend-agnostic, so filled geographic overlays
    # now work on the non-FITS custom projections (Robinson/Eckert/…) too — no
    # FITS-WCS requirement.
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


def plot_land(ax: Any, resolution: str = '110m', facecolor: Any = '0.85', *,
              lakes: bool = False, **kwargs: Any) -> list[Any]:
    """Fill continents, islands, and other land areas as solid land.

    Uses Natural Earth LAND polygons (public domain; generate once with
    :func:`prepare_earth_data`) — proper
    closed, antimeridian-safe polygons — filled via the spherical-region
    machinery (:func:`fill_boundaries_globe`). Inland lakes (Great Lakes, etc.)
    are treated as land here, matching the Natural Earth land product; overlay
    them as water with :func:`plot_lakes`, or pass ``lakes=True`` (below) to
    punch them out as true holes. Add a ``plot_coastlines(ax)`` call for a
    coastline stroke on top.

    Parameters
    ----------
    ax : WCSAxes
        A globe or flat planet frame — any projection the region machinery
        supports, including the non-FITS custom projections (Robinson/Eckert/…).
    resolution : str
        Data resolution. ``prepare_earth_data`` generates ``'110m'`` by default;
        pass finer resolutions there (or reach for cartopy for
        cartography-grade detail).
    facecolor : color
        Land fill color (default ``'0.85'``).
    lakes : bool
        If ``True``, subtract the lake polygons from the land (``land − lakes``)
        so lakes render as **true holes** — the ocean / axes background shows
        through, without a separate blue overlay. Computed with the region
        set-algebra (:meth:`CompoundRegion.difference
        <skyplothelper.geometry.CompoundRegion.difference>`), so the holes are
        real geometry (correct under further clipping, membership tests, etc.),
        not just an over-painted patch. Default ``False`` (lakes filled as land,
        matching the Natural Earth land product).
    **kwargs
        Forwarded to :func:`fill_boundaries_globe` (or, with ``lakes=True``, to
        :meth:`CompoundRegion.render` — ``edgecolor`` / ``alpha`` / ``zorder`` /
        ``stroke_color`` / ``stroke_lw`` all apply either way).

    Returns
    -------
    patches : list
    """
    data = load_boundary_data('land.npz', key=f'land_{resolution}')
    if not lakes:
        return fill_boundaries_globe(ax, data, facecolor=facecolor, **kwargs)

    # land − lakes: punch the lakes out as true holes via the region set-algebra
    # (the region machinery dogfooded on real geographic data). Works on every
    # frame the region machinery supports — FITS all-sky, the SIN globe, and the
    # non-FITS custom projections (Robinson/Eckert/…) — via _projector_for_axes.
    from ..geometry.compound import CompoundRegion
    lake_data = load_boundary_data('lakes.npz', key=f'lakes_{resolution}')
    # resolution=0: the bundled rings are already densified (as in
    # fill_boundaries_globe), so skip re-subdividing every edge.
    land_region = CompoundRegion.from_polygons(
        ax, _closed_rings(data), resolution=0)
    lake_region = CompoundRegion.from_polygons(
        ax, _closed_rings(lake_data), resolution=0)
    land_region.difference(lake_region)

    stroke_color = kwargs.pop('stroke_color', None)
    stroke_lw = kwargs.pop('stroke_lw', None)
    _pe = _stroke_path_effects(stroke_color, stroke_lw)
    if _pe is not None:
        kwargs.setdefault('path_effects', _pe)
    kwargs.setdefault('edgecolor', 'none')
    kwargs.setdefault('zorder', 1)
    return land_region.render(facecolor=facecolor, **kwargs)


def plot_lakes(ax: Any, resolution: str = '110m', facecolor: Any = '#a6cee3',
               **kwargs: Any) -> list[Any]:
    """Fill lakes (Great Lakes, Caspian, Baikal, ...) as water.

    Uses Natural Earth LAKES polygons (public domain; generate with
    :func:`prepare_earth_data`) — a layer
    separate from :func:`plot_land` (in the Natural Earth model, LAND treats
    inland lakes as land, so lakes are overlaid here). The default ``facecolor``
    is a light water blue; set it to your ocean / axes background color to make
    the lakes read as holes punched in the land underneath.

    Parameters
    ----------
    ax : WCSAxes
        A globe or flat planet frame on a FITS projection.
    resolution : str
        Only ``'110m'`` by default (see :func:`prepare_earth_data` for more).
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

    Uses Natural Earth ``rivers_lake_centerlines`` (public domain; generate
    with :func:`prepare_earth_data`).
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
    import shapely

    from ..geometry._frame_geom import _geom_to_clip_path
    from ..geometry._projector import _projector_for_axes

    # Batch-project each land ring and union ONCE (shapely.unary_union) — much
    # faster than adding 127 polygons to a CompoundRegion, which unions
    # incrementally. The resulting path drops into the same clip machinery as
    # CompoundRegion.clip. Works on both FITS and non-FITS custom-projection
    # frames — _projector_for_axes returns the matching projector (both emit
    # geometry in ax.transData coords, so the clip path is identical).
    proj = _projector_for_axes(ax)
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
        A planet frame — any projection the region machinery supports, including
        the non-FITS custom projections (Robinson/Eckert/…).
    artists : Artist or list of Artist, optional
        Artist(s) to clip in place (e.g. the return of ``ax.scatter`` /
        ``ax.quiver`` / ``ax.imshow``). If omitted, nothing is clipped and only
        the clip path is returned.
    resolution : str
        Land-polygon resolution (``'110m'`` from :func:`prepare_earth_data`).

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


def _resolve_plate_values(values: Any, fname: str,
                          n_rings: int) -> np.ndarray:
    """Resolve a choropleth ``values`` argument to a per-plate-ring float array.

    ``values`` is either a mapping ``{plate_code_or_name: value}`` — matched
    against the bundled ``plate_codes`` then ``plate_names`` (a MultiPolygon
    plate's several rings all pick up its value); plates absent from the mapping
    become ``NaN`` (unfilled) — or an array with one value per plate ring.
    """
    if isinstance(values, dict):
        d = np.load(_require_data_file(fname), allow_pickle=False)
        if 'plate_codes' not in d:
            raise FileNotFoundError(
                "plot_tectonic_plates(values={...}) needs the plate code/name "
                "metadata, absent from this tectonic_plates.npz. Re-run "
                "prepare_earth_data() to regenerate it.")
        codes = [str(c) for c in d['plate_codes']]
        names = [str(n) for n in d['plate_names']]
        out = np.full(n_rings, np.nan)
        for i in range(n_rings):
            if codes[i] in values:
                out[i] = values[codes[i]]
            elif names[i] in values:
                out[i] = values[names[i]]
        return out
    arr = np.asarray(values, dtype=float)
    if arr.shape[0] != n_rings:
        raise ValueError(
            f"plot_tectonic_plates(values=...): got {arr.shape[0]} values for "
            f"{n_rings} plate rings. Pass a dict keyed by plate code/name "
            f"(recommended), or an array of length {n_rings}.")
    return arr


def plot_tectonic_plates(ax: Any, wcs_mode: bool = True, *,
                         fill: bool = False, cmap: Any = None,
                         facecolor: Any = None, edgecolor: Any = '0.3',
                         alpha: float | None = None, values: Any = None,
                         vmin: float | None = None, vmax: float | None = None,
                         **kwargs: Any) -> Any:
    """
    Plot tectonic plate boundaries — or filled plates — on a globe / planet map.

    By default draws the Bird (2003) plate **boundary arcs** as lines. With
    ``fill=True`` it fills the closed plate **polygons** via the spherical-region
    machinery, in one of three modes:

    * **categorical** (default) — each plate a distinct color from *cmap*
      (default ``'tab20'``): a plate map.
    * **single color** — pass ``facecolor=`` to fill every plate the same (e.g.
      a translucent land tone under the boundary arcs).
    * **choropleth** — pass ``values=`` to color plates by a data value through
      a sequential *cmap* (default ``'viridis'``); returns a ``ScalarMappable``
      for a colorbar.

    Parameters
    ----------
    ax : Axes
        WCSAxes or regular Axes.
    wcs_mode : bool
        If True, uses plot_boundaries_globe (fill always needs ``wcs_mode``).
    fill : bool
        If ``True``, fill the closed plate polygons instead of drawing the
        boundary arcs. Fills on any projection the region machinery supports
        (FITS all-sky, the SIN globe, and the non-FITS custom projections) and
        needs the bundled plate polygons (shipped in ``tectonic_plates.npz``;
        regenerate with :func:`prepare_earth_data` if missing). Default
        ``False``.
    cmap : str or Colormap, optional
        Fill colormap. Default ``'tab20'`` (qualitative) for the categorical
        map, or ``'viridis'`` (sequential) when ``values=`` is given.
    facecolor : color, optional
        With ``fill=True`` and no ``values``, force **all** plates to this one
        color instead of the per-plate cycle.
    edgecolor : color
        Plate outline color for the fill (``fill=True`` only; default
        ``'0.3'``). ``'none'`` for no outline.
    alpha : float, optional
        Fill transparency (``fill=True`` only).
    values : dict or array_like, optional
        Per-plate data for a **choropleth** (``fill=True``). Either a mapping
        ``{plate_code_or_name: value}`` (e.g. ``{'PA': 1.2, 'Africa': 3.4}`` —
        matched against the bundled ``Code`` then ``PlateName``; unlisted plates
        left unfilled) or an array with one value per plate ring. Colors the
        plates through *cmap* + *vmin* / *vmax* and returns a mappable.
    vmin, vmax : float, optional
        Value range for the choropleth color scale (default: data min / max).
    **kwargs
        Line style (``color``, ``lw``, …) for the boundary-arc rendering, or —
        with ``fill=True`` — extra kwargs forwarded to the region fill (e.g.
        ``stroke_color`` / ``stroke_lw`` / ``zorder``).

    Returns
    -------
    artists : list, or ScalarMappable
        Line2D (boundary arcs); fill patches (categorical / single-color fill);
        or a :class:`~matplotlib.cm.ScalarMappable` (choropleth ``values=``),
        ready for ``sph.add_colorbar(ax, mappable=…)``.
    """
    fname = 'tectonic_plates.npz'

    if fill:
        if not wcs_mode:
            raise ValueError(
                "plot_tectonic_plates(fill=True) needs wcs_mode=True: the fill "
                "routes through the spherical-region machinery, which requires "
                "a WCSAxes projection.")
        try:
            plate_data = load_boundary_data(fname, key='plate_polygons')
        except KeyError:
            raise FileNotFoundError(
                "The tectonic_plates.npz on disk has no 'plate_polygons' (only "
                "the boundary arcs). Re-run skyplothelper.globe.prepare_earth_"
                "data() to fetch the closed plate polygons (Bird 2003 / "
                "PB2002_plates), after which fill=True will work.") from None
        rings = _closed_rings(plate_data)

        # The plate polygons carry a long straight edge along the antimeridian
        # (the split seam of MultiPolygon plates like the Pacific). Densify just
        # that edge per ring so it traces the curved frame silhouette on
        # AIT / MOL instead of chording — targeted (only seam edges), so unlike
        # a blanket resolution bump it stays fast on the coarse plate outlines.
        rings = [_densify_seam_runs(lo, la) for (lo, la) in rings]

        if values is not None:
            # Choropleth: resolve per-ring values (dict by code/name, or array)
            # and route through the shared choropleth core.
            from ..geometry.choropleth import choropleth
            per_ring = _resolve_plate_values(values, fname, len(rings))
            return choropleth(
                ax, rings, per_ring,
                cmap='viridis' if cmap is None else cmap,
                vmin=vmin, vmax=vmax, edgecolor=edgecolor, alpha=alpha,
                **kwargs)

        # Categorical: one color per PLATE, so the several rings of a
        # MultiPolygon plate (e.g. the Pacific, split across the antimeridian)
        # share a color rather than reading as different plates. Keyed by the
        # bundled plate codes; fall back to per-ring if metadata is absent.
        cmap_obj = plt.get_cmap('tab20' if cmap is None else cmap)
        try:
            meta = np.load(_require_data_file(fname), allow_pickle=False)
            codes = [str(c) for c in meta['plate_codes']]
        except Exception:
            codes = [str(i) for i in range(len(rings))]
        color_idx: dict[str, int] = {}
        for c in codes:
            color_idx.setdefault(c, len(color_idx))
        patches: list[Any] = []
        for ring, code in zip(rings, codes):
            fc = (facecolor if facecolor is not None
                  else cmap_obj(color_idx[code] % cmap_obj.N))
            one = np.vstack([np.column_stack(ring),
                             [[np.nan, np.nan]]]).astype(float)
            patches.extend(fill_boundaries_globe(
                ax, one, facecolor=fc, edgecolor=edgecolor, alpha=alpha,
                **kwargs))
        return patches

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
            np.savez_compressed(out_path, **arrs)
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
            np.savez_compressed(out_path, **arrs)
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
    # Two products from the same source: the boundary ARCS (PB2002_boundaries,
    # for the line overlay) and the closed plate POLYGONS (PB2002_plates, for
    # the filled overlay `plot_tectonic_plates(fill=True)`). Both are stored in
    # one `tectonic_plates.npz` under the `boundaries` / `plate_polygons` keys.
    if include_tectonic:
        import json
        from urllib.request import urlopen
        base = ("https://raw.githubusercontent.com/fraxen/tectonicplates/"
                "master/GeoJSON/")

        # Boundary arcs (lines).
        try:
            url = base + "PB2002_boundaries.json"
            print(f"  Downloading tectonic plate boundaries from {url}...")
            geojson = json.loads(urlopen(url).read().decode())
        except Exception as e:
            print(f"  Could not download tectonic data: {e}")
            print("  You can manually download PB2002_boundaries.json + "
                  "PB2002_plates.json from:")
            print("  https://github.com/fraxen/tectonicplates/tree/master/GeoJSON")
            print("  and place them in the data/ directory.")
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

        # Closed plate polygons (for the fill). Store each polygon's exterior
        # ring; a ring touching exactly ONE antimeridian edge is nudged inward
        # (as for land) so the region fill's d3 clipper doesn't read a
        # seam-touch as a crossing. Genuine crossers (North America) and
        # pole-enclosers (Antarctica) touch both edges and are left for the
        # clipper. The Pacific plate ships pre-split as a MultiPolygon.
        # Per-RING code / name arrays (a MultiPolygon plate, e.g. the Pacific,
        # contributes several rings that all carry its code/name) so a data
        # value can be aligned to plates by identity, and plates labeled.
        plate_pts, plate_codes, plate_names, n_plates = [], [], [], 0
        try:
            url = base + "PB2002_plates.json"
            print(f"  Downloading tectonic plate polygons from {url}...")
            plates_gj = json.loads(urlopen(url).read().decode())
            for feature in plates_gj['features']:
                geom = feature['geometry']
                props = feature.get('properties', {})
                code = str(props.get('Code', ''))
                name = str(props.get('PlateName', ''))
                polys = ([geom['coordinates']] if geom['type'] == 'Polygon'
                         else geom['coordinates'])
                for poly in polys:
                    n_plates += 1
                    ring = np.asarray(poly[0], dtype=float)[:, :2].copy()
                    lo = ring[:, 0]
                    tp, tn = np.any(lo > 179.99), np.any(lo < -179.99)
                    if tp != tn:
                        lo[lo > 179.99] = 179.9
                        lo[lo < -179.99] = -179.9
                    plate_pts.append(ring)
                    plate_pts.append(np.array([[np.nan, np.nan]]))
                    plate_codes.append(code)
                    plate_names.append(name)
        except Exception as e:
            print(f"  Could not download plate polygons (fill unavailable): {e}")

        save_kw: dict[str, Any] = {'boundaries': tect_data}
        if plate_pts:
            save_kw['plate_polygons'] = np.vstack(plate_pts).astype(np.float32)
            save_kw['plate_codes'] = np.array(plate_codes)
            save_kw['plate_names'] = np.array(plate_names)
        out_path = os.path.join(output_dir, 'tectonic_plates.npz')
        np.savez_compressed(out_path, **save_kw)
        print(f"  Tectonic plates: {len(segments)} boundary segments, "
              f"{n_plates} plate polygons")
        print(f"  Saved: {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")


