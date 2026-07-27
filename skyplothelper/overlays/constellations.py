"""Constellation boundary and label overlays.

Primary data source: ``skyplothelper/data/constellation_corners.npz``
— a compact (~14 KB) corner list from the Davenhall & Leggett 1989
catalog of IAU constellation outlines (Vizier VI/49). 1,565 ordered
corner points across 89 polygons (87 single-polygon constellations
+ Serpens split into Caput and Cauda). Each consecutive corner pair
within a polygon shares either RA or Dec at the B1875 epoch and so
defines a parallel (constant-Dec) or meridian (constant-RA) edge.
The loader walks the corner list and densifies each edge on demand
to the user-requested resolution.

Citation: Davenhall A.C. & Leggett S.K., 1989, "Outlines of the IAU
Constellations" — Vizier VI/49. Underlying canonical data: Roman
(1987) Vizier VI/42, post-Delporte (1930) IAU revision.

Fallbacks (in order):

* legacy densified-segment .npz at
  ``skyplothelper/data/constellation_boundaries.npz`` (kept for
  backward-compat with users who have the older bundle); supplied
  with a warning that v3 corner-list rendering is preferred.
* astropy's bundled ``constellation_data_roman87.dat``. Roman 87
  only tabulates parallel (constant-dec) chords; the meridional
  edges between adjacent chords are not reliably reconstructable
  from the .dat alone, so the fallback emits only the horizontal
  chords. A warning is issued because the rendered overlay is
  incomplete in that mode.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterable
from importlib import resources
from typing import Any

import numpy as np

from .._stroke import _stroke_path_effects

# ---- Constellation Boundaries ----

_CONSTELLATION_CENTERS = {
    'AND': [8.53, 37.43], 'ANT': [152.25, -33.23], 'APS': [243.75, -75.30],
    'AQR': [334.58, -10.78], 'AQL': [295.50, 2.38], 'ARA': [258.00, -55.75],
    'ARI': [40.50, 22.78], 'AUR': [90.00, 42.03], 'BOO': [218.25, 32.37],
    'CAE': [70.50, -39.98], 'CAM': [96.00, 69.40], 'CNC': [130.05, 20.15],
    'CVN': [195.00, 40.10], 'CMA': [104.25, -22.15], 'CMI': [115.50, 6.43],
    'CAP': [315.00, -18.03], 'CAR': [125.25, -63.58], 'CAS': [16.50, 62.18],
    'CEN': [199.50, -47.48], 'CEP': [330.00, 71.53], 'CET': [24.75, -7.18],
    'CHA': [169.50, -79.73], 'CIR': [222.00, -63.03], 'COL': [86.25, -36.23],
    'COM': [190.50, 23.30], 'CRA': [281.25, -40.33], 'CRB': [237.00, 32.63],
    'CRV': [187.50, -18.43], 'CRT': [172.88, -15.18], 'CRU': [186.00, -60.18],
    'CYG': [310.50, 44.55], 'DEL': [308.63, 12.43], 'DOR': [76.50, -61.90],
    'DRA': [245.25, 65.23], 'EQU': [318.75, 7.78], 'ERI': [53.25, -29.75],
    'FOR': [45.00, -30.60], 'GEM': [104.63, 23.38], 'GRU': [340.50, -45.25],
    'HER': [258.75, 30.68], 'HOR': [47.25, -52.58], 'HYA': [156.75, -16.83],
    'HYI': [30.00, -72.68], 'IND': [315.00, -57.03], 'LAC': [337.50, 45.40],
    'LEO': [159.00, 16.75], 'LMI': [155.25, 33.13], 'LEP': [82.50, -19.65],
    'LIB': [229.50, -16.73], 'LUP': [232.50, -43.13], 'LYN': [115.50, 47.48],
    'LYR': [283.50, 37.00], 'MEN': [82.50, -77.50], 'MIC': [315.75, -36.95],
    'MON': [107.25, 0.28], 'MUS': [187.50, -70.15], 'NOR': [243.75, -50.73],
    'OCT': [330.00, -82.15], 'OPH': [259.50, -3.88], 'ORI': [82.50, 4.93],
    'PAV': [289.50, -64.23], 'PEG': [345.00, 19.45], 'PER': [51.00, 45.00],
    'PHE': [18.00, -47.58], 'PIC': [89.25, -53.48], 'PSC': [1.50, 11.08],
    'PSA': [339.75, -30.63], 'PUP': [120.00, -33.98], 'PYX': [132.00, -27.35],
    'RET': [58.50, -62.48], 'SGE': [296.25, 18.88], 'SGR': [283.50, -28.48],
    'SCO': [252.00, -33.38], 'SCL': [13.50, -31.63], 'SCT': [279.00, -10.57],
    'SER': [244.50, 8.03], 'SEX': [153.00, -2.62], 'TAU': [65.25, 18.88],
    'TEL': [271.50, -50.08], 'TRI': [32.25, 31.48], 'TRA': [242.25, -65.40],
    'TUC': [357.00, -65.83], 'UMA': [165.00, 55.38], 'UMI': [225.00, 77.70],
    'VEL': [141.00, -47.17], 'VIR': [198.00, -2.73], 'VOL': [117.00, -69.80],
    'VUL': [301.50, 24.43],
}

_CONSTELLATION_LABEL_OFFSETS_DEG = {
    # Per-constellation (delta_ra, delta_dec) nudges added to the
    # polygon-centroid centers at label-render time. Hand-tuned across
    # two visual-review iterations to lift the abbreviation labels out
    # of neighbouring constellations' visible regions on the default
    # AIT(center=180) all-sky view, where the polygon centroid
    # sometimes lands in adjacent open space rather than the visually-
    # distinct region of its own constellation. In display, +delta_ra
    # is leftward (RA increases right→left on the standard CDELT1<0
    # convention) and +delta_dec is upward.
    # Pass ``apply_default_offsets=False`` to ``add_constellation_labels``
    # to restore the raw centroid positions.
    'AQR': (+10.0, 0.0),   # leftward, onto its lon midpoint
    'ERI': (+3.0, +10.0),  # was overlapping FOR; push well up + slight left
    'MON': (0.0, -3.5),    # slight downward shift
    'PIC': (-9.0, +5.0),   # right + up, off the border edge
    'PSC': (+10.0, 0.0),   # leftward, onto its lon midpoint
    'SER': (-9.0, 0.0),    # rightward, onto its lon midpoint
    'SGE': (0.0, -1.5),    # gentle downward shift (was overshooting)
    'TEL': (+10.0, -2.0),  # leftward, off the ARA border edge
}

_CONSTELLATION_NAMES = {
    'AND': 'Andromeda', 'ANT': 'Antlia', 'APS': 'Apus',
    'AQR': 'Aquarius', 'AQL': 'Aquila', 'ARA': 'Ara',
    'ARI': 'Aries', 'AUR': 'Auriga', 'BOO': 'Boötes',
    'CAE': 'Caelum', 'CAM': 'Camelopardalis', 'CNC': 'Cancer',
    'CVN': 'Canes Venatici', 'CMA': 'Canis Major', 'CMI': 'Canis Minor',
    'CAP': 'Capricornus', 'CAR': 'Carina', 'CAS': 'Cassiopeia',
    'CEN': 'Centaurus', 'CEP': 'Cepheus', 'CET': 'Cetus',
    'CHA': 'Chamaeleon', 'CIR': 'Circinus', 'COL': 'Columba',
    'COM': 'Coma Berenices', 'CRA': 'Corona Australis',
    'CRB': 'Corona Borealis', 'CRV': 'Corvus', 'CRT': 'Crater',
    'CRU': 'Crux', 'CYG': 'Cygnus', 'DEL': 'Delphinus',
    'DOR': 'Dorado', 'DRA': 'Draco', 'EQU': 'Equuleus',
    'ERI': 'Eridanus', 'FOR': 'Fornax', 'GEM': 'Gemini',
    'GRU': 'Grus', 'HER': 'Hercules', 'HOR': 'Horologium',
    'HYA': 'Hydra', 'HYI': 'Hydrus', 'IND': 'Indus',
    'LAC': 'Lacerta', 'LEO': 'Leo', 'LMI': 'Leo Minor',
    'LEP': 'Lepus', 'LIB': 'Libra', 'LUP': 'Lupus',
    'LYN': 'Lynx', 'LYR': 'Lyra', 'MEN': 'Mensa',
    'MIC': 'Microscopium', 'MON': 'Monoceros', 'MUS': 'Musca',
    'NOR': 'Norma', 'OCT': 'Octans', 'OPH': 'Ophiuchus',
    'ORI': 'Orion', 'PAV': 'Pavo', 'PEG': 'Pegasus',
    'PER': 'Perseus', 'PHE': 'Phoenix', 'PIC': 'Pictor',
    'PSC': 'Pisces', 'PSA': 'Piscis Austrinus', 'PUP': 'Puppis',
    'PYX': 'Pyxis', 'RET': 'Reticulum', 'SGE': 'Sagitta',
    'SGR': 'Sagittarius', 'SCO': 'Scorpius', 'SCL': 'Sculptor',
    'SCT': 'Scutum', 'SER': 'Serpens', 'SEX': 'Sextans',
    'TAU': 'Taurus', 'TEL': 'Telescopium', 'TRI': 'Triangulum',
    'TRA': 'Triangulum Australe', 'TUC': 'Tucana',
    'UMA': 'Ursa Major', 'UMI': 'Ursa Minor',
    'VEL': 'Vela', 'VIR': 'Virgo', 'VOL': 'Volans',
    'VUL': 'Vulpecula',
}

_constellation_boundary_cache = None


def _load_constellation_boundaries(
    data_file: str | None = None, step_deg: float = 0.5,
) -> dict[str, Any]:
    """Load constellation boundary segments.

    Source priority:

    1. **User override** via ``data_file=`` — accepts ``.npz``
       (auto-detects corner-list vs legacy segment-list by inspecting
       the array keys), ``.json`` (legacy segment-list), or Roman-87
       ``.dat``.
    2. **Bundled corner-list** at
       ``skyplothelper/data/constellation_corners.npz`` — Vizier VI/49
       (Davenhall & Leggett 1989) corner points at B1875, precessed
       to ICRS for rendering. Each consecutive corner pair forms a
       parallel or meridian edge that is densified on demand to the
       user-requested ``step_deg``.
    3. **Bundled legacy segment-list** at
       ``skyplothelper/data/constellation_boundaries.npz`` — pre-
       densified segments from a grid scan of
       ``astropy.get_constellation``. Kept for backwards-compat.
    4. **astropy Roman 87 fallback** — only parallel chord segments
       (no meridional edges); a `UserWarning` is issued because the
       rendered overlay is then visibly incomplete.

    Parameters
    ----------
    data_file : str, optional
        Override path; auto-detects format from the suffix.
    step_deg : float
        Densification step in degrees. For the corner-list source,
        each parallel/meridian edge is split into pieces no longer
        than ``step_deg`` along its varying coord. For the Roman 87
        fallback, only RA-densification of the chord segments. The
        legacy segment-list .npz is already pre-densified at 0.25°
        and ignores this parameter. Default 0.5°.

    Returns
    -------
    data : dict
        ``{'segments': [[ra1, dec1, ra2, dec2], ...],
          'centers':  _CONSTELLATION_CENTERS,
          'names':    _CONSTELLATION_NAMES,
          'source':   'corners' | 'segments' | 'json' |
                      'astropy_roman87' | 'empty'}``

    Notes
    -----
    The corner-list .npz is generated from Vizier VI/49 (Davenhall &
    Leggett 1989, *Outlines of the IAU Constellations*). The underlying
    boundary definition is the IAU 1930 (Delporte) post-revision,
    surfaced through Roman 1987 / Vizier VI/42.
    """
    global _constellation_boundary_cache
    cache_key = (data_file, step_deg)
    if _constellation_boundary_cache is not None:
        cached_key, cached_val = _constellation_boundary_cache
        if cached_key == cache_key:
            return cached_val

    # --- 1. User override ---
    if data_file is not None:
        if not os.path.exists(data_file):
            warnings.warn(f"Constellation data file not found: {data_file}")
            result = _empty_result()
            _constellation_boundary_cache = (cache_key, result)
            return result
        ext = os.path.splitext(data_file)[1].lower()
        if ext == '.npz':
            result = _load_npz(data_file, step_deg=step_deg)
        elif ext == '.json':
            result = _load_json(data_file)
        else:
            result = _load_roman87(data_file, step_deg=step_deg)
        _constellation_boundary_cache = (cache_key, result)
        return result

    # --- 2. Bundled corner-list (preferred) ---
    try:
        corners_path = str(resources.files('skyplothelper').joinpath(
            'data').joinpath('constellation_corners.npz'))
    except Exception:
        corners_path = None
    if corners_path is not None and os.path.exists(corners_path):
        result = _load_npz(corners_path, step_deg=step_deg)
        _constellation_boundary_cache = (cache_key, result)
        return result

    # --- 3. Bundled legacy segment-list (backward-compat) ---
    try:
        segs_path = str(resources.files('skyplothelper').joinpath(
            'data').joinpath('constellation_boundaries.npz'))
    except Exception:
        segs_path = None
    if segs_path is not None and os.path.exists(segs_path):
        result = _load_npz(segs_path, step_deg=step_deg)
        _constellation_boundary_cache = (cache_key, result)
        return result

    # --- 4. astropy Roman 87 fallback ---
    try:
        import astropy.coordinates as _coord
        roman_path = os.path.join(os.path.dirname(_coord.__file__),
                                   'data',
                                   'constellation_data_roman87.dat')
    except Exception as e:
        warnings.warn(f"Could not locate astropy's constellation data: {e}")
        result = _empty_result()
        _constellation_boundary_cache = (cache_key, result)
        return result
    if not os.path.exists(roman_path):
        warnings.warn(f"astropy constellation data file missing: {roman_path}")
        result = _empty_result()
        _constellation_boundary_cache = (cache_key, result)
        return result
    warnings.warn(
        "Bundled constellation data files (constellation_corners.npz / "
        "constellation_boundaries.npz) are missing — falling back to "
        "astropy's Roman 87 .dat, which contains only parallel chord "
        "segments (no meridional edges). The rendered overlay will look "
        "like horizontal-only ladder rungs. Re-install or fetch the "
        "bundled .npz for canonical IAU boundaries.",
        stacklevel=3,
    )
    result = _load_roman87(roman_path, step_deg=step_deg)
    _constellation_boundary_cache = (cache_key, result)
    return result


def _empty_result() -> dict[str, Any]:
    return {
        'segments': [],
        'centers': _CONSTELLATION_CENTERS,
        'names': _CONSTELLATION_NAMES,
        'source': 'empty',
    }


def _load_npz(path: str, step_deg: float = 0.5) -> dict[str, Any]:
    """Load constellation boundaries from a .npz file.

    Sniffs the contents to handle two formats:

    * **Corner-list** (Vizier VI/49 / Davenhall & Leggett 1989):
      arrays ``cst``, ``ra_icrs``, ``dec_icrs``, ``polygon_id``.
      Edges are densified on demand at the user-requested ``step_deg``.
    * **Legacy segment-list**: an ``(N, 4)`` ``segments`` array of
      pre-densified ``[ra1, dec1, ra2, dec2]`` rows.
    """
    d = np.load(path, allow_pickle=False)
    keys = set(d.files)
    if {'cst', 'ra_icrs', 'dec_icrs', 'polygon_id'} <= keys:
        segments = _densify_corner_list(d, step_deg=step_deg)
        return {
            'segments': segments,
            'centers': _CONSTELLATION_CENTERS,
            'names': _CONSTELLATION_NAMES,
            'source': 'corners',
        }
    if 'segments' in keys:
        segs = np.asarray(d['segments'], dtype=float)
        return {
            'segments': segs.tolist(),
            'centers': _CONSTELLATION_CENTERS,
            'names': _CONSTELLATION_NAMES,
            'source': 'segments',
        }
    raise ValueError(
        f"Unrecognized constellation .npz format at {path!r}: "
        f"keys = {sorted(keys)}. Expected either a corner-list "
        f"(cst, ra_icrs, dec_icrs, polygon_id) or a legacy segment "
        f"list (segments).")


def _densify_corner_list(d: Any, step_deg: float = 0.5) -> list[list[float]]:
    """Walk an ordered corner list and emit boundary segments.

    Each consecutive pair of corners *within the same polygon_id*
    shares either RA or Dec at the B1875 epoch (the IAU convention)
    and so defines a parallel (constant-Dec) or meridian (constant-RA)
    edge. We render in the *ICRS* coords (i.e. precessed to J2000),
    splitting each edge into pieces no longer than ``step_deg`` along
    its varying coordinate.

    Returns
    -------
    segments : list of [ra1, dec1, ra2, dec2]
    """
    cst = np.asarray(d['cst'])
    ra_b = np.asarray(d['ra_b1875'], dtype=float)
    dec_b = np.asarray(d['dec_b1875'], dtype=float)
    ra_i = np.asarray(d['ra_icrs'], dtype=float)
    dec_i = np.asarray(d['dec_icrs'], dtype=float)
    pid = np.asarray(d['polygon_id'], dtype=int)
    n = len(cst)
    if n == 0:
        return []

    # Group rows by polygon_id while preserving traversal order
    polygons: dict[int, list[int]] = {}
    for i in range(n):
        polygons.setdefault(int(pid[i]), []).append(i)

    eps = 5e-3  # B1875 share-tolerance in degrees
    segments: list[list[float]] = []
    for poly_id, idxs in polygons.items():
        m = len(idxs)
        if m < 2:
            continue
        # Walk pairwise (close the ring: last → first)
        for k in range(m):
            i_a = idxs[k]
            i_b = idxs[(k + 1) % m]
            # Determine edge type from B1875 sharing
            share_ra = abs(ra_b[i_a] - ra_b[i_b]) < eps
            share_dec = abs(dec_b[i_a] - dec_b[i_b]) < eps
            if not (share_ra or share_dec):
                # Closing edge of a polygon (last→first) — neither
                # shared. Just emit the chord between the ICRS coords.
                segments.append([float(ra_i[i_a]), float(dec_i[i_a]),
                                 float(ra_i[i_b]), float(dec_i[i_b])])
                continue
            # Densify the edge along its varying coord. We do the
            # subdivision in B1875 space (where the edge is exactly a
            # parallel or meridian) and precess each subdivision point
            # to ICRS via linear interpolation between the endpoints'
            # already-precessed values. The chord error from this
            # short-edge linear interp is sub-arcsecond.
            ra_a, dec_a = ra_b[i_a], dec_b[i_a]
            ra_b_, dec_b_ = ra_b[i_b], dec_b[i_b]
            if share_dec:
                # Parallel edge — dec is shared, RA varies. Account
                # for RA wrap (e.g. 359.0 → 1.0 should go via 360/0).
                d_ra = ra_b_ - ra_a
                if d_ra > 180:
                    d_ra -= 360
                elif d_ra < -180:
                    d_ra += 360
                k_steps = max(1, int(np.ceil(abs(d_ra) / step_deg)))
                t = np.linspace(0.0, 1.0, k_steps + 1)
            else:
                # Meridian edge — RA shared, Dec varies.
                d_dec = dec_b_ - dec_a
                k_steps = max(1, int(np.ceil(abs(d_dec) / step_deg)))
                t = np.linspace(0.0, 1.0, k_steps + 1)
            # ICRS interpolation between already-precessed endpoints.
            # For RA, handle wrap via the same delta logic.
            d_rai = ra_i[i_b] - ra_i[i_a]
            if d_rai > 180:
                d_rai -= 360
            elif d_rai < -180:
                d_rai += 360
            ra_path = ra_i[i_a] + t * d_rai
            ra_path = np.mod(ra_path, 360.0)
            dec_path = dec_i[i_a] + t * (dec_i[i_b] - dec_i[i_a])
            for j in range(len(t) - 1):
                segments.append([float(ra_path[j]), float(dec_path[j]),
                                 float(ra_path[j + 1]),
                                 float(dec_path[j + 1])])
    return segments


def _load_json(path: str) -> dict[str, Any]:
    """Load segments from the legacy JSON format
    ({'segments': [[ra1, dec1, ra2, dec2], ...]}).
    """
    import json
    with open(path) as f:
        data = json.load(f)
    return {
        'segments': data.get('segments', []),
        'centers': data.get('centers', _CONSTELLATION_CENTERS),
        'names': data.get('names', _CONSTELLATION_NAMES),
        'source': 'json',
    }


def _load_roman87(path: str, step_deg: float = 0.5) -> dict[str, Any]:
    """Parse Roman 87 .dat and densify the parallel chords."""
    segments: list[list[float]] = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            try:
                ra_lo_h = float(parts[0])
                ra_hi_h = float(parts[1])
                dec = float(parts[2])
            except (ValueError, IndexError):
                continue
            ra_lo_deg = ra_lo_h * 15.0
            ra_hi_deg = ra_hi_h * 15.0
            span = ra_hi_deg - ra_lo_deg
            if span <= 0:
                continue
            n = max(1, int(np.ceil(span / step_deg)))
            ra_steps = np.linspace(ra_lo_deg, ra_hi_deg, n + 1)
            for j in range(n):
                segments.append([float(ra_steps[j]), dec,
                                 float(ra_steps[j + 1]), dec])
    return {
        'segments': segments,
        'centers': _CONSTELLATION_CENTERS,
        'names': _CONSTELLATION_NAMES,
        'source': 'astropy_roman87',
    }


def add_constellation_boundaries(ax: Any, data_file: str | None = None,
                                  color: Any = '#555555',
                                  lw: float = 0.5, alpha: float = 0.4,
                                  ls: str = '-',
                                  zorder: int = 1,
                                  stroke_color: Any = None,
                                  stroke_lw: float = 2.0,
                                  step_deg: float = 0.5,
                                  **kwargs: Any) -> list[Any]:
    """
    Draw IAU constellation boundary lines on a WCSAxes.

    Boundary segments are parsed on-the-fly from the Roman 87 IAU
    data shipped with astropy
    (``astropy/coordinates/data/constellation_data_roman87.dat``) and
    densified into polyline chords. No separate data file needs to
    ship with skyplothelper — astropy is already a required dependency.
    A custom JSON file can still be supplied via ``data_file=``.

    Parameters
    ----------
    ax : WCSAxes
    data_file : str, optional
        Override the default astropy data source. Accepts either a
        Roman-format ``.dat`` or a pre-processed JSON file in the
        ``{'segments': [[ra1, dec1, ra2, dec2], ...]}`` format.
    color : str
        Line color.
    lw : float
        Line width.
    stroke_color : color spec or None
        Optional stroke color drawn underneath each segment. Default
        ``None`` (no stroke) — boundary lines are intentionally subtle
        with the default ``alpha=0.4``. Set to a contrasting color
        (e.g. ``'k'`` on a bright background, ``'white'`` on a dark
        starfield) to lift the boundaries visually.
    stroke_lw : float
        Total stroke width in points. Default ``2.0``.
    alpha : float
    ls : str
        Line style.
    zorder : int
    step_deg : float
        Densification step (degrees) along each boundary's varying
        coordinate. Smaller traces the boundary more finely on a
        zoomed frame; larger is cheaper on an all-sky map. The loader
        caches per ``(data_file, step_deg)``. Default 0.5.
    **kwargs
        Passed to ``ax.plot()``.

    Returns
    -------
    lines : list of Line2D

    Examples
    --------
    >>> add_constellation_boundaries(ax, color='#666', lw=0.3)
    >>> add_constellation_boundaries(ax, color='cyan', alpha=0.2)
    """
    data = _load_constellation_boundaries(data_file, step_deg=step_deg)
    segments = data.get('segments', [])

    if not segments:
        warnings.warn("No constellation boundary segments loaded.")
        return []

    # Convert the ICRS data into the axes' NATIVE world frame (bulk) and draw
    # through the 'world' transform, so BOTH the placement and the antimeridian
    # split are correct on galactic / ecliptic / other frames — a boundary
    # segment that is seam-safe in RA can straddle the native seam. The split
    # runs in a shifted coordinate where the frame seam (seam_center ± 180) maps
    # to 0/360; each straddling piece is drawn to its own map edge. On an ICRS
    # frame this reduces to the equatorial behavior.
    transform = ax.get_transform('world')
    ra1 = np.array([s[0] for s in segments], dtype=float)
    dec1 = np.array([s[1] for s in segments], dtype=float)
    ra2 = np.array([s[2] for s in segments], dtype=float)
    dec2 = np.array([s[3] for s in segments], dtype=float)
    lon1, lat1, seam_center = _native_frame_coords(ax, ra1, dec1)
    lon2, lat2, _ = _native_frame_coords(ax, ra2, dec2)
    _eps = 1e-6

    stroke_effect = _stroke_path_effects(stroke_color, stroke_lw)
    lines: list[Any] = []

    for k in range(len(lon1)):
        lo1, la1, lo2, la2 = lon1[k], lat1[k], lon2[k], lat2[k]
        rel1 = (lo1 - seam_center + 180.0) % 360.0 - 180.0
        rel2 = (lo2 - seam_center + 180.0) % 360.0 - 180.0
        if abs(rel1 - rel2) > 180.0:
            # Straddles the frame seam — draw one piece to each map edge (a
            # tiny eps keeps each endpoint on its own side of the wrap).
            edge1 = seam_center + np.copysign(180.0 - _eps, rel1)
            edge2 = seam_center + np.copysign(180.0 - _eps, rel2)
            for lons_p, decs_p in (([lo1, edge1], [la1, la2]),
                                   ([edge2, lo2], [la1, la2])):
                ln, = ax.plot(lons_p, decs_p, color=color, lw=lw,
                                alpha=alpha, ls=ls, transform=transform,
                                zorder=zorder, **kwargs)
                if stroke_effect is not None:
                    ln.set_path_effects(stroke_effect)
                lines.append(ln)
            continue
        if abs(rel1 - rel2) > 90.0:
            # Implausibly long single edge — skip (defensive).
            continue
        line, = ax.plot([lo1, lo2], [la1, la2], color=color, lw=lw,
                        alpha=alpha, ls=ls, transform=transform,
                        zorder=zorder, **kwargs)
        if stroke_effect is not None:
            line.set_path_effects(stroke_effect)
        lines.append(line)

    return lines


_constellation_polygons_cache: dict[float, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}


def _load_constellation_polygons(
    step_deg: float = 0.5,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Load per-constellation densified polygon outlines.

    Returns a dict ``{abbr: [(lons, lats), ...]}`` mapping each IAU
    3-letter constellation code to one or more closed-polygon outlines
    in ICRS world coords (deg). Most constellations have a single
    polygon; Serpens has two (Caput + Cauda) and they share the same
    ``'SER'`` key.

    Densification follows the same parallel/meridian walker that
    ``_densify_corner_list`` uses for ``add_constellation_boundaries``:
    each parallel/meridian edge is split into pieces no longer than
    ``step_deg`` along its varying coord, so the polygon outline
    follows the underlying B1875 parallel/meridian curves after
    precession to ICRS.

    Cached per ``step_deg`` value.
    """
    cache_key = float(step_deg)
    if cache_key in _constellation_polygons_cache:
        return _constellation_polygons_cache[cache_key]

    bundled = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'data', 'constellation_corners.npz')
    npz_path = os.path.normpath(bundled)
    d = np.load(npz_path, allow_pickle=False)
    cst = np.asarray(d['cst'])
    ra_b = np.asarray(d['ra_b1875'], dtype=float)
    dec_b = np.asarray(d['dec_b1875'], dtype=float)
    ra_i = np.asarray(d['ra_icrs'], dtype=float)
    dec_i = np.asarray(d['dec_icrs'], dtype=float)
    pid = np.asarray(d['polygon_id'], dtype=int)

    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    eps = 5e-3

    by_pid: dict[int, list[int]] = {}
    for i in range(len(cst)):
        by_pid.setdefault(int(pid[i]), []).append(i)

    for poly_id, idxs in by_pid.items():
        m = len(idxs)
        if m < 3:
            continue
        abbr = str(cst[idxs[0]])
        lons: list[float] = []
        lats: list[float] = []
        for k in range(m):
            i_a = idxs[k]
            i_b = idxs[(k + 1) % m]
            share_ra = abs(ra_b[i_a] - ra_b[i_b]) < eps
            share_dec = abs(dec_b[i_a] - dec_b[i_b]) < eps
            # Determine step count along the varying coord.
            if share_dec and not share_ra:
                d_ra = ra_b[i_b] - ra_b[i_a]
                if d_ra > 180:
                    d_ra -= 360
                elif d_ra < -180:
                    d_ra += 360
                k_steps = max(1, int(np.ceil(abs(d_ra) / step_deg)))
            elif share_ra and not share_dec:
                d_dec = dec_b[i_b] - dec_b[i_a]
                k_steps = max(1, int(np.ceil(abs(d_dec) / step_deg)))
            else:
                # Neither shared — closing chord. Emit a single
                # straight segment.
                k_steps = 1
            t = np.linspace(0.0, 1.0, k_steps + 1)
            # ICRS interpolation between the precessed endpoints,
            # with RA-wrap handling.
            d_rai = ra_i[i_b] - ra_i[i_a]
            if d_rai > 180:
                d_rai -= 360
            elif d_rai < -180:
                d_rai += 360
            ra_path = np.mod(ra_i[i_a] + t * d_rai, 360.0)
            dec_path = dec_i[i_a] + t * (dec_i[i_b] - dec_i[i_a])
            # Drop the last point so the next edge's first point
            # isn't duplicated (it equals this edge's last vertex).
            lons.extend(float(x) for x in ra_path[:-1])
            lats.extend(float(y) for y in dec_path[:-1])
        out.setdefault(abbr, []).append(
            (np.asarray(lons), np.asarray(lats)))

    _constellation_polygons_cache[cache_key] = out
    return out


_constellation_lines_cache: dict[str, Any] | None = None


def _load_constellation_lines(data_file: str | None = None) -> dict[str, Any]:
    """Load constellation asterism line segments.

    Bundled source: ``skyplothelper/data/constellation_lines.npz``
    derived from the d3-celestial ``constellations.lines.json`` file
    (BSD; in turn derived from the IAU Constellation page with minor
    cleanup by Olaf Frohn). Coordinates are equatorial (RA, Dec) in
    degrees at J2000 / ICRS, RA on the [0, 360) branch.

    Parameters
    ----------
    data_file : str, optional
        Override path to a ``.npz`` with the same schema as the
        bundled file (see ``data/fetch_constellation_lines.py``).

    Returns
    -------
    data : dict
        ``{'cst':         (N,) U3,
           'rank':        (N,) int8,
           'ra':          (V,) float32,
           'dec':         (V,) float32,
           'seg_offsets': (S+1,) int32,
           'cst_seg_ids': (S,) int32,
           'source':      str}``
    """
    global _constellation_lines_cache
    if data_file is None and _constellation_lines_cache is not None:
        return _constellation_lines_cache

    # Remember whether we're loading the bundled data *before* resolving the
    # path — only the bundled result is cached (a user-supplied data_file
    # could change between calls).
    use_bundled = data_file is None
    if data_file is None:
        bundled = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'data', 'constellation_lines.npz')
        data_file = os.path.normpath(bundled)

    npz = np.load(data_file, allow_pickle=False)
    data = {
        'cst': npz['cst'],
        'rank': npz['rank'],
        'ra': npz['ra'],
        'dec': npz['dec'],
        'seg_offsets': npz['seg_offsets'],
        'cst_seg_ids': npz['cst_seg_ids'],
        'source': str(npz['source']),
    }
    if use_bundled:
        _constellation_lines_cache = data
    return data


def add_constellation_lines(ax: Any, data_file: str | None = None,
                             constellations: Iterable[str] | None = None,
                             rank_max: int | None = None,
                             color: Any = '#C7A86A', lw: float = 0.5,
                             alpha: float = 0.7,
                             ls: str = '-', zorder: int = 2,
                             stroke_color: Any = None, stroke_lw: float = 2.0,
                             **kwargs: Any) -> list[Any]:
    """
    Draw IAU constellation asterism lines (connect-the-dots star patterns).

    Sources line data from ``skyplothelper/data/constellation_lines.npz``,
    derived from the d3-celestial dataset (BSD-licensed; in turn drawn
    from the IAU's Constellation page with cleanup by Olaf Frohn).
    Coordinates are equatorial (RA, Dec) at J2000 / ICRS.

    Each constellation contributes one or more polyline segments —
    Orion's belt, the Big Dipper's handle, Cassiopeia's W, etc. The
    underlying lines carry a ``rank`` from 1 (most prominent — bright
    canonical stars) to 3 (faintest auxiliary lines); pass
    ``rank_max=`` to draw only the highest-prominence subset.

    Parameters
    ----------
    ax : WCSAxes
    data_file : str, optional
        Override the bundled ``.npz`` (same schema; see
        ``skyplothelper/data/fetch_constellation_lines.py``).
    constellations : iterable of str, optional
        Draw lines only for this subset (3-letter IAU codes,
        case-insensitive). If ``None``, draws all 89 entries
        (88 IAU constellations + Serpens Caput/Cauda split).
    rank_max : int, optional
        Maximum prominence rank to include (1 = brightest only,
        3 = include faint auxiliary lines). ``None`` (default)
        draws all ranks.
    color : str
        Line color. Default warm gold (``#C7A86A``) blends well
        on a dark sky background; pass a brighter color for
        light-themed plots.
    lw : float
    alpha : float
    ls : str
    zorder : int
    stroke_color : color spec or None
        Optional stroke color drawn under each polyline. Default
        ``None`` (no stroke). Useful on busy star-field backgrounds:
        ``stroke_color='k'`` for a dark outline on a bright canvas,
        or ``'white'`` for a soft glow on a dark sky.
    stroke_lw : float
        Total stroke width in points. Default ``2.0``.
    **kwargs
        Forwarded to ``ax.plot()``.

    Returns
    -------
    lines : list of Line2D

    Examples
    --------
    >>> add_constellation_lines(ax)
    >>> add_constellation_lines(ax, rank_max=1, lw=0.8)  # bright only
    >>> add_constellation_lines(ax, constellations=['ORI', 'UMA', 'CAS'])
    """
    data = _load_constellation_lines(data_file)
    cst = data['cst']
    rank = data['rank']
    ra = data['ra']
    dec = data['dec']
    seg_offsets = data['seg_offsets']
    cst_seg_ids = data['cst_seg_ids']

    if constellations is None:
        cst_mask = np.ones(len(cst), dtype=bool)
    else:
        wanted = {c.upper() for c in constellations}
        cst_mask = np.array([c in wanted for c in cst], dtype=bool)

    # Convert the ICRS data into the axes' NATIVE world frame (bulk) and draw
    # through 'world' so both placement AND the antimeridian NaN-break are
    # correct on non-ICRS frames: the break is detected in a shifted coordinate
    # where the frame seam maps to 0/360. See add_constellation_boundaries. On
    # an ICRS frame this reduces to the equatorial behavior.
    transform = ax.get_transform('world')
    all_lon, all_lat, seam_center = _native_frame_coords(ax, ra, dec)
    stroke_effect = _stroke_path_effects(stroke_color, stroke_lw)
    lines: list[Any] = []
    n_segments = len(seg_offsets) - 1
    for s in range(n_segments):
        cst_idx = int(cst_seg_ids[s])
        if not cst_mask[cst_idx]:
            continue
        if rank_max is not None and int(rank[cst_idx]) > int(rank_max):
            continue
        a = int(seg_offsets[s])
        b = int(seg_offsets[s + 1])
        if b - a < 2:
            continue
        seg_lon = all_lon[a:b]
        seg_lat = all_lat[a:b]
        # Break the polyline at frame-seam wraps: shift so the seam maps to
        # 0/360, then a consecutive jump > 180° marks a crossing. Inserting NaN
        # lifts the pen so we don't get a long sweep across the figure.
        shifted = (seg_lon - seam_center + 180.0) % 360.0
        d = np.abs(np.diff(shifted))
        if np.any(d > 180.0):
            lon_with_nan = []
            lat_with_nan = []
            for i in range(len(seg_lon)):
                lon_with_nan.append(float(seg_lon[i]))
                lat_with_nan.append(float(seg_lat[i]))
                if i + 1 < len(seg_lon) and d[i] > 180.0:
                    lon_with_nan.append(np.nan)
                    lat_with_nan.append(np.nan)
            seg_lon = np.asarray(lon_with_nan)
            seg_lat = np.asarray(lat_with_nan)
        line, = ax.plot(seg_lon, seg_lat, color=color, lw=lw,
                          alpha=alpha, ls=ls, transform=transform,
                          zorder=zorder, **kwargs)
        if stroke_effect is not None:
            line.set_path_effects(stroke_effect)
        lines.append(line)
    return lines


def _icrs_to_axes_frame(ax: Any, lons: Any, lats: Any) -> tuple[Any, Any]:
    """Convert ICRS RA/Dec arrays into the axes' native WCS-frame lon/lat.

    The constellation *polygon* path projects through the shapely pipeline
    (:func:`~skyplothelper.geometry.shapes.add_spherical_polygon`), which reads
    its input in the axes' native frame; the *lines* / *boundaries* similarly
    convert + draw through ``'world'`` (see :func:`_native_frame_coords`) so
    their antimeridian split runs against the frame's real seam. Only the
    point-based *labels* stay on the ``'icrs'`` transform. On an equatorial
    (ICRS) frame this is a no-op; on a galactic / ecliptic / supergalactic
    frame it re-expresses the data so the shape lands correctly.
    """
    from ..wcs_frame import _get_wcs_frame_name
    frame_name = _get_wcs_frame_name(ax)
    if frame_name in (None, 'icrs'):
        return lons, lats
    from astropy.coordinates import SkyCoord
    sc = SkyCoord(np.asarray(lons, dtype=float),
                  np.asarray(lats, dtype=float), unit='deg', frame='icrs')
    if frame_name == 'galactic':
        return sc.galactic.l.deg, sc.galactic.b.deg
    if frame_name == 'ecliptic':
        ec = sc.geocentrictrueecliptic
        return ec.lon.deg, ec.lat.deg
    if frame_name == 'supergalactic':
        return sc.supergalactic.sgl.deg, sc.supergalactic.sgb.deg
    return lons, lats


def _native_frame_coords(ax: Any, ra: Any, dec: Any) -> tuple[Any, Any, float]:
    """``(native_lon, native_lat, seam_center)`` for wrap-aware polyline plots.

    Converts ICRS RA/Dec into the axes' native world frame and returns the
    projection-center longitude, so the antimeridian split can run against the
    frame's ACTUAL seam (``seam_center ± 180``) rather than the equatorial one:
    a segment that is seam-safe in RA can straddle the galactic seam (and some
    equatorial splits are unnecessary on a galactic frame). Lines / boundaries
    therefore split in native coords and draw through the ``'world'`` transform.
    On an ICRS frame the coords are returned unchanged.
    """
    from ..wcs_frame import _get_wcs_center_lon
    lon, lat = _icrs_to_axes_frame(ax, ra, dec)
    return (np.asarray(lon, dtype=float), np.asarray(lat, dtype=float),
            float(_get_wcs_center_lon(ax)))


def add_constellation_polygon(ax: Any, constellation: str,
                                step_deg: float = 0.5,
                                facecolor: Any = 'C0', edgecolor: Any = 'C0',
                                alpha: float = 0.25, lw: float = 1.0,
                                stroke_color: Any = None, stroke_lw: float = 2.5,
                                **kwargs: Any) -> list[Any]:
    """
    Fill a single IAU constellation as a closed polygon overlay.

    Resolves ``constellation`` (3-letter IAU code, case-insensitive)
    against the bundled corner-list and hands each polygon outline to
    :func:`skyplothelper.geometry.shapes.add_spherical_polygon` for
    rendering — so the overlay inherits the standard antimeridian /
    pole / frame-edge handling and works on AIT / MOL / SIN-globe /
    TAN frames alike. Serpens has two polygons (Caput + Cauda) and
    both are drawn.

    Parameters
    ----------
    ax : WCSAxes
    constellation : str
        IAU 3-letter abbreviation (e.g. ``'UMi'``, ``'ORI'``).
        Case-insensitive.
    step_deg : float
        Edge densification step in degrees along each
        parallel/meridian B1875 edge. Matches the default used by
        ``add_constellation_boundaries``. Default 0.5°.
    facecolor : matplotlib color
        Polygon fill. Default ``'C0'``; combine with ``alpha`` for a
        translucent highlight.
    edgecolor : matplotlib color
        Polygon outline. Default ``'C0'``.
    alpha : float
        Fill alpha. Default 0.25.
    lw : float
        Outline linewidth. Default 1.0.
    stroke_color : color spec or None
        Optional stroke color drawn under the polygon outline (via
        :class:`matplotlib.patheffects.withStroke`). Default ``None``
        (no stroke). Useful when highlighting a constellation against
        a busy background.
    stroke_lw : float
        Total stroke width in points. Default ``2.5``.
    **kwargs
        Forwarded to
        :func:`~skyplothelper.geometry.shapes.add_spherical_polygon`.
        Useful ones: ``clip='auto'/'d3'/'project_shape'``,
        ``geodesic='auto'/True/False``, ``zorder=``,
        ``hatch=``, ``label=``.

    Returns
    -------
    patches : list of matplotlib.patches.Patch
        Patch artists added to ``ax``; usually one entry, two for
        Serpens (Caput + Cauda).

    Raises
    ------
    KeyError
        If ``constellation`` doesn't match any IAU abbreviation in
        the bundled corner list.

    Examples
    --------
    Highlight Ursa Minor::

        add_constellation_polygon(ax, 'UMi', facecolor='lightblue',
                                  alpha=0.3, edgecolor='steelblue')

    Outline-only (no fill) for Cygnus::

        add_constellation_polygon(ax, 'Cyg', facecolor='none',
                                  edgecolor='gold', lw=1.5)

    Highlight both halves of Serpens (single call, two patches)::

        add_constellation_polygon(ax, 'ser', facecolor='salmon',
                                  alpha=0.35)
    """
    from ..geometry.shapes import add_spherical_polygon

    polygons = _load_constellation_polygons(step_deg=step_deg)
    key = str(constellation).upper()
    if key not in polygons:
        raise KeyError(
            f"add_constellation_polygon: unknown IAU code {constellation!r}. "
            f"Use list_constellations() to see all 88 codes.")

    stroke_effect = _stroke_path_effects(stroke_color, stroke_lw)
    patches: list[Any] = []
    for lons, lats in polygons[key]:
        # add_spherical_polygon projects in the axes' native frame; the corner
        # data is ICRS, so re-express it for galactic / ecliptic / ... frames.
        lons, lats = _icrs_to_axes_frame(ax, lons, lats)
        out = add_spherical_polygon(
            ax, lons, lats, resolution=1,
            facecolor=facecolor, edgecolor=edgecolor,
            alpha=alpha, lw=lw, **kwargs)
        # add_spherical_polygon returns a list of patches (one per
        # frame-clipped sub-polygon). Flatten.
        if isinstance(out, list):
            patches.extend(out)
        elif out is not None:
            patches.append(out)
    if stroke_effect is not None:
        for p in patches:
            p.set_path_effects(stroke_effect)
    return patches


def add_constellation_labels(ax: Any, labels: str = 'abbr',
                              color: Any = '#888888',
                              fontsize: float = 7, alpha: float = 0.6,
                              zorder: int = 2,
                              constellations: Iterable[str] | None = None,
                              apply_default_offsets: bool = True,
                              stroke_color: Any = None, stroke_lw: float = 2.0,
                              **kwargs: Any) -> list[Any]:
    """
    Add constellation name labels at approximate center positions.

    Works independently of boundary data — the 88 constellation
    centers are embedded in the module.

    Parameters
    ----------
    ax : WCSAxes
    labels : str
        'abbr' (default) — 3-letter IAU abbreviation (e.g. 'ORI').
        'name' — full name (e.g. 'Orion').
        'both' — abbreviation + name.
    color : str
    fontsize : float
    alpha : float
    zorder : int
    constellations : list of str, optional
        Only label these constellations (IAU abbreviations).
        If None, labels all 88.
    apply_default_offsets : bool
        When ``True`` (default), nudge a handful of labels off their
        raw polygon-centroid positions (AQR, ERI, MON, PIC, PSC, SER,
        SGE, TEL) so they sit inside their own visually-distinct regions on the
        default AIT(center=180) all-sky view. Set ``False`` to restore
        the unmodified centroid positions (useful for projection
        centers other than 180° where the offsets may not apply).
    stroke_color : color spec or None
        Optional stroke color drawn under each label. Default
        ``None`` (no stroke) — the default gray text at
        ``alpha=0.6`` is intentionally subtle. Enable on busy
        backgrounds: ``stroke_color='k'`` for dark stroke on
        bright canvases, ``'white'`` on a starfield.
    stroke_lw : float
        Total stroke width in points. Default ``2.0``.
    **kwargs
        Passed to ``ax.text()``.

    Returns
    -------
    texts : list of Text

    Examples
    --------
    >>> add_constellation_labels(ax, labels='abbr', fontsize=6)
    >>> add_constellation_labels(ax, labels='name',
    ...     constellations=['ORI', 'CYG', 'SCO', 'CRU', 'UMA'])
    """
    # Constellation data is ICRS RA/Dec. Draw it through the 'icrs' transform,
    # not 'world' (the axes' native frame) — otherwise the RA/Dec are read as
    # native lon/lat and land in the wrong place on a galactic / ecliptic /
    # other-equinox frame. On an ICRS frame 'icrs' == 'world', so equatorial
    # behavior is unchanged.
    transform = ax.get_transform('icrs')
    stroke_effect = _stroke_path_effects(stroke_color, stroke_lw)
    texts: list[Any] = []

    keys = ([c.upper() for c in constellations] if constellations
            else sorted(_CONSTELLATION_CENTERS.keys()))

    for abbr in keys:
        if abbr not in _CONSTELLATION_CENTERS:
            continue
        ra, dec = _CONSTELLATION_CENTERS[abbr]
        if apply_default_offsets:
            dra, ddec = _CONSTELLATION_LABEL_OFFSETS_DEG.get(
                abbr, (0.0, 0.0))
            ra = (ra + dra) % 360.0
            dec = dec + ddec

        # Skip labels whose center is unprojectable — a globe's far-side
        # positions transform to NaN, and an ax.text there spams "posx and
        # posy should be finite values" on every draw. Lines / boundaries
        # already self-cull this way.
        try:
            px, py = transform.transform((ra, dec))
        except Exception:
            px = py = np.nan
        if not (np.isfinite(px) and np.isfinite(py)):
            continue

        if labels == 'name':
            text = _CONSTELLATION_NAMES.get(abbr, abbr)
        elif labels == 'both':
            text = f"{abbr} {_CONSTELLATION_NAMES.get(abbr, '')}"
        else:
            text = abbr

        text_kw = dict(ha='center', va='center', zorder=zorder, clip_on=True)
        text_kw.update(kwargs)
        txt = ax.text(ra, dec, text, transform=transform,
                      fontsize=fontsize, color=color, alpha=alpha, **text_kw)
        if stroke_effect is not None:
            txt.set_path_effects(stroke_effect)
        texts.append(txt)

    return texts


def list_constellations(sort: str = 'abbr') -> None:
    """
    Print the 88 IAU constellations with abbreviations and centers.

    Parameters
    ----------
    sort : str
        Sort by 'abbr' (default), 'name', 'ra', or 'dec'.
    """
    items = [(a, _CONSTELLATION_NAMES.get(a, ''),
              _CONSTELLATION_CENTERS[a][0], _CONSTELLATION_CENTERS[a][1])
             for a in _CONSTELLATION_CENTERS]

    sort_keys = {'name': lambda x: x[1], 'ra': lambda x: x[2],
                 'dec': lambda x: x[3]}
    items.sort(key=sort_keys.get(sort, lambda x: x[0]))

    print(f"{'Abbr':<5s} {'Name':<24s} {'RA (°)':>8s}  {'Dec (°)':>8s}")
    print("-" * 52)
    for abbr, name, ra, dec in items:
        print(f"{abbr:<5s} {name:<24s} {ra:>8.2f}  {dec:>+8.2f}")

