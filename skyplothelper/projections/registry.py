"""Projection registry, alias resolver, and lister.

Single source of truth for the mapping between human-readable projection
names, FITS codes, and frame classes.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any

from astropy.visualization.wcsaxes.frame import EllipticalFrame

from .frames import (
    CircularFrame,
    Eckert4Frame,
    KavrayskiyFrame,
    McBrydeFrame,
    ParabolicFrame,
    RobinsonFrame,
    SinusoidalFrame,
    WinkelTripelFrame,
)

_ProjInfo = namedtuple('_ProjInfo', [
    'fits_code', 'frame_shape', 'proj_class', 'allsky', 'description'
])

_PROJECTION_REGISTRY = {
    # --- Pseudocylindrical (all-sky) ---
    'ait': _ProjInfo('AIT', 'elliptical', 'pseudocylindrical', True,
                     'Hammer-Aitoff equal-area'),
    'mol': _ProjInfo('MOL', 'elliptical', 'pseudocylindrical', True,
                     'Mollweide equal-area'),
    'par': _ProjInfo('PAR', 'parabolic', 'pseudocylindrical', True,
                     'Parabolic'),
    'sfl': _ProjInfo('SFL', 'sinusoidal', 'pseudocylindrical', True,
                     'Sanson-Flamsteed (sinusoidal)'),

    # --- Cylindrical ---
    'car': _ProjInfo('CAR', 'rectangular', 'cylindrical', True,
                     'Plate Carrée (equirectangular)'),
    'mer': _ProjInfo('MER', 'rectangular', 'cylindrical', True,
                     'Mercator conformal'),
    'cea': _ProjInfo('CEA', 'rectangular', 'cylindrical', True,
                     'Cylindrical equal-area'),
    'cyp': _ProjInfo('CYP', 'rectangular', 'cylindrical', True,
                     'Cylindrical perspective'),

    # --- Zenithal (field or globe) ---
    'tan': _ProjInfo('TAN', 'rectangular', 'zenithal', False,
                     'Gnomonic (tangent plane)'),
    'sin': _ProjInfo('SIN', 'circular', 'zenithal', False,
                     'Slant orthographic'),
    'arc': _ProjInfo('ARC', 'circular', 'zenithal', False,
                     'Zenithal equidistant'),
    'zea': _ProjInfo('ZEA', 'circular', 'zenithal', False,
                     'Lambert azimuthal equal-area'),
    'stg': _ProjInfo('STG', 'circular', 'zenithal', False,
                     'Stereographic conformal'),
    'azp': _ProjInfo('AZP', 'circular', 'zenithal', False,
                     'Zenithal perspective'),
    'szp': _ProjInfo('SZP', 'circular', 'zenithal', False,
                     'Slant zenithal perspective'),
    'air': _ProjInfo('AIR', 'circular', 'zenithal', False,
                     'Airy'),

    # --- Conic ---
    # allsky=True: the bare default centers on the standard parallel
    # (CRVAL2=PV2_1) and frames the wedge (kapteyn recipe; see
    # make_wcs_frame's all-sky conic branch). A zoomed field view is
    # still available by passing fov_deg / cdelt (the field-override path).
    'cop': _ProjInfo('COP', 'rectangular', 'conic', True,
                     'Conic perspective'),
    'coe': _ProjInfo('COE', 'rectangular', 'conic', True,
                     'Conic equal-area'),
    'cod': _ProjInfo('COD', 'rectangular', 'conic', True,
                     'Conic equidistant'),
    'coo': _ProjInfo('COO', 'rectangular', 'conic', True,
                     'Conic orthomorphic'),

    # --- Pseudoconic ---
    # BON (Bonne's equal-area) renders into a standard
    # ``RectangularFrame`` (kapteyn-style) — the projection lives
    # as a heart-cardioid inside the rectangle. Per-projection
    # extent dispatch in ``make_wcs_frame`` sizes the rectangle
    # to ±(90+lat_1)° in y and ±140° in x so the cardioid fits
    # cleanly.
    'bon': _ProjInfo('BON', 'rectangular', 'pseudoconic', True,
                     'Bonne equal-area'),
    'pco': _ProjInfo('PCO', 'rectangular', 'polyconic', True,
                     'Polyconic'),

    # --- HEALPix ---
    # The HPX projection's visible region is a stepped diamond (4 polar
    # pixels above + equatorial band + 4 polar pixels below) which
    # fits a rectangle far more tightly than an ellipse. The
    # rectangular frame leaves less empty space and lets the
    # stepped-diamond outline drawn by the gallery helper read clearly.
    'hpx': _ProjInfo('HPX', 'rectangular', 'healpix', True,
                     'HEALPix'),
    'xph': _ProjInfo('XPH', 'rectangular', 'healpix', True,
                     'HEALPix polar (butterfly)'),

    # --- Quadcube ---
    'tsc': _ProjInfo('TSC', 'rectangular', 'quadcube', True,
                     'Tangential spherical cube'),
    'csc': _ProjInfo('CSC', 'rectangular', 'quadcube', True,
                     'COBE spherical cube'),
    'qsc': _ProjInfo('QSC', 'rectangular', 'quadcube', True,
                     'Quadrilateralized spherical cube'),

    # --- Non-FITS (planned, not yet implemented) ---
    # These will need custom CurvedTransform subclasses
    'robinson':     _ProjInfo(None, 'robinson', 'pseudocylindrical', True,
                              'Robinson compromise [non-FITS]'),
    'mcbryde':      _ProjInfo(None, 'mcbryde', 'pseudocylindrical', True,
                              'McBryde-Thomas flat-polar quartic [non-FITS]'),
    'eckert_iv':    _ProjInfo(None, 'eckert4', 'pseudocylindrical', True,
                              'Eckert IV equal-area [non-FITS]'),
    'kavrayskiy':   _ProjInfo(None, 'kavrayskiy', 'pseudocylindrical', True,
                              'Kavrayskiy VII compromise [non-FITS]'),
    'winkel_tripel':_ProjInfo(None, 'winkel_tripel', 'pseudocylindrical', True,
                              'Winkel Tripel compromise [non-FITS]'),
}

# Human-readable aliases → canonical registry key
_PROJECTION_ALIASES = {
    # Hammer-Aitoff
    'aitoff': 'ait', 'hammer': 'ait', 'hammer_aitoff': 'ait', 'hammer-aitoff': 'ait',
    'hammeraitoff': 'ait',
    # Mollweide
    'mollweide': 'mol',
    # Parabolic
    'parabolic': 'par',
    # Sanson-Flamsteed
    'sinusoidal': 'sfl', 'sanson': 'sfl', 'sanson_flamsteed': 'sfl',
    'sanson-flamsteed': 'sfl', 'sansonflamsteed': 'sfl', 'global_sinusoid': 'sfl',
    # Plate Carrée
    'platecarree': 'car', 'plate_carree': 'car', 'plate-carree': 'car',
    'equirectangular': 'car', 'cartesian': 'car',
    # Mercator
    'mercator': 'mer',
    # Cylindrical equal-area
    'cylindrical_equal_area': 'cea', 'cylindricalequalarea': 'cea',
    # Cylindrical perspective
    'cylindrical_perspective': 'cyp', 'cylindricalperspective': 'cyp',
    # Gnomonic
    'gnomonic': 'tan', 'tangent': 'tan', 'tangent_plane': 'tan',
    # Orthographic
    'orthographic': 'sin', 'ortho': 'sin',
    # Zenithal equidistant
    'zenithal_equidistant': 'arc', 'zenithequidistant': 'arc',
    # Lambert azimuthal equal-area
    'lambert': 'zea', 'lambert_azimuthal': 'zea', 'lambertazimuthal': 'zea',
    # Stereographic
    'stereographic': 'stg', 'stereo': 'stg',
    # Zenithal perspective
    'zenithal_perspective': 'azp', 'zenithalperspective': 'azp',
    # Airy
    'airy': 'air',
    # Conics
    'conic_perspective': 'cop', 'conic_equal_area': 'coe',
    'conic_equidistant': 'cod', 'conic_orthomorphic': 'coo',
    # Bonne (renders in a kapteyn-style rectangular frame)
    'bonne': 'bon', 'bonne_equal_area': 'bon',
    # Polyconic
    'polyconic': 'pco',
    # HEALPix
    'healpix': 'hpx', 'healpix_polar': 'xph', 'butterfly': 'xph',
    # Quadcube
    'tangential_spherical_cube': 'tsc', 'cobe_spherical_cube': 'csc',
    'quadrilateralized_spherical_cube': 'qsc',
    # Non-FITS (planned)
    'eckert': 'eckert_iv', 'eckert4': 'eckert_iv', 'eckert_4': 'eckert_iv',
    'kavrayskiy_vii': 'kavrayskiy', 'kavrayskiy7': 'kavrayskiy',
    'winkel': 'winkel_tripel', 'winkeltripel': 'winkel_tripel',
    'mcbryde_thomas': 'mcbryde',
}


def _resolve_projection(name: str) -> tuple[str, Any]:
    """
    Resolve a projection name (FITS code, human-readable, or alias) to
    its canonical registry key and ProjInfo.

    Parameters
    ----------
    name : str
        Projection name in any accepted form

    Returns
    -------
    key : str
        Canonical lowercase registry key
    info : _ProjInfo
        Projection metadata

    Raises
    ------
    ValueError
        If the projection name is not recognized
    """
    # Normalize: lowercase, strip, replace hyphens/spaces with underscores
    norm = name.lower().strip().replace('-', '_').replace(' ', '_')

    # Direct match in registry
    if norm in _PROJECTION_REGISTRY:
        return norm, _PROJECTION_REGISTRY[norm]

    # Check aliases
    if norm in _PROJECTION_ALIASES:
        key = _PROJECTION_ALIASES[norm]
        return key, _PROJECTION_REGISTRY[key]

    # Try without underscores (e.g., 'platecarree' or 'hammeraitoff')
    norm_nouscore = norm.replace('_', '')
    if norm_nouscore in _PROJECTION_ALIASES:
        key = _PROJECTION_ALIASES[norm_nouscore]
        return key, _PROJECTION_REGISTRY[key]

    # Build helpful error message
    raise ValueError(
        f"Unknown projection '{name}'. Use list_projections() to see available "
        f"projections, or try one of: {', '.join(sorted(_PROJECTION_REGISTRY.keys()))}"
    )


def list_projections(
        shape: str | None = None,
        allsky: bool | None = None,
        fits_only: bool = False,
        proj_class: str | None = None,
        as_table: bool = False) -> list[dict[str, Any]] | None:
    """
    List all available projections with descriptions.

    Parameters
    ----------
    shape : str, optional
        Filter by frame shape, e.g. 'elliptical', 'rectangular',
        'circular', 'sinusoidal', 'parabolic' (matched against each
        projection's ``frame_shape``).
    allsky : bool, optional
        If True, show only all-sky projections; if False, only non-all-sky
    fits_only : bool
        If True, exclude non-FITS projections
    proj_class : str, optional
        Filter by projection class: 'pseudocylindrical', 'cylindrical',
        'zenithal', 'conic', etc.
    as_table : bool
        If True, return as list of dicts instead of printing

    Returns
    -------
    None (prints table) or list of dicts (if as_table=True)
    """
    # Collect entries
    entries = []
    for key, info in sorted(_PROJECTION_REGISTRY.items()):
        if shape is not None and info.frame_shape != shape.lower():
            continue
        if allsky is not None and info.allsky != allsky:
            continue
        if fits_only and info.fits_code is None:
            continue
        if proj_class is not None and info.proj_class != proj_class.lower():
            continue

        # Find aliases for this key
        aliases = sorted([a for a, k in _PROJECTION_ALIASES.items() if k == key])
        alias_str = ', '.join(aliases[:4])  # limit to 4 for display
        if len(aliases) > 4:
            alias_str += ', ...'

        entry = {
            'code': info.fits_code or '-',
            'key': key,
            'aliases': alias_str,
            'shape': info.frame_shape,
            'class': info.proj_class,
            'allsky': 'yes' if info.allsky else 'no',
            'description': info.description,
        }
        entries.append(entry)

    if as_table:
        return entries

    # Print formatted table
    if not entries:
        print("No projections match the specified filters.")
        return None

    # Column widths
    w_code = max(4, max(len(e['code']) for e in entries))
    w_alias = max(7, min(40, max(len(e['aliases']) for e in entries)))
    w_shape = max(5, max(len(e['shape']) for e in entries))
    w_class = max(5, max(len(e['class']) for e in entries))

    header = (f"{'Code':<{w_code}}  {'Aliases':<{w_alias}}  "
              f"{'Shape':<{w_shape}}  {'Class':<{w_class}}  "
              f"{'Sky':>3}  Description")
    print(header)
    print('-' * len(header))
    for e in entries:
        print(f"{e['code']:<{w_code}}  {e['aliases']:<{w_alias}}  "
              f"{e['shape']:<{w_shape}}  {e['class']:<{w_class}}  "
              f"{e['allsky']:>3}  {e['description']}")
    return None


# Frame shape → frame class mapping
_FRAME_CLASSES = {
    'elliptical': EllipticalFrame,
    'sinusoidal': SinusoidalFrame,
    'circular': CircularFrame,
    'parabolic': ParabolicFrame,
    'robinson': RobinsonFrame,
    'kavrayskiy': KavrayskiyFrame,
    'eckert4': Eckert4Frame,
    'winkel_tripel': WinkelTripelFrame,
    'mcbryde': McBrydeFrame,
    'rectangular': None,  # use default (RectangularFrame or no frame_class)
}


def get_frame_class(projection: Any) -> Any:
    """
    Return the appropriate WCSAxes frame class for a projection.

    Astropy's ``WCSAxes`` defaults to a rectangular frame regardless
    of the WCS projection.  This function looks up the correct frame
    class (e.g., ``EllipticalFrame`` for Aitoff/Mollweide) so you can
    pass it to ``fig.add_subplot()``.

    Parameters
    ----------
    projection : str or WCS
        A projection name/code (e.g., ``'AIT'``, ``'mollweide'``),
        or an ``astropy.wcs.WCS`` object whose CTYPE will be inspected.

    Returns
    -------
    frame_class : class or None
        The frame class to pass as ``frame_class=`` when creating a
        WCSAxes subplot.  Returns ``None`` for rectangular projections
        (the astropy default is already correct).

    Examples
    --------
    >>> wcs = make_allsky_wcs('AIT', 180)
    >>> fc = sph.get_frame_class(wcs)
    >>> ax = fig.add_subplot(111, projection=wcs, frame_class=fc)

    >>> # Or with a projection name directly:
    >>> fc = sph.get_frame_class('mollweide')
    """
    if isinstance(projection, str):
        proj_code = projection
    elif hasattr(projection, 'wcs') and hasattr(projection.wcs, 'ctype'):
        ctype1 = projection.wcs.ctype[0]
        proj_code = ctype1.split('-')[-1].strip()
    else:
        return None

    try:
        _, proj_info = _resolve_projection(proj_code)
        return _FRAME_CLASSES.get(proj_info.frame_shape)
    except (ValueError, KeyError):
        return None
