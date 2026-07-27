"""Survey footprint overlays.

The ``SURVEY_FOOTPRINTS`` dict catalogs major imaging / spectroscopic
survey areas as polygon or rectangle bounding boxes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from astropy.coordinates import SkyCoord
from matplotlib.colors import to_rgba

from ..geometry.bands import add_latitude_band, add_lonlat_box
from ..geometry.shapes import add_spherical_polygon
from ..wcs_frame import _get_wcs_frame_name


def _apply_fill_edge_style(kw: dict[str, Any], *, fill: bool, fc: Any,
                           ec: Any, alpha: float,
                           edge_alpha: float | None) -> None:
    """Populate facecolor / edgecolor (and scalar alpha) on a footprint's
    render-kwargs dict, in place.

    ``edge_alpha=None`` (default) keeps the legacy behavior — a scalar
    ``alpha`` on the whole patch, so the edge inherits the fill alpha —
    which leaves rendered output unchanged. An explicit ``edge_alpha``
    bakes both alphas into RGBA face/edge colors (no scalar alpha) so the
    edge gets its own transparency.
    """
    if edge_alpha is None:
        kw['facecolor'] = fc if fill else 'none'
        if fill:
            kw['alpha'] = alpha
        kw['edgecolor'] = ec
    else:
        kw['facecolor'] = to_rgba(fc, alpha) if fill else 'none'
        kw['edgecolor'] = to_rgba(ec, edge_alpha)


# ---- Survey Footprints ----

# Survey footprint definitions: dict of survey name → list of
# (lon_min, lon_max, lat_min, lat_max) rectangles in ICRS degrees.
# For surveys with complex boundaries, these are simplified bounding boxes.
# More precise boundaries can be added as polygon vertices.

SURVEY_FOOTPRINTS: dict[str, dict[str, Any]] = {
    # ----------------------------------------------------------------
    # Wide-field optical/IR imaging surveys
    # ----------------------------------------------------------------
    'sdss': {
        'name': 'SDSS',
        'color': 'C0',
        'description': 'Sloan Digital Sky Survey (Legacy + BOSS)',
        'frame': 'icrs',
        'regions': [
            # Bounding boxes (used when boundary_style='box')
            {'type': 'rect', 'lon': (100, 260), 'lat': (-5, 70)},
            {'type': 'rect', 'lon': (310, 60), 'lat': (-1.25, 1.25)},
            # NGC polygon: tapered shape from great-circle stripe geometry.
            # Vertices traced from DR7/DR16 footprint maps; the boundary
            # narrows at high Dec where stripes converge, and has slight
            # irregularity near the galactic plane crossing (~RA 200).
            {'type': 'polygon', 'vertices': [
                [100, -3], [120, -5], [145, -4], [175, -3],
                [210, -2], [240, -3], [255, 0], [260, 10],
                [263, 25], [262, 40], [257, 55], [248, 64],
                [235, 68], [210, 70], [180, 70], [150, 68],
                [130, 64], [118, 55], [112, 40], [108, 25],
                [105, 10],
            ]},
            # Stripe 82 (SGC): narrow equatorial strip, ~270 sq deg
            # RA -50° to +59° (= 310° to 59°), Dec ±1.25°
            {'type': 'polygon', 'vertices': [
                [310, -1.25], [330, -1.25], [350, -1.25],
                [10, -1.25], [30, -1.25], [59, -1.25],
                [59, 1.25], [30, 1.25], [10, 1.25],
                [350, 1.25], [330, 1.25], [310, 1.25],
            ]},
        ],
    },
    'stripe82': {
        'name': 'Stripe 82',
        'color': 'deepskyblue',
        'description': ('SDSS Stripe 82 — narrow deep equatorial '
                        'field heavily reused by follow-up surveys '
                        '(HSC, DES, VLA-Stripe82, etc.)'),
        'frame': 'icrs',
        'regions': [
            # RA −50° to +59° (i.e. 310° → 59° wrapping the
            # antimeridian), Dec ±1.25°. ``add_survey_footprint``
            # detects ``lon_max < lon_min`` and splits into two
            # non-wrapping bands automatically.
            {'type': 'rect', 'lon': (310, 59), 'lat': (-1.25, 1.25)},
        ],
    },
    'des': {
        'name': 'DES',
        'color': 'C1',
        'description': 'Dark Energy Survey (~5000 sq deg)',
        'frame': 'icrs',
        'regions': [
            # Bounding box
            {'type': 'rect', 'lon': (315, 105), 'lat': (-65, 5)},
            # Improved polygon: DES covers southern sky from
            # RA ~310° to ~105°, Dec ~-65° to ~+5°, overlapping SPT
            # and Stripe 82. Shape avoids galactic plane on western
            # edge. LMC/SMC are within the footprint (not excised).
            # Traced from DES DR1/DR2 footprint maps.
            {'type': 'polygon', 'vertices': [
                # Stripe 82 / equatorial strip (north edge)
                [310, -1], [330, -1], [350, -1],
                [10, -1], [30, -1], [50, 1], [70, 3], [90, 3],
                # Eastern boundary curving south
                [100, 0], [104, -5], [105, -15], [104, -30],
                [100, -45], [94, -55],
                # Southern boundary
                [85, -62], [70, -65], [50, -65], [30, -65],
                [10, -64], [350, -62],
                # Western boundary — curves north, avoids MW
                [335, -55], [325, -45], [318, -35],
                [314, -25], [312, -15], [310, -8],
            ]},
        ],
    },
    'hsc': {
        'name': 'HSC-SSP',
        'color': 'C2',
        'description': 'Hyper Suprime-Cam Subaru Strategic Program (Wide)',
        'frame': 'icrs',
        'regions': [
            # HSC Wide layer: 6 disjoint fields, ~1400 sq deg total.
            # Each field is roughly rectangular.
            # XMM-LSS
            {'type': 'rect', 'lon': (30, 40), 'lat': (-7, -1)},
            # GAMA09H
            {'type': 'rect', 'lon': (129, 141), 'lat': (-2, 3)},
            # Wide12H / GAMA12
            {'type': 'rect', 'lon': (170, 190), 'lat': (-2, 3)},
            # GAMA15H
            {'type': 'rect', 'lon': (210, 225), 'lat': (-2, 3)},
            # HECTOMAP
            {'type': 'rect', 'lon': (210, 250), 'lat': (42, 45)},
            # VVDS
            {'type': 'rect', 'lon': (330, 345), 'lat': (-1, 5)},
        ],
    },
    'panstarrs': {
        'name': 'Pan-STARRS',
        'color': 'C3',
        'description': 'Pan-STARRS1 3π Survey',
        'frame': 'icrs',
        'regions': [
            {'type': 'rect', 'lon': (0, 360), 'lat': (-30, 90)},
        ],
    },
    'decals': {
        'name': 'DECaLS',
        'color': 'C4',
        'description': 'Dark Energy Camera Legacy Survey',
        'frame': 'icrs',
        'regions': [
            {'type': 'rect', 'lon': (0, 360), 'lat': (-20, 34)},
        ],
    },
    'kids': {
        'name': 'KiDS',
        'color': 'deepskyblue',
        'description': 'Kilo-Degree Survey (VST)',
        'frame': 'icrs',
        'regions': [
            # KiDS-North (overlaps GAMA fields)
            {'type': 'rect', 'lon': (128, 238), 'lat': (-5, 4)},
            # KiDS-South (overlaps VIKING, SGC)
            {'type': 'rect', 'lon': (329, 53), 'lat': (-36, -25)},
        ],
    },
    'cfhtls': {
        'name': 'CFHTLS Wide',
        'color': 'mediumvioletred',
        'description': ('Canada-France-Hawaii Telescope Legacy Survey '
                        '— Wide (4 patches, ~155 sq deg total)'),
        'frame': 'icrs',
        'regions': [
            # W1: 7° × 9° centered (RA 02h18m, Dec -07°)
            {'type': 'rect', 'lon': (30, 39), 'lat': (-11.5, -2.5)},
            # W2: 4° × 4° centered (RA 08h54m, Dec -04°)
            {'type': 'rect', 'lon': (131.5, 135.5), 'lat': (-6, -2)},
            # W3: 7° × 7° centered (RA 14h17m54s, Dec +54°30')
            {'type': 'rect', 'lon': (210.8, 217.8), 'lat': (51, 58)},
            # W4: 4° × 4° centered (RA 22h13m18s, Dec +01°19')
            {'type': 'rect', 'lon': (331.3, 335.3), 'lat': (-0.7, 3.3)},
        ],
    },
    # ----------------------------------------------------------------
    # Space surveys
    # ----------------------------------------------------------------
    'erosita': {
        'name': 'eROSITA',
        'color': 'indianred',
        'description': 'eROSITA All-Sky Survey (western galactic half)',
        'frame': 'galactic',
        'regions': [
            # eROSITA German consortium: western galactic hemisphere
            # l = 180°–360° (galactic longitude)
            {'type': 'rect', 'lon': (180, 360), 'lat': (-90, 90)},
        ],
    },
    # ----------------------------------------------------------------
    # Radio surveys
    # ----------------------------------------------------------------
    'first': {
        'name': 'FIRST',
        'color': 'darkorange',
        'description': 'Faint Images of the Radio Sky at Twenty-cm',
        'frame': 'icrs',
        'regions': [
            # Main NGC area
            {'type': 'rect', 'lon': (120, 250), 'lat': (-5, 58)},
            # SGC strip
            {'type': 'rect', 'lon': (315, 55), 'lat': (-1.5, 1.5)},
        ],
    },
    'nvss': {
        'name': 'NVSS',
        'color': 'forestgreen',
        'description': 'NRAO VLA Sky Survey',
        'frame': 'icrs',
        'regions': [
            {'type': 'rect', 'lon': (0, 360), 'lat': (-40, 90)},
        ],
    },
    'vlass': {
        'name': 'VLASS',
        'color': 'teal',
        'description': 'VLA Sky Survey',
        'frame': 'icrs',
        'regions': [
            {'type': 'rect', 'lon': (0, 360), 'lat': (-40, 90)},
        ],
    },
    'lotss': {
        'name': 'LoTSS',
        'color': 'royalblue',
        'description': 'LOFAR Two-metre Sky Survey',
        'frame': 'icrs',
        'regions': [
            {'type': 'rect', 'lon': (100, 250), 'lat': (20, 70)},
        ],
    },
    'racs': {
        'name': 'RACS',
        'color': 'slateblue',
        'description': 'Rapid ASKAP Continuum Survey',
        'frame': 'icrs',
        'regions': [
            {'type': 'rect', 'lon': (0, 360), 'lat': (-90, 49)},
        ],
    },
    # ----------------------------------------------------------------
    # Spectroscopic surveys
    # ----------------------------------------------------------------
    'desi': {
        'name': 'DESI',
        'color': 'crimson',
        'description': 'Dark Energy Spectroscopic Instrument',
        'frame': 'icrs',
        'regions': [
            {'type': 'rect', 'lon': (0, 360), 'lat': (-20, 78)},
        ],
    },
    'gama': {
        'name': 'GAMA',
        'color': 'darkcyan',
        'description': 'Galaxy And Mass Assembly survey (5 fields)',
        'frame': 'icrs',
        'regions': [
            # G09 — equatorial
            {'type': 'rect', 'lon': (129, 141), 'lat': (-2, 3)},
            # G12 — equatorial
            {'type': 'rect', 'lon': (174, 186), 'lat': (-3, 2)},
            # G15 — equatorial
            {'type': 'rect', 'lon': (211.5, 223.5), 'lat': (-2, 3)},
            # G02 — early-extension equatorial field
            {'type': 'rect', 'lon': (30.2, 38.8), 'lat': (-10.25, -3.72)},
            # G23 — southern field
            {'type': 'rect', 'lon': (339, 351), 'lat': (-35, -30)},
        ],
    },
    'twodf': {
        'name': '2dF GRS',
        'color': 'mediumorchid',
        'description': ('2dF Galaxy Redshift Survey — NGP + SGP '
                        'strips (Colless et al. 2001)'),
        'frame': 'icrs',
        'regions': [
            # NGP strip: 75° × 10°, centered RA 13h, Dec -2.5°
            {'type': 'rect', 'lon': (157.5, 232.5), 'lat': (-7.5, 2.5)},
            # SGP strip: 75° × 15°, centered RA 0h, Dec -30°
            # (wraps the antimeridian — handled by add_survey_footprint)
            {'type': 'rect', 'lon': (322.5, 37.5), 'lat': (-37.5, -22.5)},
        ],
    },
    'sixdf': {
        'name': '6dF GS',
        'color': 'olive',
        'description': ('6dF Galaxy Survey — southern hemisphere '
                        'with galactic-plane avoidance (~17000 sq deg)'),
        'frame': 'icrs',
        'compound_ops': [
            ('add_latitude_band', {'lat_min': -90, 'lat_max': 0}),
            ('subtract_frame_band',
             {'lat_min': -10, 'lat_max': 10, 'frame': 'galactic'}),
        ],
        'regions': [
            # Fallback rect for boundary_style='box'.
            {'type': 'rect', 'lon': (0, 360), 'lat': (-90, 0)},
        ],
    },
    # ----------------------------------------------------------------
    # CMB / sub-mm
    # ----------------------------------------------------------------
    'spt': {
        'name': 'SPT',
        'color': 'steelblue',
        'description': 'South Pole Telescope (SPT-SZ / SPTpol / SPT-3G)',
        'frame': 'icrs',
        'regions': [
            # SPT-SZ / SPT-3G: southern patch, ~2500 sq deg
            {'type': 'rect', 'lon': (320, 110), 'lat': (-65, -40)},
        ],
    },
    # ----------------------------------------------------------------
    # Upcoming / future
    # ----------------------------------------------------------------
    'lsst': {
        'name': 'LSST/Rubin',
        'color': 'darkviolet',
        'description': 'Legacy Survey of Space and Time (~18000 sq deg)',
        'frame': 'icrs',
        'regions': [
            # Bounding box
            {'type': 'rect', 'lon': (0, 360), 'lat': (-70, 12)},
            # Two-part polygon: southern sky (Dec < +12.4°) minus
            # galactic plane (|b| < ~15°). The GP cut splits the
            # footprint into a large region and a donut-hole gap.
            # Part 1: outer boundary (full southern hemisphere to +12°)
            # plus the galactic plane gap traced as an indentation.
            # East side of GP gap (RA ~75–110)
            {'type': 'polygon', 'vertices': [
                [0, 12], [30, 12], [60, 12], [75, 12],
                # Dip south along eastern GP edge
                [82, 5], [89, -6], [96, -18], [104, -35],
                [112, -50], [122, -62], [138, -70],
                [160, -76], [185, -78], [210, -76],
                [235, -70], [255, -62], [268, -50],
                [277, -35], [286, -18], [293, -6],
                # Rise north along western GP edge
                [299, 5], [305, 12],
                [330, 12], [359, 12],
                # Southern boundary
                [359, -90], [0, -90],
            ]},
            # Part 2: the island inside the GP gap (roughly
            # RA 110–270, Dec -48 to +12, bounded by |b|=15)
            {'type': 'polygon', 'vertices': [
                [116, 8], [125, -6], [134, -18],
                [145, -30], [158, -40], [175, -46],
                [195, -48], [215, -46],
                [230, -40], [243, -30], [253, -18],
                [260, -6], [265, 8],
                [250, 12], [230, 12], [200, 12],
                [170, 12], [140, 12],
            ]},
        ],
        # CompoundRegion recipe: southern sky (Dec ≤ +12.4°) with
        # the galactic plane masked at |b|<15°. Used in preference
        # to the hand-traced polygons above when boundary_style=
        # 'polygon' — handles polar-cap closure cleanly.
        'compound_ops': [
            ('add_latitude_band', {'lat_min': -90, 'lat_max': 12}),
            ('subtract_frame_band',
             {'lat_min': -15, 'lat_max': 15, 'frame': 'galactic'}),
        ],
    },
    'euclid': {
        'name': 'Euclid',
        'color': 'goldenrod',
        'description': 'Euclid Wide Survey (~14500 sq deg)',
        'frame': 'icrs',
        'regions': [
            # Bounding boxes (very rough ICRS approximation of
            # the dual avoidance zone, used for fallback only)
            {'type': 'rect', 'lon': (0, 360), 'lat': (20, 90)},
            {'type': 'rect', 'lon': (0, 360), 'lat': (-90, -20)},
            # Improved polygons in ICRS, computed from the dual
            # avoidance zone: |b| >= 20° AND |β| >= 15°.
            # Creates 4 disconnected regions (2 mainlands + 2 islands).
            # Vertices computed via contour extraction on the ROI mask.
            #
            # Northern mainland (large): RA 0–360, Dec ~0–83
            {'type': 'polygon', 'vertices': [
                [0, 83], [25, 83], [50, 82], [70, 80],
                [82, 76], [90, 67], [99, 56], [107, 42],
                [121, 36], [140, 32], [158, 27], [174, 21],
                [190, 15], [205, 9], [220, 3], [237, -3],
                [253, -7], [263, 0], [270, 14], [277, 28],
                [284, 43], [292, 56], [302, 68], [316, 76],
                [335, 80], [355, 82],
            ]},
            # Southern mainland (large): RA 0–360, Dec ~-83–-5
            {'type': 'polygon', 'vertices': [
                [359, -17], [345, -23], [330, -28],
                [315, -33], [300, -36], [290, -42],
                [283, -50], [277, -60], [268, -70],
                [255, -77], [238, -80], [218, -82],
                [195, -83], [170, -83], [148, -81],
                [132, -77], [118, -71], [108, -61],
                [100, -48], [94, -34], [87, -19],
                [80, -5], [70, 8], [55, 5],
                [38, 0], [22, -6], [8, -12],
            ]},
            # Northern island (small): RA ~0–50, Dec ~16–43
            {'type': 'polygon', 'vertices': [
                [0, 17], [8, 20], [16, 23], [24, 26],
                [32, 29], [40, 32], [48, 34],
                [46, 36], [40, 38], [33, 40],
                [25, 42], [17, 43], [8, 43], [0, 42],
            ]},
            # Southern island (small): RA ~125–225, Dec ~-43–+3
            {'type': 'polygon', 'vertices': [
                [125, 1], [132, -7], [140, -14],
                [148, -21], [156, -28], [165, -34],
                [175, -39], [188, -42], [202, -42],
                [215, -40], [224, -36], [226, -32],
                [216, -30], [206, -27], [196, -23],
                [186, -19], [176, -15], [166, -10],
                [156, -6], [145, -2], [134, 2],
            ]},
        ],
        # CompoundRegion recipe for the dual-avoidance footprint
        # (|b|>=20° AND |β|>=15°). When boundary_style='polygon',
        # this exact set-algebraic representation is used in
        # preference to the hand-traced polygons above — it
        # naturally includes the polar caps and small disconnected
        # islands without needing per-piece vertex lists.
        'compound_ops': [
            ('add_latitude_band', {'lat_min': -90, 'lat_max': 90}),
            ('subtract_frame_band',
             {'lat_min': -20, 'lat_max': 20, 'frame': 'galactic'}),
            ('subtract_frame_band',
             {'lat_min': -15, 'lat_max': 15, 'frame': 'ecliptic'}),
        ],
    },
    'ska': {
        'name': 'SKA',
        'color': 'navy',
        'description': 'Square Kilometre Array (planned)',
        'frame': 'icrs',
        'regions': [
            {'type': 'rect', 'lon': (0, 360), 'lat': (-90, 30)},
        ],
    },
    'hatlas': {
        'name': 'H-ATLAS',
        'color': 'sienna',
        'description': 'Herschel Astrophysical Terahertz Large Area Survey',
        'frame': 'icrs',
        'regions': [
            # GAMA-9 field
            {'type': 'rect', 'lon': (129, 141), 'lat': (-2, 3)},
            # GAMA-12 field
            {'type': 'rect', 'lon': (174, 186), 'lat': (-3, 2)},
            # GAMA-15 field
            {'type': 'rect', 'lon': (211.5, 223.5), 'lat': (-2, 3)},
            # North Galactic Pole (NGP)
            {'type': 'rect', 'lon': (191, 207), 'lat': (25, 36)},
            # South Galactic Pole (SGP)
            {'type': 'rect', 'lon': (350, 54), 'lat': (-36, -26)},
        ],
    },
    # ----------------------------------------------------------------
    # Deep fields (small, iconic — < a few sq deg each; rendered as
    # tiny patches on all-sky frames but visible/labelable on
    # zoomed views)
    # ----------------------------------------------------------------
    # Note for the deep-field entries below: ``polygon`` rather than
    # ``rect`` is used as the primary geometry. ``rect`` would route
    # through ``add_latitude_band → _project_band``, whose complement-
    # detection heuristic mis-classifies a small thin band on a
    # high-zoom TAN frame and fills the whole frame instead. An
    # explicit polygon goes through ``add_spherical_polygon`` and
    # renders correctly on both all-sky and zoomed views. A box-style
    # ``rect`` is also kept as a fallback for ``boundary_style='box'``.
    'cosmos': {
        'name': 'COSMOS',
        'color': 'magenta',
        'description': ('Cosmic Evolution Survey — ~2 sq deg deep '
                        'field at RA 10h, Dec +02° '
                        '(Scoville et al. 2007)'),
        'frame': 'icrs',
        'regions': [
            # 1.4° × 1.4° centered (10h00m28.6s, +02°12'21")
            {'type': 'rect', 'lon': (149.42, 150.82), 'lat': (1.51, 2.91)},
            {'type': 'polygon', 'vertices': [
                [149.42, 1.51], [150.82, 1.51],
                [150.82, 2.91], [149.42, 2.91],
            ]},
        ],
    },
    'uds': {
        'name': 'UKIDSS UDS',
        'color': 'darkmagenta',
        'description': ('UKIDSS Ultra Deep Survey — ~0.8 sq deg '
                        'in XMM-LSS (RA 02h18m, Dec −05°)'),
        'frame': 'icrs',
        'regions': [
            # 0.9° × 0.9° centered (02h18m00s, -05°00'00")
            {'type': 'rect', 'lon': (34.05, 34.95), 'lat': (-5.45, -4.55)},
            {'type': 'polygon', 'vertices': [
                [34.05, -5.45], [34.95, -5.45],
                [34.95, -4.55], [34.05, -4.55],
            ]},
        ],
    },
    'goodsn': {
        'name': 'GOODS-N',
        'color': 'darkgoldenrod',
        'description': ('Great Observatories Origins Deep Survey '
                        '— North (Hubble Deep Field area, '
                        'RA 12h36m55s, Dec +62°14′)'),
        'frame': 'icrs',
        'regions': [
            # 10' × 16' = 0.167° × 0.267° centered (12h36m55s, +62°14'15")
            {'type': 'rect', 'lon': (189.10, 189.36), 'lat': (62.10, 62.37)},
            {'type': 'polygon', 'vertices': [
                [189.10, 62.10], [189.36, 62.10],
                [189.36, 62.37], [189.10, 62.37],
            ]},
        ],
    },
    'goodss': {
        'name': 'GOODS-S',
        'color': 'darkgreen',
        'description': ('Great Observatories Origins Deep Survey '
                        '— South (Chandra Deep Field-South, '
                        'RA 03h32m30s, Dec −27°48′)'),
        'frame': 'icrs',
        'regions': [
            # 10' × 16' centered (03h32m30s, -27°48'30")
            {'type': 'rect', 'lon': (53.00, 53.27), 'lat': (-27.95, -27.68)},
            {'type': 'polygon', 'vertices': [
                [53.00, -27.95], [53.27, -27.95],
                [53.27, -27.68], [53.00, -27.68],
            ]},
        ],
    },
}


def survey_keys() -> list[str]:
    """Return a sorted list of registered survey keys.

    The returned strings are exactly the lookups accepted by
    ``add_survey_footprint(ax, <key>)``. Useful for parametrizing
    panels / tests / iterators over the full registry.

    Examples
    --------
    >>> for key in sph.survey_keys():
    ...     sph.add_survey_footprint(ax, key)
    """
    return sorted(SURVEY_FOOTPRINTS.keys())


def list_surveys(return_data: bool = False) -> list[dict[str, Any]] | None:
    """List available pre-defined survey footprints.

    Parameters
    ----------
    return_data : bool, optional
        If False (default), print a formatted summary table to
        stdout and return ``None``. If True, return a list of
        per-survey dicts with keys ``key``, ``name``, ``frame``,
        ``description``, ``n_regions``, ``has_compound_ops`` —
        suitable for programmatic filtering or display in
        notebooks / GUIs.

    Examples
    --------
    >>> sph.list_surveys()                       # print to stdout
    >>> rows = sph.list_surveys(return_data=True)
    >>> [r['key'] for r in rows if r['has_compound_ops']]
    ['euclid', 'lsst', 'sixdf']
    """
    rows = []
    for key, info in sorted(SURVEY_FOOTPRINTS.items()):
        rows.append({
            'key': key,
            'name': info.get('name', key),
            'frame': info.get('frame', 'icrs'),
            'description': info.get('description', ''),
            'n_regions': len(info.get('regions', [])),
            'has_compound_ops': bool(info.get('compound_ops')),
        })
    if return_data:
        return rows
    print(f"{'Key':<12s} {'Name':<16s} {'Frame':<10s} "
          f"{'Regions':<10s} Description")
    print("-" * 100)
    for r in rows:
        tag = (f"{r['n_regions']}+cmpnd"
               if r['has_compound_ops'] else f"{r['n_regions']}")
        print(f"{r['key']:<12s} {r['name']:<16s} {r['frame']:<10s} "
              f"{tag:<10s} {r['description']}")
    return None


def _apply_footprint_stroke(patches: Any, stroke_color: Any,
                            stroke_lw: float) -> None:
    """Give footprint patches the package's standard outline stroke.

    Every other overlay in the package exposes ``stroke_color`` /
    ``stroke_lw``; footprints did not, and ``**kwargs`` was no escape hatch
    because those reach patch properties, not path effects.
    """
    if stroke_color is None:
        return
    from .._stroke import _stroke_path_effects
    effects = _stroke_path_effects(stroke_color, stroke_lw)
    if not effects:
        return
    for patch in patches:
        patch.set_path_effects(effects)


def add_survey_footprint(ax: Any, survey: str | dict[str, Any],
                         color: str | None = None, alpha: float = 0.15,
                         edgecolor: str | None = None,
                         edge_alpha: float | None = None,
                         lw: float = 1.5, label: str | None = None,
                         zorder: int = 2, fill: bool = True,
                         hatch: str | None = None,
                         boundary_style: str = 'polygon',
                         stroke_color: Any = None, stroke_lw: float = 2.0,
                         clip: str = 'auto', **kwargs: Any) -> list[Any]:
    """
    Add a survey footprint outline/fill to a WCSAxes.

    By default renders as filled regions with transparency
    (``fill=True``, ``alpha=0.15``). Pass ``fill=False`` for
    edge-only outlines.

    Parameters
    ----------
    ax : WCSAxes
    survey : str or dict
        Survey name (key in SURVEY_FOOTPRINTS) or a custom dict with
        the same structure.
    color : str, optional
        Fill/edge color. Defaults to the survey's assigned color.
    alpha : float
        Fill transparency (only used when fill=True).
    edgecolor : str, optional
        Edge color. Defaults to the fill color.
    edge_alpha : float, optional
        Independent opacity for the footprint edge. Default ``None`` —
        the edge inherits the fill ``alpha`` (unchanged behavior). Set a
        value in ``[0, 1]`` to make the edge more/less opaque than the
        fill (e.g. a crisp edge over a faint fill).
    lw : float
        Edge line width.
    label : str, optional
        Legend label. Defaults to the survey name.
    zorder : int
    fill : bool
        If True (default), show filled region with transparency.
        If False, edge only.
    hatch : str, optional
        Hatch pattern (e.g. '//', '\\\\', '..', 'xx')
    stroke_color : color, optional
        Outline stroke drawn behind the footprint edge, matching the
        rest of the overlays. ``None`` (default) draws none. Note that
        ``**kwargs`` cannot supply this — those reach patch
        properties, not path effects.
    stroke_lw : float
        Width of that stroke. Default 2.0.
    boundary_style : str
        'polygon' (default) — use detailed polygon boundaries when
        available in the survey definition. Falls back to 'box' for
        surveys without polygon definitions.
        'box' — use simplified rectangular bounding boxes stored in
        the survey registry. Fast and works everywhere.
    **kwargs
        Additional kwargs passed to the polygon patches.

    Returns
    -------
    patches : list
        The matplotlib patches added.

    Examples
    --------
    >>> add_survey_footprint(ax, 'sdss')
    >>> add_survey_footprint(ax, 'des', alpha=0.3, hatch='//')
    >>> add_survey_footprint(ax, 'lsst', fill=False, lw=2)
    """
    if isinstance(survey, str):
        key = survey.lower().replace('-', '').replace(' ', '').replace('_', '')
        # Try exact match first, then fuzzy
        if key in SURVEY_FOOTPRINTS:
            info = SURVEY_FOOTPRINTS[key]
        else:
            # Try partial match
            matches = [k for k in SURVEY_FOOTPRINTS if key in k]
            if len(matches) == 1:
                info = SURVEY_FOOTPRINTS[matches[0]]
            else:
                raise ValueError(
                    f"Unknown survey '{survey}'. Use list_surveys() to see "
                    f"available options.")
    else:
        info = survey

    fc = color or info.get('color', 'C0')
    ec = edgecolor or fc
    if label is None:
        label = info.get('name', survey)

    patches = []
    label_used = False

    # Compound-region path: when boundary_style='polygon' AND the
    # survey defines a ``compound_ops`` recipe, build a CompoundRegion
    # via set algebra. This is preferred over hand-traced polygons
    # for surveys whose footprints are naturally expressed as bands ±
    # avoidance zones (LSST, Euclid) — the set-algebra closure handles
    # polar caps, antimeridian crossings, and small disconnected
    # islands automatically. The hand-traced polygon entries are kept
    # in the registry as fallbacks (and as documentation of the
    # intended shape) but skipped here.
    compound_ops = info.get('compound_ops') or []
    if boundary_style == 'polygon' and compound_ops:
        from ..geometry.compound import CompoundRegion
        region_obj = CompoundRegion(ax)
        for op_name, op_kwargs in compound_ops:
            getattr(region_obj, op_name)(**op_kwargs)
        render_kw = dict(zorder=zorder, **kwargs)
        _apply_fill_edge_style(render_kw, fill=fill, fc=fc, ec=ec,
                               alpha=alpha, edge_alpha=edge_alpha)
        render_kw['linewidth'] = lw
        if hatch:
            render_kw['hatch'] = hatch
        render_kw['label'] = label
        new_patches = region_obj.render(**render_kw)
        # Subsequent patches should not duplicate the legend entry.
        for p in new_patches[1:]:
            p.set_label('_nolegend_')
        _apply_footprint_stroke(new_patches, stroke_color, stroke_lw)
        return list(new_patches)

    for i, region in enumerate(info.get('regions', [])):
        rtype = region['type']

        # Skip polygon regions if boundary_style='box', and vice versa
        if boundary_style == 'box' and rtype == 'polygon':
            continue
        if boundary_style == 'polygon' and rtype == 'rect':
            # Check if there are polygon regions; if not, fall back to rect
            has_poly = any(r['type'] == 'polygon' for r in info.get('regions', []))
            if has_poly:
                continue

        if rtype == 'rect':
            lon_min, lon_max = region['lon']
            lat_min, lat_max = region['lat']

            survey_frame = info.get('frame', 'icrs').lower()
            wcs_frame = _get_wcs_frame_name(ax)

            patch_kw = dict(zorder=zorder, **kwargs)
            _apply_fill_edge_style(patch_kw, fill=fill, fc=fc, ec=ec,
                                   alpha=alpha, edge_alpha=edge_alpha)
            patch_kw['linewidth'] = lw
            if hatch:
                patch_kw['hatch'] = hatch

            lab = label if not label_used else None
            if lab:
                patch_kw['label'] = lab
                label_used = True

            if survey_frame == wcs_frame:
                # Same frame — use the geometry primitive directly.
                # Handles full-360, polar-touching, and lon-bounded
                # rects via CompoundRegion's antimeridian-aware pipeline.
                if lon_max - lon_min == 360:
                    new_patches = add_latitude_band(
                        ax, lat_min, lat_max, clip=clip, **patch_kw)
                elif lon_max < lon_min:
                    # Wrap-around: split into two non-wrapping bands so
                    # intersect_longitude_band doesn't pick the long way.
                    new_patches = add_latitude_band(
                        ax, lat_min, lat_max,
                        lon_min=lon_min, lon_max=360, clip=clip,
                        **patch_kw)
                    pk2 = dict(patch_kw)
                    pk2.pop('label', None)
                    new_patches = list(new_patches) + list(add_latitude_band(
                        ax, lat_min, lat_max,
                        lon_min=0, lon_max=lon_max, clip=clip, **pk2))
                else:
                    new_patches = add_latitude_band(
                        ax, lat_min, lat_max,
                        lon_min=lon_min, lon_max=lon_max, clip=clip,
                        **patch_kw)
            else:
                # Cross-frame: delegate to the geometry primitive that
                # handles the four-edge outline, antimeridian wrap,
                # polar collapse, and the axes-frame projection in one
                # place.
                new_patches = add_lonlat_box(
                    ax, lat_min, lat_max, lon_min, lon_max,
                    frame=survey_frame, clip=clip, **patch_kw)
            for p in new_patches[1:]:
                p.set_label('_nolegend_')
            patches.extend(new_patches)

        elif rtype == 'polygon':
            # Custom polygon vertices: region['vertices'] = [[lon,lat], ...]
            poly_verts = np.atleast_2d(region['vertices'])
            lons = poly_verts[:, 0].copy()
            lats = poly_verts[:, 1].copy()

            # Frame conversion (same logic as rect)
            survey_frame = info.get('frame', 'icrs').lower()
            wcs_frame = _get_wcs_frame_name(ax)
            if survey_frame != wcs_frame:
                if survey_frame in ('ecliptic', 'geocentrictrueecliptic'):
                    coords = SkyCoord(lon=lons, lat=lats,
                                      frame='geocentrictrueecliptic', unit='deg')
                elif survey_frame == 'galactic':
                    coords = SkyCoord(l=lons, b=lats, frame='galactic',
                                      unit='deg')
                else:
                    coords = SkyCoord(lons, lats, unit='deg', frame='icrs')
                if wcs_frame == 'galactic':
                    gc = coords.galactic
                    lons, lats = gc.l.deg, gc.b.deg
                elif wcs_frame == 'ecliptic':
                    gc = coords.geocentrictrueecliptic
                    lons, lats = gc.lon.deg, gc.lat.deg
                else:
                    gc = coords.icrs
                    lons, lats = gc.ra.deg, gc.dec.deg

            patch_kw = dict(zorder=zorder, **kwargs)
            _apply_fill_edge_style(patch_kw, fill=fill, fc=fc, ec=ec,
                                   alpha=alpha, edge_alpha=edge_alpha)
            patch_kw['linewidth'] = lw
            if hatch:
                patch_kw['hatch'] = hatch
            lab = label if not label_used else None
            if lab:
                patch_kw['label'] = lab
                label_used = True
            # ``min_piece_area=0`` so deep-field surveys (sub-pixel
            # at all-sky scale) survive ``_stitch_and_project``'s
            # default 5 px² sliver filter and still produce a patch
            # the user can find / style. Survey vertex lists are
            # author-provided; we trust them rather than filtering.
            new_patches = add_spherical_polygon(ax, lons, lats,
                                                min_piece_area=0.0,
                                                clip=clip, **patch_kw)
            for p in new_patches[1:]:
                p.set_label('_nolegend_')
            patches.extend(new_patches)

    _apply_footprint_stroke(patches, stroke_color, stroke_lw)
    return patches
