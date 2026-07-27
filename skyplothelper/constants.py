"""Single source of truth for shared data dictionaries.

Contains solar-system body constants (obliquities, rotation periods,
equatorial radii) and reference catalogs (sky positions, radio/filter
bands, facility angular resolutions, sexagesimal separator presets).
Module-coupled registries (projection registry, theme dict, stretch
registry, survey footprints, constellation centers, etc.) live in
their respective submodules, not here.
"""

import astropy.units as u
from astropy.coordinates import SkyCoord

# Axial obliquities (deg) of solar system bodies relative to their orbital plane
obliquities = {
    # Star
    'sun': 7.25,
    # Planets
    'mercury': 0.01,
    'venus': 177.3608,
    'earth': 23.44,
    'mars': 25.19,
    'jupiter': 3.12,
    'saturn': 26.73,
    'uranus': 97.86,
    'neptune': 28.33,
    # Dwarf planets
    'pluto': 119.59,
    'ceres': 4.0,
    'eris': 78.0,       # poorly constrained
    'haumea': 126.0,     # approximate
    'makemake': 29.0,    # approximate
    # Major moons
    'moon': 6.68, 'luna': 6.68,
    'io': 2.213,
    'europa': 0.1,
    'ganymede': 2.214,
    'callisto': 0.0,     # near zero
    'titan': 0.3,
    'enceladus': 0.0,    # near zero
    'triton': 0.0,       # near zero, retrograde orbit
    'charon': 0.0,       # tidally locked to Pluto
}
"""Axial obliquities (degrees) of solar-system bodies, keyed by lowercase name."""

# Rotation periods of solar system bodies
# Sidereal rotation periods
rot_periods = {
    # Star
    'sun': 609.12 * u.h,
    # Planets
    'mercury': 1407.5 * u.h,
    'venus': 5832.5 * u.h,       # retrograde
    'earth': 23.9345 * u.h,
    'mars': 24.623 * u.h,
    'jupiter': 9.925 * u.h,
    'saturn': 10.656 * u.h,
    'uranus': 17.24 * u.h,       # retrograde
    'neptune': 16.11 * u.h,
    # Dwarf planets
    'pluto': 153.29 * u.h,       # retrograde
    'ceres': 9.074 * u.h,
    'eris': 25.9 * u.h,          # approximate
    'haumea': 3.915 * u.h,       # unusually fast
    'makemake': 22.827 * u.h,
    # Major moons (synchronous unless noted)
    'moon': 655.73 * u.h, 'luna': 655.73 * u.h,
    'io': 42.459 * u.h,
    'europa': 85.228 * u.h,
    'ganymede': 171.709 * u.h,
    'callisto': 400.536 * u.h,
    'titan': 382.68 * u.h,
    'enceladus': 32.885 * u.h,
    'triton': 141.043 * u.h,     # retrograde
    'charon': 153.29 * u.h,      # same as Pluto (double tidal lock)
}
"""Sidereal rotation periods of solar-system bodies (astropy hour quantities)."""

# Equatorial radii of solar system objects (km).
planet_radii = {
    'sun': 695700., 'mercury': 2439.7, 'venus': 6051.8,
    'earth': 6371.0, 'moon': 1737.4, 'luna': 1737.4, 'mars': 3389.5,
    'jupiter': 69911., 'io': 1821.6, 'ganymede': 2634.1,
    'saturn': 58232., 'uranus': 25362., 'neptune': 24622., 'pluto': 1188.3,
}
"""Equatorial radii (km) of solar-system objects, keyed by lowercase name."""

# Reference sky positions (ICRS, degrees)
# Useful for quick overlays, annotation, and orientation checks
SKY_POSITIONS = {
    # Galactic structure
    'galactic_center':     SkyCoord(266.417, -28.936, unit='deg'),
    'galactic_anticenter': SkyCoord(86.405, 28.936, unit='deg'),
    'galactic_north_pole': SkyCoord(192.859, 27.128, unit='deg'),
    'galactic_south_pole': SkyCoord(12.859, -27.128, unit='deg'),
    # CMB
    'cmb_dipole':          SkyCoord(167.99, -6.94, unit='deg'),  # Planck 2018
    # Local Group
    'lmc':                 SkyCoord(80.894, -69.756, unit='deg'),
    'smc':                 SkyCoord(13.187, -72.829, unit='deg'),
    'm31':                 SkyCoord(10.685, 41.269, unit='deg'),
    'm33':                 SkyCoord(23.462, 30.660, unit='deg'),
    # Galaxy clusters
    'virgo_cluster':       SkyCoord(187.706, 12.391, unit='deg'),  # M87
    'coma_cluster':        SkyCoord(194.953, 27.981, unit='deg'),  # Abell 1656
    'perseus_cluster':     SkyCoord(49.951, 41.512, unit='deg'),   # Abell 426
    # Calibrators / bright radio sources
    'cyg_a':               SkyCoord(299.868, 40.734, unit='deg'),
    'cas_a':               SkyCoord(350.850, 58.815, unit='deg'),
    'tau_a':               SkyCoord(83.633, 22.015, unit='deg'),   # Crab Nebula
    'vir_a':               SkyCoord(187.706, 12.391, unit='deg'),  # M87
    'cen_a':               SkyCoord(201.365, -43.019, unit='deg'), # NGC 5128
    'her_a':               SkyCoord(252.784, 4.993, unit='deg'),
    '3c273':               SkyCoord(187.278, 2.052, unit='deg'),
    '3c279':               SkyCoord(194.047, -5.789, unit='deg'),
    '3c84':                SkyCoord(49.951, 41.512, unit='deg'),   # Perseus A
    'sgr_a_star':          SkyCoord(266.417, -29.008, unit='deg'), # Sgr A*
    'm87':                 SkyCoord(187.706, 12.391, unit='deg'),
    # Ecliptic reference
    'ecliptic_north_pole': SkyCoord(270.000, 66.561, unit='deg'),
    'ecliptic_south_pole': SkyCoord(90.000, -66.561, unit='deg'),
    'vernal_equinox':      SkyCoord(0.0, 0.0, unit='deg'),
    # Supergalactic
    'supergalactic_north_pole': SkyCoord(283.75, 15.71, unit='deg'),
}
"""Reference sky positions (ICRS :class:`~astropy.coordinates.SkyCoord`), keyed by name."""

# Radio frequency band definitions: name -> (freq_min_GHz, freq_max_GHz)
# IEEE standard designations commonly used in radio astronomy
RADIO_BANDS = {
    'P':  (0.230, 0.470),    # 90-65 cm
    'L':  (1.0, 2.0),        # 30-15 cm (21 cm HI line)
    'S':  (2.0, 4.0),        # 15-7.5 cm
    'C':  (4.0, 8.0),        # 7.5-3.75 cm
    'X':  (8.0, 12.0),       # 3.75-2.5 cm
    'Ku': (12.0, 18.0),      # 2.5-1.67 cm
    'K':  (18.0, 26.5),      # 1.67-1.13 cm (22 GHz H2O maser)
    'Ka': (26.5, 40.0),      # 1.13-0.75 cm
    'Q':  (40.0, 50.0),      # 7.5-6.0 mm (43 GHz SiO maser)
    'W':  (75.0, 110.0),     # 4.0-2.7 mm
    'mm': (30.0, 300.0),     # generic millimeter
    'submm': (300.0, 1000.0), # submillimeter
}
"""Radio band definitions, ``name -> (freq_min_GHz, freq_max_GHz)`` (IEEE designations)."""

# Optical/IR photometric filter central wavelengths (nm) and bandwidths
FILTER_BANDS = {
    # Johnson-Cousins
    'U': {'center_nm': 365, 'width_nm': 66, 'system': 'Johnson'},
    'B': {'center_nm': 445, 'width_nm': 94, 'system': 'Johnson'},
    'V': {'center_nm': 551, 'width_nm': 88, 'system': 'Johnson'},
    'R': {'center_nm': 658, 'width_nm': 138, 'system': 'Cousins'},
    'I': {'center_nm': 806, 'width_nm': 149, 'system': 'Cousins'},
    # SDSS
    'u': {'center_nm': 354, 'width_nm': 57, 'system': 'SDSS'},
    'g': {'center_nm': 477, 'width_nm': 138, 'system': 'SDSS'},
    'r': {'center_nm': 623, 'width_nm': 138, 'system': 'SDSS'},
    'i': {'center_nm': 762, 'width_nm': 153, 'system': 'SDSS'},
    'z': {'center_nm': 913, 'width_nm': 95, 'system': 'SDSS'},
    # 2MASS / near-IR
    'J':  {'center_nm': 1235, 'width_nm': 162, 'system': '2MASS'},
    'H':  {'center_nm': 1662, 'width_nm': 251, 'system': '2MASS'},
    'Ks': {'center_nm': 2159, 'width_nm': 262, 'system': '2MASS'},
    # WISE
    'W1': {'center_nm': 3368, 'width_nm': 663, 'system': 'WISE'},
    'W2': {'center_nm': 4618, 'width_nm': 1042, 'system': 'WISE'},
    'W3': {'center_nm': 12082, 'width_nm': 5511, 'system': 'WISE'},
    'W4': {'center_nm': 22194, 'width_nm': 4101, 'system': 'WISE'},
    # Common HST filters
    'F275W': {'center_nm': 271, 'width_nm': 40, 'system': 'HST/WFC3'},
    'F336W': {'center_nm': 336, 'width_nm': 51, 'system': 'HST/WFC3'},
    'F435W': {'center_nm': 433, 'width_nm': 61, 'system': 'HST/ACS'},
    'F555W': {'center_nm': 531, 'width_nm': 123, 'system': 'HST/ACS'},
    'F606W': {'center_nm': 591, 'width_nm': 218, 'system': 'HST/ACS'},
    'F814W': {'center_nm': 806, 'width_nm': 166, 'system': 'HST/ACS'},
    'F110W': {'center_nm': 1153, 'width_nm': 443, 'system': 'HST/WFC3'},
    'F160W': {'center_nm': 1537, 'width_nm': 268, 'system': 'HST/WFC3'},
    # Common JWST filters
    'F070W':  {'center_nm': 704, 'width_nm': 132, 'system': 'JWST/NIRCam'},
    'F090W':  {'center_nm': 902, 'width_nm': 194, 'system': 'JWST/NIRCam'},
    'F115W':  {'center_nm': 1154, 'width_nm': 225, 'system': 'JWST/NIRCam'},
    'F150W':  {'center_nm': 1501, 'width_nm': 318, 'system': 'JWST/NIRCam'},
    'F200W':  {'center_nm': 1990, 'width_nm': 457, 'system': 'JWST/NIRCam'},
    'F277W':  {'center_nm': 2762, 'width_nm': 672, 'system': 'JWST/NIRCam'},
    'F356W':  {'center_nm': 3568, 'width_nm': 781, 'system': 'JWST/NIRCam'},
    'F444W':  {'center_nm': 4408, 'width_nm': 1029, 'system': 'JWST/NIRCam'},
    'F560W':  {'center_nm': 5600, 'width_nm': 1200, 'system': 'JWST/MIRI'},
    'F770W':  {'center_nm': 7700, 'width_nm': 2200, 'system': 'JWST/MIRI'},
    'F1000W': {'center_nm': 10000, 'width_nm': 2000, 'system': 'JWST/MIRI'},
    'F1280W': {'center_nm': 12800, 'width_nm': 2400, 'system': 'JWST/MIRI'},
    'F2100W': {'center_nm': 21000, 'width_nm': 5000, 'system': 'JWST/MIRI'},
}
"""Photometric filter bands, ``name -> {'center_nm', 'width_nm', 'system'}``."""

# Angular resolution of major astronomical facilities (arcsec)
# Typical or best-case values for common configurations
FACILITY_RESOLUTION = {
    # Radio interferometers
    'vla_a_L':     {'resolution_asec': 1.3,    'band': 'L-band', 'note': 'VLA A-config 1.4 GHz'},
    'vla_a_C':     {'resolution_asec': 0.33,   'band': 'C-band', 'note': 'VLA A-config 5 GHz'},
    'vla_a_X':     {'resolution_asec': 0.20,   'band': 'X-band', 'note': 'VLA A-config 10 GHz'},
    'vla_a_K':     {'resolution_asec': 0.08,   'band': 'K-band', 'note': 'VLA A-config 22 GHz'},
    'vla_a_Ka':    {'resolution_asec': 0.055,  'band': 'Ka-band', 'note': 'VLA A-config 33 GHz'},
    'vla_a_Q':     {'resolution_asec': 0.04,   'band': 'Q-band', 'note': 'VLA A-config 43 GHz'},
    'vla_b_L':     {'resolution_asec': 3.9,    'band': 'L-band', 'note': 'VLA B-config 1.4 GHz'},
    'vla_b_C':     {'resolution_asec': 1.0,    'band': 'C-band', 'note': 'VLA B-config 5 GHz'},
    'vla_c_L':     {'resolution_asec': 12.5,   'band': 'L-band', 'note': 'VLA C-config 1.4 GHz'},
    'vla_d_L':     {'resolution_asec': 46.0,   'band': 'L-band', 'note': 'VLA D-config 1.4 GHz'},
    'vlba_L':      {'resolution_asec': 0.015,  'band': 'L-band', 'note': 'VLBA 1.4 GHz'},
    'vlba_C':      {'resolution_asec': 0.003,  'band': 'C-band', 'note': 'VLBA 5 GHz'},
    'vlba_X':      {'resolution_asec': 0.002,  'band': 'X-band', 'note': 'VLBA 8 GHz'},
    'vlba_K':      {'resolution_asec': 0.0009, 'band': 'K-band', 'note': 'VLBA 22 GHz'},
    'vlba_Q':      {'resolution_asec': 0.0004, 'band': 'Q-band', 'note': 'VLBA 43 GHz'},
    'eht':         {'resolution_asec': 2.5e-5, 'band': '230 GHz', 'note': 'EHT ~25 uas'},
    'alma_max':    {'resolution_asec': 0.005,  'band': 'Band 7', 'note': 'ALMA 16 km baseline 345 GHz'},
    'alma_compact':{'resolution_asec': 1.0,    'band': 'Band 7', 'note': 'ALMA compact 345 GHz'},
    'lofar':       {'resolution_asec': 0.3,    'band': '150 MHz', 'note': 'LOFAR international 150 MHz'},
    'meerkat_L':   {'resolution_asec': 5.0,    'band': 'L-band', 'note': 'MeerKAT L-band'},
    'gmrt_L':      {'resolution_asec': 2.0,    'band': 'L-band', 'note': 'uGMRT band 5'},
    'noema':       {'resolution_asec': 0.2,    'band': '3 mm', 'note': 'NOEMA extended'},
    # Single-dish radio
    'gbt_L':       {'resolution_asec': 510.0,  'band': 'L-band', 'note': 'GBT 100m 1.4 GHz'},
    'effelsberg_C':{'resolution_asec': 144.0,  'band': 'C-band', 'note': 'Effelsberg 100m 5 GHz'},
    'arecibo_L':   {'resolution_asec': 210.0,  'band': 'L-band', 'note': 'Arecibo 305m 1.4 GHz'},
    'fast':        {'resolution_asec': 174.0,  'band': 'L-band', 'note': 'FAST 500m 1.4 GHz'},
    # Space - optical/IR/X-ray
    'hst_optical': {'resolution_asec': 0.05,   'band': 'optical', 'note': 'HST/ACS F606W'},
    'hst_ir':      {'resolution_asec': 0.13,   'band': 'near-IR', 'note': 'HST/WFC3 F160W'},
    'jwst_nircam': {'resolution_asec': 0.031,  'band': '2 um', 'note': 'JWST/NIRCam short'},
    'jwst_miri':   {'resolution_asec': 0.19,   'band': '5.6 um', 'note': 'JWST/MIRI F560W'},
    'chandra':     {'resolution_asec': 0.5,    'band': 'X-ray', 'note': 'Chandra ACIS on-axis'},
    'xmm':         {'resolution_asec': 6.0,    'band': 'X-ray', 'note': 'XMM-Newton EPIC'},
    'nustar':      {'resolution_asec': 18.0,   'band': 'hard X-ray', 'note': 'NuSTAR 3-79 keV'},
    # Ground - optical/IR
    'subaru_hsc':  {'resolution_asec': 0.6,    'band': 'optical', 'note': 'Subaru HSC typical seeing'},
    'vlt_ao':      {'resolution_asec': 0.05,   'band': 'K-band', 'note': 'VLT with AO'},
    'keck_ao':     {'resolution_asec': 0.04,   'band': 'K-band', 'note': 'Keck with AO'},
    'gemini_ao':   {'resolution_asec': 0.06,   'band': 'K-band', 'note': 'Gemini with AO'},
    'seeing_good': {'resolution_asec': 0.7,    'band': 'optical', 'note': 'Typical good seeing'},
    'seeing_avg':  {'resolution_asec': 1.2,    'band': 'optical', 'note': 'Typical average seeing'},
    # Gamma-ray
    'fermi_lat':   {'resolution_asec': 3600.0, 'band': 'gamma-ray', 'note': 'Fermi LAT >1 GeV ~1 deg'},
}
"""Angular resolution (arcsec) of major facilities, ``key -> {'resolution_asec', 'band', 'note'}``."""

# Separator presets for set_separator()
SEPARATORS = {
    # RA / HMS.  Default ('hms_full') uses mathtext superscripts: they
    # render true superscripts in any base font, whereas the Unicode
    # modifier letters ʰ ᵐ ˢ (kept as 'hms_unicode') are missing from
    # some common fonts (e.g. Arial), producing tofu boxes.
    'hms_full':   (r'$^\mathregular{h}$', r'$^\mathregular{m}$', r'$^\mathregular{s}$'),
    'hms_latex':  (r'$^\mathrm{h}$', r'$^\mathrm{m}$', r'$^\mathrm{s}$'),
    'hms_unicode': ('ʰ', 'ᵐ', 'ˢ'),       # Unicode superscripts (font-dependent)
    'hms_letter': ('h', 'm', 's'),                       # plain ASCII
    'hms_colon':  (':',),                                # 05:34:31.9
    'hms_space':  (' ',),                                # IAU tabular
    # DEC / DMS
    'dms_full':   ('°', '′', '″'),        # degree, prime, double-prime
    'dms_letter': ('d', 'm', 's'),                       # plain ASCII
    'dms_colon':  (':',),                                # +22:00:52.2
    'dms_latex':  (r'$^\circ$', r'$^\prime$', r'$^{\prime\prime}$'),
    # Decimal degree separator (just the degree symbol after the number)
    'deg_symbol': ('°',),                           # 180 deg
}
"""Sexagesimal separator presets for tick formatting, keyed by style name."""


# Curated palette for region / patch overlays, chosen to read well
# together when several regions overlap on a single sky map. Use
# ``REGION_PALETTE[i % len(REGION_PALETTE)]`` to cycle, or pick by
# name from ``REGION_PALETTE_NAMED``.
REGION_PALETTE = (
    '#264653',  # deep teal
    '#2A9D8F',  # teal
    '#4ECDC4',  # cyan
    '#E9C46A',  # mustard
    '#E76F51',  # warm coral
    '#FF6B35',  # orange
    '#CC4400',  # rust
    '#C77B4A',  # tan
    '#E8A87C',  # peach
)
"""Ordered hex-color palette for region / patch overlays; cycle with ``[i % len(...)]``."""

REGION_PALETTE_NAMED = {
    'deep_teal': '#264653',
    'teal':      '#2A9D8F',
    'cyan':      '#4ECDC4',
    'mustard':   '#E9C46A',
    'coral':     '#E76F51',
    'orange':    '#FF6B35',
    'rust':      '#CC4400',
    'tan':       '#C77B4A',
    'peach':     '#E8A87C',
}
"""Region overlay palette keyed by color name (see :data:`REGION_PALETTE`)."""


# Short axis / hover labels per coordinate frame. Kept here rather than in each
# renderer because three copies had already diverged: the cartopy backend's
# lacked the fk4/fk5 and spelled-out ecliptic keys, so an fk5 frame silently
# fell back to a generic 'Lon'/'Lat' while the same frame labeled correctly
# elsewhere. Long-form axis labels are a separate concern and stay in ticks.py.
FRAME_SHORT_LABELS = {
    'icrs':                     ('RA', 'Dec'),
    'fk5':                      ('RA', 'Dec'),
    'fk4':                      ('RA', 'Dec'),
    'galactic':                 ('l', 'b'),
    'supergalactic':            ('SGL', 'SGB'),
    'ecliptic':                 ('λ', 'β'),
    'geocentrictrueecliptic':   ('λ', 'β'),
    'barycentrictrueecliptic':  ('λ', 'β'),
    'heliocentrictrueecliptic': ('λ', 'β'),
}
"""Short ``(lon, lat)`` labels per frame, e.g. ``('RA', 'Dec')``."""


def frame_short_labels(frame: str | None,
                       default: tuple[str, str] = ('lon', 'lat'),
                       ) -> tuple[str, str]:
    """Short ``(lon, lat)`` labels for *frame*, e.g. ``('l', 'b')``.

    Matching is case-insensitive and tolerates the ``'ecliptic'`` aliases
    astropy spells out in full.
    """
    if not frame:
        return default
    return FRAME_SHORT_LABELS.get(str(frame).lower(), default)
