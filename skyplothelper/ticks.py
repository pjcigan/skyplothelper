"""Tick label formatting for WCSAxes and matplotlib axes.

Includes the public ``format_ticklabels`` dispatcher (~14 styles), per-axis
helpers (``format_WCS_ticklabels``, ``format_mpl_ticklabels``), the offset and
anchored-offset formatters with their ``apply_*`` convenience functions, and a
range of internal helpers consumed by ``make_wcs_frame`` and the cone /
globe modules.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Callable

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes.frame import EllipticalFrame
from astropy.wcs.utils import proj_plane_pixel_scales, wcs_to_celestial_frame
from matplotlib import rcParams

from ._compat import (
    _ASTROPY_GE_7,
    _compat_format,
    _safe_ticklabel_kwargs,
    coord_ticklabels,
)
from ._stroke import _stroke_path_effects
from ._text_layout import _resolve_text_anchor
from .constants import SEPARATORS
from .core.math_utils import map_to_newrange, wrap_24hr, wrap_360, wrap_pm180
from .geometry._parsing import _spherical_deg
from .projections.frames import _AllSkyCustomFrame

# ===== RA label formatting + SkyCoord helpers =====

def RAlabelformatter(RA_deg: float, style: str | None = 'deg') -> Any:
    """
    Format an RA value in degrees into a tick label string.

    Parameters
    ----------
    RA_deg : float
    style : str or None
        None, 'deg', 'h', 'H', '^h', '^H'
    """
    if style is None:
        return RA_deg
    elif style.lower() in ['d', 'deg']:
        return u'{0:.0f}\u00B0'.format(float(wrap_360(RA_deg)))
    elif style in ['h', 'hr', 'hour']:
        return '{0:.0f}h'.format(float(wrap_24hr(RA_deg / 15)))
    elif style in ['H', 'Hr', 'Hour']:
        return '{0:.0f}H'.format(float(wrap_24hr(RA_deg / 15)))
    elif style in ['^h', '^hr']:
        return r'{0:.0f}$^\mathregular{{h}}$'.format(float(wrap_24hr(RA_deg / 15)))
    elif style in ['^H', '^Hr', '^HR']:
        return r'{0:.0f}$^\mathregular{{H}}$'.format(float(wrap_24hr(RA_deg / 15)))
    else:
        return RA_deg


def RAlabellist(centerRA_deg: float, style: str | None = 'deg',
                RAticklabels_orig: npt.ArrayLike = np.array(
                    [150, 120, 90, 60, 30, 0,
                     -30, -60, -90, -120, -150])) -> np.ndarray:
    """
    Generate formatted RA tick labels for a given center RA.
    """
    RAticks_deg = map_to_newrange(RAticklabels_orig, [-180, 180],
                                  [centerRA_deg - 180, centerRA_deg + 180])
    return np.array([RAlabelformatter(ra, style=style) for ra in RAticks_deg])


def format_SkyCoord_lon_for_mpl(longitudes_deg: Any,
                                center_deg: float = 0.) -> npt.NDArray[np.float64]:
    """
    Format longitude values for matplotlib geographic projections
    (negative radians, RA increasing left).
    """
    if isinstance(longitudes_deg, SkyCoord):
        # Frame-agnostic: `.ra` only exists on equatorial frames, so a galactic
        # or ecliptic SkyCoord used to raise AttributeError here.
        vals = _spherical_deg(longitudes_deg)[0]
    else:
        vals = np.asarray(longitudes_deg)
    coords_plotrange = map_to_newrange(vals, [-180, 180],
                                       [-(center_deg + 180), -(center_deg - 180)])
    return -np.pi / 180 * wrap_pm180(coords_plotrange)


def format_SkyCoord_pairs_for_mpl(coord_pairs_deg: Any,
                                  center_deg: float = 0.) -> np.ndarray:
    """
    Format [lon, lat] coordinate pairs for matplotlib geographic projections.
    """
    if isinstance(coord_pairs_deg, SkyCoord):
        # Frame-agnostic — see format_SkyCoord_lon_for_mpl.
        lons, lats = _spherical_deg(coord_pairs_deg)
    elif hasattr(coord_pairs_deg, '__len__'):
        arr = np.asarray(coord_pairs_deg)
        if arr.ndim < 2:
            if len(arr) > 0 and isinstance(arr[0], SkyCoord):
                pairs = [_spherical_deg(c) for c in arr]
                lons = np.array([p[0] for p in pairs])
                lats = np.array([p[1] for p in pairs])
            else:
                lons = np.array([arr[0]])
                lats = np.array([arr[1]])
        else:
            lons, lats = arr[:, 0], arr[:, 1]
    else:
        raise ValueError('Input must be SkyCoord, array of pairs, or array of SkyCoords')

    lons_fmt = format_SkyCoord_lon_for_mpl(lons, center_deg=center_deg)
    return np.array([lons_fmt, np.radians(lats)]).T




# ===== Stroke-effect tick label formatters (per-axis helpers) =====

def format_WCS_ticklabels(ax: Any, which: str = 'both', fontsize: float | None = None,
                          fontweight: str = 'normal',
                          fontstyle: str = 'normal', color: str | None = None,
                          stroke_lw: float = 1.2,
                          stroke_color: str = 'w', **kwargs: Any) -> None:
    """
    Format tick labels on an astropy WCSAxes with stroke effects.

    Parameters
    ----------
    ax : WCSAxes
    which : str
        'x'/'ra'/'lon', 'y'/'dec'/'lat', or 'both'
    fontsize : float, optional
        Tick-label fontsize in points. ``None`` (default) preserves
        whatever fontsize was already set on the coord — including the
        auto-sized value applied by ``make_wcs_frame(auto_fontsize=True)``.
        Pass an explicit number to override.
    """
    if color is None:
        color = rcParams['xtick.color']
    # ``None`` now disables the stroke rather than reaching the artist as
    # ``foreground=None``, matching the package-wide contract.
    pe = _stroke_path_effects(stroke_color, stroke_lw)
    common = dict(color=color, fontweight=fontweight, fontstyle=fontstyle,
                  path_effects=pe, **kwargs)
    if fontsize is not None:
        common['fontsize'] = fontsize
    if which.lower() in ['x', 'ra', 'l', 'glon', 'lon', 'both']:
        ax.coords[0].set_ticklabel(**common)
    if which.lower() in ['y', 'dec', 'b', 'glat', 'lat', 'both']:
        ax.coords[1].set_ticklabel(**common)


def format_mpl_ticklabels(ax: Any, which: str = 'both', fontsize: float | None = None,
                          fontweight: str = 'normal',
                          fontstyle: str = 'normal', color: str | None = None,
                          stroke_lw: float = 1.2,
                          stroke_color: str = 'w', **kwargs: Any) -> None:
    """
    Format tick labels on a standard matplotlib axis with stroke effects.

    ``fontsize=None`` (default) preserves whatever fontsize was already
    set on the tick labels; pass an explicit number to override.
    """
    if color is None:
        color = rcParams['xtick.color']
    # ``None`` now disables the stroke rather than reaching the artist as
    # ``foreground=None``, matching the package-wide contract.
    pe = _stroke_path_effects(stroke_color, stroke_lw)
    common = dict(color=color, fontweight=fontweight, fontstyle=fontstyle,
                  path_effects=pe, **kwargs)
    if fontsize is not None:
        common['fontsize'] = fontsize
    if which.lower() in ['x', 'ra', 'l', 'glon', 'lon', 'both']:
        for lbl in ax.get_xticklabels():
            lbl.set(**common)
    if which.lower() in ['y', 'dec', 'b', 'glat', 'lat', 'both']:
        for lbl in ax.get_yticklabels():
            lbl.set(**common)


# ===== Frame defaults, frame detection, and core formatters =====


# Frame-to-default-unit mapping
_FRAME_DEFAULT_UNITS = {
    'icrs': (u.hourangle, u.deg),
    'fk5':  (u.hourangle, u.deg),
    'fk4':  (u.hourangle, u.deg),
    'galactic': (u.deg, u.deg),
    'supergalactic': (u.deg, u.deg),
    'ecliptic': (u.deg, u.deg),
    'geocentrictrueecliptic': (u.deg, u.deg),
    'heliocentrictrueecliptic': (u.deg, u.deg),
}


def _detect_frame(ax: Any) -> str:
    """Detect coordinate frame name from a WCSAxes object."""
    if hasattr(ax, '_sph_frame'):
        return ax._sph_frame
    try:
        frame = wcs_to_celestial_frame(ax.wcs)
        return frame.name.lower()
    except Exception:
        try:
            ctype1 = ax.wcs.wcs.ctype[0][:4].upper()
            if ctype1 == 'RA--':
                return 'icrs'
            elif ctype1 == 'GLON':
                return 'galactic'
            elif ctype1 == 'SLON':
                return 'supergalactic'
            elif ctype1 in ('ELON', 'HLON'):
                return 'ecliptic'
        except Exception:
            pass
    return 'icrs'


class OffsetFormatter:
    """
    Offset tick label formatter for WCSAxes.

    Displays tick values as offsets from a reference coordinate, in
    any angular unit (arcsec, mas, μas, arcmin). This is the standard
    approach for zoomed astronomical images where absolute coordinates
    are unwieldy but relative scale matters.

    Works by monkey-patching astropy's internal tick formatter, so it
    integrates with WCSAxes without modifying the WCS itself — data
    can still be plotted using ``transform=ax.get_transform('world')``.

    Parameters
    ----------
    ref_value_deg : float
        Reference coordinate in degrees. Ticks display as offsets
        from this value.
    unit : str
        Offset display unit: 'arcsec' (default), 'mas', 'uas', 'arcmin'
    precision : int
        Number of decimal places for offset labels. Default 1.
    show_sign : bool
        Show +/- on all labels. Default True.
    show_unit : bool
        Append the unit to every tick label (``'+400 mas'``). Default True.
        ``False`` gives bare numbers (``'+400'``, ``'0'``, ``'-200'``) — tidy
        when the axis label already carries the unit, so it isn't repeated on
        every tick.

    Examples
    --------
    >>> fmt = OffsetFormatter(ref_value_deg=330.075, unit='mas', precision=1)
    >>> fmt(330.075)       # '0 mas'
    >>> fmt(330.075001)    # '+3.6 mas'
    >>> fmt(330.074999)    # '-3.6 mas'

    Notes
    -----
    Unlike astropy's ``set_format_unit(u.mas, decimal=True)`` which
    displays the *absolute* coordinate in the specified unit (e.g.,
    ``1188270005 mas``), this formatter computes the offset from a
    meaningful reference position, producing labels like ``+5.0 mas``.

    Use ``apply_offset_ticks()`` to apply this formatter to a WCSAxes,
    or use it directly via the ``_formatter_locator`` monkey-patch
    (see ``apply_anchored_offset()`` source for the pattern).

    See Also
    --------
    AnchoredOffsetFormatter : Extension with inline sexagesimal reference tick.
    apply_offset_ticks : Convenience function to apply offset ticks to axes.
    """

    _UNIT_LABELS = {
        'deg': ('°', 1.),
        'arcmin': ('′', 60.),
        'arcsec': ('″', 3600.),
        'mas': ('mas', 3600e3),
        'uas': ('μas', 3600e6),
    }

    def __init__(self, ref_value_deg: float, unit: str = 'arcsec',
                 precision: int = 1, show_sign: bool = True,
                 cos_factor: float = 1.0, show_unit: bool = True) -> None:
        self.ref_value_deg = float(ref_value_deg)
        self.unit = unit
        self.precision = precision
        self.show_sign = show_sign
        self.cos_factor = float(cos_factor)
        self.show_unit = show_unit

        if unit not in self._UNIT_LABELS:
            raise ValueError(f"unit must be one of {list(self._UNIT_LABELS.keys())}")

    def __call__(self, value_deg: float) -> str:
        """Format a tick value as offset from reference."""
        delta_deg = float(value_deg) - self.ref_value_deg
        delta_deg *= self.cos_factor
        unit_label, factor = self._UNIT_LABELS[self.unit]
        delta_unit = delta_deg * factor

        # Fix negative zero
        if abs(delta_unit) < 0.5 * 10**(-self.precision):
            delta_unit = 0.0

        suffix = f' {unit_label}' if self.show_unit else ''
        if delta_unit == 0:
            return f'0{suffix}'

        sign = '+' if (self.show_sign and delta_unit > 0) else ''
        return f'{sign}{delta_unit:.{self.precision}f}{suffix}'

    def format_ticks(self, values_deg: Sequence[float]) -> list[str]:
        """Format a list of tick values."""
        return [self(v) for v in values_deg]


class AnchoredOffsetFormatter(OffsetFormatter):
    """
    Anchored-offset tick label formatter.

    Extension of ``OffsetFormatter`` that displays one *anchor* tick as a
    full sexagesimal coordinate while showing all others as offsets from it.
    Useful whenever both an absolute reference position and relative scale
    should be visible on the same axis — e.g. a VLBI map anchored on the
    pointing center, or an all-sky plot marking a named source with offsets
    around it.

    For pure offset axes (the standard convention), use ``OffsetFormatter``
    directly or ``apply_offset_ticks()``.

    Parameters
    ----------
    ref_value_deg : float
        Reference coordinate in degrees.
    is_ra : bool
        If True, format reference as hours:min:sec (RA).
        If False, format as deg:arcmin:arcsec (Dec).
    unit : str
        Offset unit: 'mas' (default), 'uas', or 'arcsec'
    ref_precision : int
        Decimal places for the reference tick's sub-field. Default 3.
    offset_precision : int
        Decimal places for offset labels. Default 3.
    sep : str
        Separator style for reference: 'unicode', 'colon', 'letter'
    show_sign : bool
        Show +/- on offset labels. Default True.
    offsets_only : bool
        If True, show all ticks as offsets (no inline reference).
        Equivalent to using ``OffsetFormatter`` directly.
    anchor_format : {'sexagesimal', 'decimal'} or callable
        How the anchor tick is rendered. ``'sexagesimal'`` (default) →
        HMS/DMS; ``'decimal'`` → decimal degrees (``ref_precision`` places);
        a callable ``f(value_deg) -> str`` for full control. Offset ticks are
        unaffected.

    Examples
    --------
    >>> fmt = AnchoredOffsetFormatter(ref_value_deg=330.075, is_ra=True, unit='mas')
    >>> fmt(330.075)    # '22ʰ00ᵐ18.000ˢ'  (reference tick)
    >>> fmt(330.0751)   # '+360.000 mas'     (offset tick)

    >>> # Decimal-degree anchor (natural for galactic l/b)
    >>> fmt = AnchoredOffsetFormatter(ref_value_deg=120.5, unit='mas',
    ...                               anchor_format='decimal', ref_precision=4)
    >>> fmt(120.5)      # '+120.5000°'
    """

    def __init__(self, ref_value_deg: float, is_ra: bool = False,
                 unit: str = 'mas', ref_precision: int = 3,
                 offset_precision: int = 3, sep: str = 'super',
                 show_sign: bool = True, offsets_only: bool = False,
                 cos_factor: float = 1.0,
                 anchor_format: str | Callable[[float], str]
                 = 'sexagesimal') -> None:
        super().__init__(ref_value_deg, unit=unit, precision=offset_precision,
                         show_sign=show_sign, cos_factor=cos_factor)
        self.is_ra = is_ra
        self.ref_precision = ref_precision
        self.offset_precision = offset_precision
        self.sep = sep
        self.offsets_only = offsets_only
        if (not callable(anchor_format)
                and anchor_format not in ('sexagesimal', 'sex', 'decimal',
                                          'deg')):
            raise ValueError(
                "anchor_format must be 'sexagesimal', 'decimal', or a "
                f"callable; got {anchor_format!r}")
        self.anchor_format = anchor_format

    def _format_reference(self, value_deg: float) -> str:
        """Format the anchor value per ``anchor_format``.

        ``'sexagesimal'`` (default) renders full HMS (RA) / DMS (Dec) — for RA
        split across two lines (HM on top, S on bottom) so the wide anchor
        label doesn't get dropped by astropy's overlap exclusion at small FOVs.
        ``'decimal'`` renders decimal degrees (``ref_precision`` places, RA in
        0–360°, Dec signed) — the natural choice for galactic / ecliptic
        anchors. A callable receives ``value_deg`` and returns the label.
        """
        if callable(self.anchor_format):
            return self.anchor_format(value_deg)
        if self.anchor_format in ('decimal', 'deg'):
            if self.is_ra:
                return f'{value_deg % 360.0:.{self.ref_precision}f}°'
            return f'{value_deg:+.{self.ref_precision}f}°'
        if self.is_ra:
            # Convert degrees → hours
            hours_total = value_deg / 15.0
            if hours_total < 0:
                hours_total += 24.0
            h = int(hours_total)
            remainder = (hours_total - h) * 60.0
            m = int(remainder)
            s = (remainder - m) * 60.0

            # Handle rounding carry-over (59.9999... → 60.0)
            s = round(s, self.ref_precision)
            if s >= 60.0:
                s -= 60.0
                m += 1
            if m >= 60:
                m -= 60
                h += 1
            if h >= 24:
                h -= 24

            if self.sep == 'super':
                # mathtext superscripts — portable across fonts (default).
                seps = (r'$^\mathregular{h}$', r'$^\mathregular{m}$', r'$^\mathregular{s}$')
            elif self.sep == 'unicode':
                seps = ('\u02B0', '\u1D50', '\u02E2')  # ʰ ᵐ ˢ
            elif self.sep == 'colon':
                seps = (':', ':', '')
            else:  # letter
                seps = ('h', 'm', 's')

            s_str = f'{s:0{self.ref_precision + 3}.{self.ref_precision}f}'
            # Two-line form: 'HHʰMMᵐ\nSS.sssˢ' — narrower bbox so adjacent
            # offset ticks aren't hidden by overlap exclusion.
            return (f'{h:02d}{seps[0]}{m:02d}{seps[1]}\n'
                    f'{s_str}{seps[2]}')
        else:
            # Dec: degrees, arcmin, arcsec
            sign = '+' if value_deg >= 0 else '-'
            val = abs(value_deg)
            d = int(val)
            remainder = (val - d) * 60.0
            m = int(remainder)
            s = (remainder - m) * 60.0

            # Handle rounding carry-over (59.9999... → 60.0)
            s = round(s, self.ref_precision)
            if s >= 60.0:
                s -= 60.0
                m += 1
            if m >= 60:
                m -= 60
                d += 1

            if self.sep in ('super', 'unicode'):
                seps = ('\u00B0', '\u2032', '\u2033')  # ° ′ ″
            elif self.sep == 'colon':
                seps = (':', ':', '')
            else:  # letter
                seps = ('d', "'", '"')

            s_str = f'{s:0{self.ref_precision + 3}.{self.ref_precision}f}'
            return f'{sign}{d:02d}{seps[0]}{m:02d}{seps[1]}{s_str}{seps[2]}'

    def _format_offset(self, value_deg: float) -> str:
        """Format as offset — delegates to OffsetFormatter.__call__."""
        return super().__call__(value_deg)

    def __call__(self, value_deg: float) -> str:
        """
        Format a single tick value.

        Reference tick → full sexagesimal (unless offsets_only=True).
        All other ticks → offset from reference.
        """
        # Check if this is the reference tick (within ~0.1 μas tolerance)
        if abs(float(value_deg) - self.ref_value_deg) < 1e-10:
            if self.offsets_only:
                return super().__call__(value_deg)
            return self._format_reference(value_deg)
        return super().__call__(value_deg)

    def format_ticks(self, values_deg: Sequence[float]) -> list[str]:
        """Format a list of tick values."""
        return [self(v) for v in values_deg]


def _enable_minor_ticks_for_explicit_tick_values(coord: Any,
                                                  frequency: int = 5) -> None:
    """Turn on minor ticks for a coord, patching the locator if needed.

    Astropy's ``AngleFormatterLocator.minor_locator`` short-circuits
    to an empty array when ``self.values`` is set (i.e. when major
    ticks come from an explicit value list rather than a spacing).
    That means ``coord.display_minor_ticks(True)`` is silently a
    no-op for axes built with ``apply_offset_ticks`` /
    ``apply_anchored_offset`` (both of which need exact major tick
    positions in cosine-corrected angular units, so they go through
    the ``values=`` code path).

    When ``coord``'s ``_formatter_locator`` has ``values`` set, this
    helper installs a replacement ``minor_locator`` that
    interpolates ``frequency - 1`` minor positions between
    consecutive explicit majors *and* extrapolates with the same
    step beyond the first / last major to fill the visible range
    (matching matplotlib's standard minor-tick behavior). When
    ``values`` is not set, astropy's stock minor locator already
    works — the helper just enables display.

    Also calls ``coord.set_minor_frequency`` and
    ``coord.display_minor_ticks(True)`` so the caller doesn't need
    those extra two lines.

    Parameters
    ----------
    coord : astropy.visualization.wcsaxes.CoordinateHelper
        The coord whose minor ticks need restoring. Typically
        ``ax.coords[0]`` or ``ax.coords[1]``.
    frequency : int
        Number of subdivisions per major-tick interval — i.e. the
        rendered minor tick count between two adjacent majors is
        ``frequency - 1``. Default 5.
    """
    fl = coord._formatter_locator
    # Drop any interpolating minor locator WE previously installed before
    # deciding what to do, so this is idempotent and never keeps a stale one.
    # A prior install bakes the majors at install time; if the majors have
    # since changed — most importantly ``set_ticks(spacing=...)`` switching to
    # spacing-based majors (``values`` → None) without clearing our locator —
    # the stale closure would keep subdividing the OLD majors (e.g. 0.5° majors
    # at freq 4 → spurious 0.125° minors). Popping the instance override here
    # restores astropy's native ``minor_locator`` method. (Mirrors the
    # restore-then-reinstall pattern in style._apply_tick_stroke.)
    if getattr(fl.__dict__.get('minor_locator'), '_sph_interp_minor', False):
        fl.__dict__.pop('minor_locator', None)

    major_values = getattr(fl, 'values', None)
    if major_values is None or len(major_values) < 2:
        # Majors are spacing-based (or absent) — astropy's stock spacing-based
        # minor locator works; just enable display at the requested frequency.
        coord.set_minor_frequency(int(frequency))
        coord.display_minor_ticks(True)
        return
    sorted_majors = np.sort(major_values.to_value(u.deg))

    def _minor_locator(spacing: Any, freq: Any, value_min: float,
                       value_max: float) -> Any:
        f = int(freq)
        lo, hi = (value_min, value_max) if value_min <= value_max \
            else (value_max, value_min)
        # The major values may be expressed in a different 360° branch than
        # astropy's (value_min, value_max) — e.g. RA offset majors at 187.7°
        # vs a wrapped axis range at -172.3°, a full turn apart. Shift the
        # majors by whole turns onto the visible branch first; otherwise the
        # extrapolation below would march a full 360° at a sub-arcsec minor
        # step (hundreds of millions of iterations → frozen draw). The shift
        # is a multiple of 360°, so the minors stay the same sky positions.
        majors = np.asarray(sorted_majors, dtype=float)
        shift = round((0.5 * (lo + hi)
                       - 0.5 * (majors[0] + majors[-1])) / 360.0) * 360.0
        majors = majors + shift
        minors = []
        # Interior minors between consecutive majors.
        for i in range(len(majors) - 1):
            a = majors[i]
            b = majors[i + 1]
            step = (b - a) / f
            for k in range(1, f):
                minors.append(a + k * step)
        # Extrapolate beyond the first / last major using the mean major
        # spacing (caller-supplied majors are evenly spaced in the offset /
        # anchored cases; mean handles uneven inputs too). Capped as a failsafe
        # against any residual runaway from an unexpected (value_min, value_max).
        avg_step = (majors[-1] - majors[0]) / (len(majors) - 1)
        minor_step = avg_step / f
        max_extra = 1000
        if minor_step > 0:
            v = majors[0] - minor_step
            n = 0
            while v >= lo and n < max_extra:
                minors.append(v)
                v -= minor_step
                n += 1
            v = majors[-1] + minor_step
            n = 0
            while v <= hi and n < max_extra:
                minors.append(v)
                v += minor_step
                n += 1
        return np.array(sorted(minors)) * u.deg

    setattr(_minor_locator, '_sph_interp_minor', True)  # tag for clearing
    fl.minor_locator = _minor_locator
    coord.set_minor_frequency(int(frequency))
    coord.display_minor_ticks(True)


def apply_offset_ticks(ax: Any, ref_ra_deg: float | None = None,
                       ref_dec_deg: float | None = None,
                       unit: str = 'auto', spacing: Any = None,
                       precision: int | None = None, show_unit: bool = True,
                       axis_labels: bool | dict[str, Any] = True,
                       fontsize: float | None = None, color: str | None = None,
                       stroke_lw: float | None = None,
                       stroke_color: str | None = None,
                       minor_ticks: bool = True, minor_frequency: int = 5,
                       auto_fontsize: bool = True) -> None:
    """
    Apply offset tick labels to a WCSAxes.

    All ticks show offsets from a reference position in an appropriate
    angular unit. The reference coordinate is optionally shown in the
    axis label. This is the standard convention for zoomed astronomical
    images (VLBI, HST, Chandra, etc.).

    Use this on an ordinary **celestial** WCS — typically a ``TAN`` field
    from :func:`make_wcs_frame` / :func:`offset_figure` centered on the real
    sky position of interest. The WCS stays celestial (so you can still
    overplot catalog sources by RA/Dec, and the reference coordinate can
    appear in the axis label); only the tick *labels* become offsets. There
    is no need to synthesize a linear-offset WCS whose coordinates are the
    offsets themselves — that fights the tick machinery for no benefit. For
    absolute sexagesimal / decimal-degree labels instead, use
    :func:`format_ticklabels`.

    Parameters
    ----------
    ax : WCSAxes
    ref_ra_deg, ref_dec_deg : float, optional
        Reference coordinates in degrees. If None, uses the image center.
    unit : str
        Offset unit: ``'auto'`` (default, chosen from the field of view,
        deg → μas), ``'deg'``, ``'arcmin'``, ``'arcsec'``, ``'mas'``, ``'uas'``.
    spacing : float, `~astropy.units.Quantity`, or pair, optional
        Major-tick spacing. Default ``None`` picks a "nice" 1/2/5/10 value
        from the field of view. A bare number is in ``unit``; an angular
        ``Quantity`` (e.g. ``5 * u.arcsec``) is converted regardless of
        ``unit``; a ``(lon, lat)`` pair sets each axis independently.
    precision : int, optional
        Decimal places. If None, auto-chosen from tick spacing.
    show_unit : bool, optional
        Append the unit to every tick label (``'+400 mas'``). Default True.
        Pass ``False`` for bare numbers (``'+400'``, ``'0'``, ``'-200'``) when
        the axis label already carries the unit — the usual case here, since
        ``axis_labels=True`` puts ``(mas)`` in the axis title, so the per-tick
        unit is redundant.
    axis_labels : bool
        If True, set axis labels showing offset unit.
    fontsize, color, stroke_lw, stroke_color : optional
        Label styling. ``fontsize`` sizes the tick labels; ``color`` and the
        stroke pair (``stroke_lw`` + ``stroke_color`` together) style both the
        tick labels *and* the axis labels, so a recolored offset frame stays
        one piece.
    minor_ticks : bool, optional
        If ``True`` (default), render minor ticks between the major
        offset ticks. ``minor_frequency - 1`` minor ticks land
        between each pair of adjacent majors. Astropy's stock
        minor-tick path returns empty when major ticks are set via
        ``set_ticks(values=...)``; this function installs a
        replacement locator (see
        :func:`_enable_minor_ticks_for_explicit_tick_values`) that
        interpolates positions between the explicit majors. Set to
        ``False`` to keep the spine bare between major ticks.
    minor_frequency : int, optional
        Number of subdivisions per major-tick interval. Default 5
        (4 minor ticks between adjacent majors). Ignored when
        ``minor_ticks=False``.
    auto_fontsize : bool, optional
        Re-trigger ``make_wcs_frame``'s auto-fontsize heuristic after
        the format switch. Default ``True``. Needed because
        ``make_wcs_frame`` originally sized the labels for the long
        HMS / DMS defaults (e.g. ``"12h00m02.0s"`` — 11 chars), but
        offset labels are much shorter (``"+20 ″"`` — 5 chars), so
        a recompute lets the fontsize grow back toward the rcParams
        ceiling. Pass ``auto_fontsize=False`` (or an explicit
        ``fontsize=``) to keep the prior size.

    Examples
    --------
    >>> ax = sph.make_wcs_frame(111, 'TAN', center=(83.6, 22.0),
    ...                         npix=500, cdelt=0.01/3600)
    >>> sph.apply_offset_ticks(ax, unit='mas')

    >>> # Auto unit and precision from field of view
    >>> sph.apply_offset_ticks(ax)

    >>> # Render at half the minor density (1 minor between each major)
    >>> sph.apply_offset_ticks(ax, minor_frequency=2)
    """
    wcs = ax.wcs

    # Size the FOV / tick spacing / default reference. By default this uses the
    # full image (``wcs.pixel_shape``), exactly as before. But when the axes
    # have been cropped to a sub-window (``set_xlim``/``set_ylim``), size from
    # the VISIBLE window instead, so re-applying after a crop lays
    # appropriately finer ticks rather than the full-image spacing (which would
    # leave only the central "0" tick in a small crop). An uncropped window
    # falls through to the full-image branch, so full-frame behavior is
    # byte-identical to before.
    x0, x1 = sorted(float(v) for v in ax.get_xlim())
    y0, y1 = sorted(float(v) for v in ax.get_ylim())
    if wcs.pixel_shape is not None:
        nx_full, ny_full = wcs.pixel_shape[0], wcs.pixel_shape[1]
        # Visible window clipped to the image bounds.
        xv0, xv1 = max(x0, -0.5), min(x1, nx_full - 0.5)
        yv0, yv1 = max(y0, -0.5), min(y1, ny_full - 0.5)
        nxv, nyv = abs(xv1 - xv0), abs(yv1 - yv0)
        cropped = (np.isfinite(nxv) and np.isfinite(nyv)
                   and nxv >= 2.0 and nyv >= 2.0
                   and (nxv < nx_full - 1.0 or nyv < ny_full - 1.0))
        if cropped:
            nx, ny = nxv, nyv
            cx, cy = (xv0 + xv1) / 2.0, (yv0 + yv1) / 2.0
        else:
            nx, ny = float(nx_full), float(ny_full)
            cx, cy = nx_full / 2.0, ny_full / 2.0
    else:
        nx = abs(x1 - x0) if np.isfinite(x1 - x0) and abs(x1 - x0) >= 2.0 \
            else 100.0
        ny = abs(y1 - y0) if np.isfinite(y1 - y0) and abs(y1 - y0) >= 2.0 \
            else 100.0
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    # Determine reference from the (visible-window or image) center if absent.
    if ref_ra_deg is None or ref_dec_deg is None:
        center_world = wcs.pixel_to_world_values(cx, cy)
        if ref_ra_deg is None:
            ref_ra_deg = float(center_world[0])
        if ref_dec_deg is None:
            ref_dec_deg = float(center_world[1])

    # Auto unit from field of view. _auto_offset_unit returns the display
    # label ('μas'); map it to the _UNIT_LABELS key ('uas') so a sub-mas field
    # doesn't KeyError.
    if unit == 'auto':
        try:
            pix_scales = proj_plane_pixel_scales(wcs)
            fov_deg = max(pix_scales[0] * nx, pix_scales[1] * ny)
            _, auto_label = _auto_offset_unit(fov_deg)
            unit = {'μas': 'uas'}.get(auto_label, auto_label)
        except Exception:
            unit = 'arcsec'

    # Auto precision from tick spacing
    if precision is None:
        try:
            pix_scales = proj_plane_pixel_scales(wcs)
            fov_asec = max(pix_scales) * max(nx, ny) * 3600
            tick_asec = fov_asec / 4.0
            _, factor = OffsetFormatter._UNIT_LABELS[unit]
            tick_in_unit = tick_asec / 3600 * factor
            precision = max(0, int(-np.log10(max(tick_in_unit, 1e-20)) + 1))
            precision = min(precision, 4)
        except Exception:
            precision = 1

    # Compute symmetric tick positions centered on the reference
    _, unit_factor = OffsetFormatter._UNIT_LABELS[unit]
    cos_dec = np.cos(np.radians(ref_dec_deg))

    for ci, ref_deg in [(0, ref_ra_deg), (1, ref_dec_deg)]:
        # Determine field extent in offset units for this axis
        # proj_plane_pixel_scales returns angular scales (already
        # includes cos(dec) via the CD matrix), so half_fov is in
        # angular units — correct for both axes.
        try:
            pix_scales = proj_plane_pixel_scales(wcs)
            if ci == 0:
                fov_deg = pix_scales[0] * nx
            else:
                fov_deg = pix_scales[1] * ny
            half_fov = fov_deg / 2.0 * unit_factor
        except Exception:
            half_fov = 100.0

        # Tick spacing in offset units: an explicit ``spacing`` (a number in
        # ``unit``, an angular Quantity, or a per-axis pair) wins; otherwise
        # pick a "nice" 1/2/5/10 value from the field of view.
        if spacing is not None:
            _s = spacing[ci] if isinstance(spacing, (tuple, list)) else spacing
            nice_spacing = _coerce_offset_spacing(_s, unit_factor)
        else:
            raw_spacing = half_fov / 3.0
            mag = 10 ** np.floor(np.log10(max(abs(raw_spacing), 1e-20)))
            nice_steps = [1, 2, 5, 10, 20, 50]
            nice_spacing = mag
            for s in nice_steps:
                candidate = s * mag
                if candidate >= raw_spacing * 0.8:
                    nice_spacing = candidate
                    break

        # Generate symmetric ticks: 0, ±spacing, ±2*spacing, ... The count is
        # clamped so a tiny explicit ``spacing`` can't enumerate a runaway
        # number of ticks and freeze the draw.
        if not (np.isfinite(nice_spacing) and nice_spacing > 0):
            continue
        n_ticks = min(int(np.ceil(half_fov / nice_spacing)) + 1, 500)
        offsets = []
        for k in range(-n_ticks, n_ticks + 1):
            offsets.append(k * nice_spacing)

        # Convert offsets back to world coordinates (degrees).
        # For RA, angular offset → coordinate offset requires / cos(dec).
        if ci == 0:
            tick_world = [ref_deg + off / unit_factor / cos_dec
                          for off in offsets]
        else:
            tick_world = [ref_deg + off / unit_factor for off in offsets]
        ax.coords[ci].set_ticks(values=tick_world * u.deg)

        # Apply offset formatter with cos(dec) correction for RA
        cos_fac = cos_dec if ci == 0 else 1.0
        fmt = OffsetFormatter(ref_value_deg=ref_deg, unit=unit,
                              precision=precision, cos_factor=cos_fac,
                              show_unit=show_unit)
        fl = ax.coords[ci]._formatter_locator
        def _make_fmt(f: OffsetFormatter) -> Callable[..., list[str]]:
            def _formatter(values: Any, spacing: Any,
                           format: str = 'auto') -> list[str]:
                return [f(float(v.to(u.deg).value) if hasattr(v, 'to')
                          else float(v)) for v in values]
            return _formatter
        fl.formatter = _make_fmt(fmt)

    # Axis labels — adapt to coordinate frame
    unit_label = OffsetFormatter._UNIT_LABELS[unit][0]
    if isinstance(axis_labels, dict):
        if 'lon' in axis_labels:
            ax.coords[0].set_axislabel(axis_labels['lon'])
        if 'lat' in axis_labels:
            ax.coords[1].set_axislabel(axis_labels['lat'])
    elif axis_labels:
        frame = _detect_frame(ax)
        is_equatorial = frame in ('icrs', 'fk5', 'fk4')
        is_galactic = frame == 'galactic'
        is_supergalactic = frame == 'supergalactic'

        if is_equatorial:
            center_coord = SkyCoord(ref_ra_deg, ref_dec_deg, unit='deg')
            lon_str = center_coord.ra.to_string(unit=u.hourangle, sep='hms',
                                                precision=4, pad=True)
            lat_str = center_coord.dec.to_string(unit=u.deg, sep='dms',
                                                  precision=3, alwayssign=True,
                                                  pad=True)
            lon_sym = '\u0394\u03b1 cos \u03b4'
            lat_sym = '\u0394\u03b4'
        elif is_galactic:
            lon_str = f'{ref_ra_deg:.4f}\u00b0'
            lat_str = f'{ref_dec_deg:+.4f}\u00b0'
            lon_sym = '\u0394l cos b'
            lat_sym = '\u0394b'
        elif is_supergalactic:
            lon_str = f'{ref_ra_deg:.4f}\u00b0'
            lat_str = f'{ref_dec_deg:+.4f}\u00b0'
            lon_sym = '\u0394SGL cos SGB'
            lat_sym = '\u0394SGB'
        else:
            lon_str = f'{ref_ra_deg:.4f}\u00b0'
            lat_str = f'{ref_dec_deg:+.4f}\u00b0'
            lon_sym = '\u0394\u03bb cos \u03b2'
            lat_sym = '\u0394\u03b2'

        ax.coords[0].set_axislabel(
            f'{lon_sym} ({unit_label})  \u2014  {lon_str}')
        ax.coords[1].set_axislabel(
            f'{lat_sym} ({unit_label})  \u2014  {lat_str}')

    # Extend the tick-label color / stroke to the axis labels too, so a
    # recolored offset frame (e.g. white text on a dark background) reads as
    # one piece instead of leaving the axis labels their default color. Only
    # when labels exist; re-set with the current text so nothing else changes.
    if axis_labels:
        axislabel_style: dict[str, Any] = {}
        if color is not None:
            axislabel_style['color'] = color
        _pe = _stroke_path_effects(stroke_color, stroke_lw)
        if _pe is not None:
            axislabel_style['path_effects'] = _pe
        if axislabel_style:
            for ci in (0, 1):
                ax.coords[ci].set_axislabel(
                    ax.coords[ci].get_axislabel(), **axislabel_style)

    # Styling
    # Always disable `simplify` — that feature suppresses shared leading
    # sexagesimal fields across adjacent ticks (e.g. common hours in
    # '12h00m', '12h01m'), which is inappropriate for our offset labels:
    # for a symmetric negative–positive list it would strip the common '-'
    # prefix from consecutive negatives (e.g. '-2', '-1' → '-2', '1').
    # See https://github.com/astropy/astropy issue tracking for details.
    kw: dict[str, Any] = {'simplify': False}
    if fontsize is not None:
        kw['size'] = fontsize
    if color is not None:
        kw['color'] = color
    _pe = _stroke_path_effects(stroke_color, stroke_lw)
    if _pe is not None:
        kw['path_effects'] = _pe
    ax.coords[0].set_ticklabel(**_safe_ticklabel_kwargs(kw))
    ax.coords[1].set_ticklabel(**_safe_ticklabel_kwargs(kw))

    # Restore minor ticks. Both coords have ``values=`` set above, so
    # astropy's stock minor-tick locator returns empty; the helper
    # installs an interpolating replacement and enables display.
    if minor_ticks:
        for ci in (0, 1):
            _enable_minor_ticks_for_explicit_tick_values(
                ax.coords[ci], frequency=minor_frequency)

    # Re-trigger auto-fontsize: make_wcs_frame sized the labels for
    # the (long) HMS / DMS defaults; the offset labels we just installed
    # are much shorter, so a recompute lets fontsize grow back toward
    # the rcParams ceiling. Skipped when caller pinned ``fontsize=``
    # explicitly or opted out via ``auto_fontsize=False``. The call is
    # try/excepted: auto-fontsize is a convenience, never a reason for
    # apply_offset_ticks to fail.
    if auto_fontsize and fontsize is None:
        from .autosize import auto_size_ticklabels
        try:
            ax.figure.canvas.draw()
        except Exception:
            pass
        try:
            auto_size_ticklabels(ax)
        except Exception as exc:
            warnings.warn(
                f"apply_offset_ticks: auto_fontsize failed "
                f"({type(exc).__name__}: {exc}); keeping the prior "
                f"fontsize. Pass auto_fontsize=False to suppress.",
                UserWarning, stacklevel=2)


def _nice_offset_spacing(raw: float) -> float:
    """Round ``raw`` up to the nearest ``1/2/5/10·10ⁿ`` value (offset units)."""
    if not np.isfinite(raw) or raw <= 0:
        return 1.0
    mag = 10.0 ** np.floor(np.log10(raw))
    for s in (1, 2, 5, 10, 20, 50):
        candidate = s * mag
        if candidate >= raw * 0.8:
            return float(candidate)
    return float(10 * mag)


def _anchored_axis_half_fov(ax: Any, coord_idx: int, unit_factor: float,
                            cos_dec: float) -> float:
    """Half field-of-view of ``coord_idx`` in offset units (angular)."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    cx = (xlim[0] + xlim[1]) / 2.0
    cy = (ylim[0] + ylim[1]) / 2.0
    if coord_idx == 0:
        wa = ax.wcs.pixel_to_world_values(xlim[0], cy)
        wb = ax.wcs.pixel_to_world_values(xlim[1], cy)
        rng = abs(float(wb[0]) - float(wa[0])) * cos_dec
    else:
        wa = ax.wcs.pixel_to_world_values(cx, ylim[0])
        wb = ax.wcs.pixel_to_world_values(cx, ylim[1])
        rng = abs(float(wb[1]) - float(wa[1]))
    return (rng / 2.0) * unit_factor


def _coerce_offset_spacing(s: Any, unit_factor: float) -> float:
    """A spacing → a float in the offset unit.

    A bare number is taken as already in the offset ``unit``. An astropy
    ``Quantity`` (any angular unit, e.g. ``2 * u.arcmin``) is converted via
    degrees, so the spacing unit is independent of the label ``unit``.
    """
    if hasattr(s, 'to'):  # astropy Quantity
        return float(s.to(u.deg).value) * unit_factor
    return float(s)


def _resolve_anchored_offset_spacing(
        ax: Any,
        spacing: Any,
        unit_factor: float, cos_dec: float) -> tuple[float, float]:
    """Resolve the (lon, lat) major-tick spacing in the offset unit.

    ``spacing=None`` picks one shared "nice" 1/2/5/10 value (so lon and lat
    carry the same round increment); a scalar (number or angular ``Quantity``)
    applies to both axes; a ``(lon, lat)`` pair sets each axis explicitly.
    """
    if spacing is not None:
        if isinstance(spacing, (tuple, list)):
            return (_coerce_offset_spacing(spacing[0], unit_factor),
                    _coerce_offset_spacing(spacing[1], unit_factor))
        return (_coerce_offset_spacing(spacing, unit_factor),
                _coerce_offset_spacing(spacing, unit_factor))
    try:
        lon_half = _anchored_axis_half_fov(ax, 0, unit_factor, cos_dec)
        lat_half = _anchored_axis_half_fov(ax, 1, unit_factor, cos_dec)
    except Exception:
        return 1.0, 1.0
    # ~3 major intervals per half-FOV; share the coarser nice value so both
    # axes show the same round increment without overcrowding the smaller one.
    nice = max(_nice_offset_spacing(lon_half / 3.0),
               _nice_offset_spacing(lat_half / 3.0))
    return nice, nice


def _offset_label_decimals(spacing_off: float, cap: int = 6) -> int:
    """Minimal decimal places to render multiples of ``spacing_off`` exactly.

    The offset spacing is a nice ``1/2/5·10ⁿ`` value, so the labels are
    integers when the spacing is ≥ 1 unit (e.g. 20 mas → ``"20 mas"``, not
    ``"20.000 mas"``) and gain just enough decimals for sub-unit spacings
    (0.5 → 1 place). Mirrors the auto-precision the pure-offset / absolute
    paths already use; the anchored-offset path previously fixed it at 3.
    """
    if not np.isfinite(spacing_off) or spacing_off <= 0:
        return 1
    return min(max(0, -int(np.floor(np.log10(spacing_off)))), cap)


def apply_anchored_offset(ax: Any, ref_tick: str | int = 'center',
                      unit: str = 'mas',
                      spacing: Any = None,
                      ref_precision: int = 3,
                      offset_precision: int | None = None,
                      sep: str = 'super',
                      anchor_format: str | Callable[[float], str]
                      = 'sexagesimal',
                      fontsize: float | None = None,
                      color: str | None = None,
                      stroke_lw: float | None = None,
                      stroke_color: str | None = None,
                      axis_labels: bool | dict[str, Any] = True,
                      compact: bool = False,
                      lat_rotation: float | None = None,
                      lon_rotation: float | None = None,
                      minor_ticks: bool = True, minor_frequency: int = 5,
                      auto_fontsize: bool = True,
                      max_ticks: int = 500) -> None:
    """
    Apply anchored-offset tick labels to a WCSAxes.

    Shows one *anchor* tick as a full sexagesimal coordinate and all other
    ticks as mas/μas/arcsec offsets from that anchor — for cases where both
    an absolute reference position and relative offsets are wanted on the
    same axis. The canonical use is a VLBI map anchored on the pointing
    center, but it works at any scale (e.g. an all-sky plot marking a named
    source, with offsets read around it).

    For pure offset axes (all ticks as offsets, no inline anchor), use
    ``format_ticklabels(ax, style='offset_mas')`` instead — the conventional
    approach for published offset maps.

    Parameters
    ----------
    ax : WCSAxes
        The axes to format. Must have been created with make_wcs_frame
        or equivalent.
    ref_tick : str or int
        Which tick to use as the full-coordinate reference:
        'center' (default) — the tick closest to the axis center,
        'first' — the first tick,
        'last' — the last tick,
        int — specific tick index.
    unit : str
        Offset unit: ``'deg'``, ``'arcmin'``, ``'arcsec'``, ``'mas'`` (default),
        ``'uas'``, or ``'auto'`` (pick a sensible unit from the field of view,
        deg → μas — recommended for non-mas fields so the offsets don't default
        to a tiny/huge mas scale).
    spacing : float, `~astropy.units.Quantity`, or pair, optional
        Major-tick spacing. A bare number is in ``unit`` (e.g. ``spacing=10``
        is 10 mas when ``unit='mas'``); an angular ``Quantity`` is converted
        regardless of ``unit`` (e.g. ``spacing=2 * u.arcmin``). Default
        ``None`` chooses one "nice" round value (1/2/5/10·10ⁿ) shared by both
        axes. Pass a scalar for both axes, or a ``(lon, lat)`` pair — each a
        number or ``Quantity`` — to set them independently
        (e.g. ``spacing=(10, 20)`` or ``spacing=(1*u.arcmin, 2*u.arcmin)``).
    ref_precision : int
        Decimal places for the anchor tick — the arcseconds/seconds field
        in sexagesimal mode, or the degrees field when
        ``anchor_format='decimal'``.
    offset_precision : int, optional
        Decimal places for the offset labels. Default ``None`` auto-chooses
        the minimum needed for the tick spacing — round spacings get integer
        labels (``"20 mas"``, not ``"20.000 mas"``) and sub-unit spacings gain
        just enough places. Pass an int to force a fixed precision.
    sep : str
        Separator style for a sexagesimal anchor: 'unicode', 'colon', 'letter'
    anchor_format : {'sexagesimal', 'decimal'} or callable
        How the anchor tick's absolute coordinate is rendered. Default
        ``'sexagesimal'`` (HMS/DMS). ``'decimal'`` shows decimal degrees
        (``ref_precision`` places; RA in 0–360°, Dec signed) — the natural
        choice for galactic/ecliptic anchors. A callable ``f(value_deg) -> str``
        gives full control (e.g. decimal hours, a fixed label). The offset
        ticks are unaffected.
    fontsize : float, optional
    color : str, optional
    stroke_lw : float, optional
    stroke_color : str, optional
    axis_labels : bool or dict
        If True (default), set axis labels indicating the offset unit.
        If dict, use custom labels as ``{'lon': ..., 'lat': ...}``.
    compact : bool
        If True, rotate tick labels (lat 45°, lon -30°), reduce
        fontsize to 85%, and right-align labels at the tick mark.
        Useful for tight multi-panel layouts.
    lat_rotation : float, optional
        Explicit rotation angle for latitude tick labels (degrees).
        Overrides compact default. 0 = horizontal, 90 = vertical.
    lon_rotation : float, optional
        Explicit rotation angle for longitude tick labels (degrees).
        Overrides compact default. 0 = horizontal.
    minor_ticks : bool, optional
        If ``True`` (default), render minor ticks between the major
        offset ticks. Same mechanism as
        :func:`apply_offset_ticks` — installs an interpolating
        replacement locator (see
        :func:`_enable_minor_ticks_for_explicit_tick_values`)
        because astropy's stock minor-tick path returns empty when
        the major ticks were set via ``set_ticks(values=...)``.
    minor_frequency : int, optional
        Number of subdivisions per major-tick interval. Default 5
        (4 minor ticks between adjacent majors).
    auto_fontsize : bool, optional
        Re-trigger ``make_wcs_frame``'s auto-fontsize heuristic on the
        lat coord after the format switch. Default ``True``. Skipped
        in ``compact`` mode (which has its own fontsize scaling) and
        when the caller pinned ``fontsize=`` explicitly.
    max_ticks : int, optional
        Cap on ticks enumerated per side, so a very fine ``spacing`` cannot
        generate a runaway number of them and stall the draw. Default 500.
        A spacing finer than about ``half_axis / max_ticks`` hits the cap,
        at which point the ticks stop partway across the axis; a warning is
        issued when that happens, since correctly-spaced ticks that simply
        end early look intentional. Raise it if you genuinely want them.

    Notes
    -----
    The reference tick is displayed in full sexagesimal (e.g.,
    ``22ʰ00ᵐ18.000ˢ`` for RA or ``+22°00′54.000″`` for Dec).
    Other ticks show the offset: ``+5.0 mas``, ``-5.0 mas``, etc.
    The reference label is longer than offset labels, which can cause
    crowding — use ``compact=True`` for tighter layouts.

    Because this replaces astropy's formatter, call it as the last
    formatting step. Subsequent calls to ``format_ticklabels()`` will
    override it.

    **Implementation:** Works by monkey-patching the ``formatter`` method
    on astropy's internal ``_formatter_locator`` object for each
    coordinate axis. This intercepts within astropy's render pipeline,
    so the anchored labels persist across redraws and ``savefig()`` calls.
    A draw callback approach was tried earlier but astropy recomputes
    tick text on each draw, overwriting callback modifications.

    **For pure offset axes** (all ticks as offsets, no inline anchor), use
    ``format_ticklabels(ax, style='offset_mas')`` instead — the conventional
    approach for published offset maps.

    **Minor ticks** are not rendered after this call; the anchored layout
    sets explicit RA tick positions via ``set_ticks(values=...)`` and
    astropy then skips computing minor-tick pixel positions. See the
    same note in :func:`apply_offset_ticks` for details. Use a
    sexagesimal style if minor ticks are needed.

    Examples
    --------
    >>> ax = sph.make_wcs_frame(111, 'TAN', center=(330.075, 22.015),
    ...                         npix=500, cdelt=0.5/3600)
    >>> fig.canvas.draw()  # ensure ticks are computed
    >>> sph.apply_anchored_offset(ax, unit='mas')

    >>> # With μas offsets for space VLBI
    >>> sph.apply_anchored_offset(ax, unit='uas', ref_precision=5)

    >>> # Decimal-degree anchor (e.g. a galactic-frame field)
    >>> sph.apply_anchored_offset(ax, unit='arcsec', anchor_format='decimal')

    >>> # Custom anchor label via a callable
    >>> sph.apply_anchored_offset(ax, anchor_format=lambda d: f'M87 ({d:.3f}°)')
    """
    frame = _detect_frame(ax)
    is_equatorial = frame in ('icrs', 'fk5', 'fk4')

    # Resolve unit='auto' from the visible field of view (deg → μas), so a
    # wide field gets a sensible unit instead of defaulting to mas. Map the
    # FOV-picker's label form to a _UNIT_LABELS key ('μas' → 'uas').
    if unit == 'auto':
        try:
            scales = proj_plane_pixel_scales(ax.wcs)
            xs = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
            ys = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
            fov_deg = max(scales[0] * xs, scales[1] * ys)
            _, auto_label = _auto_offset_unit(fov_deg)
            unit = {'μas': 'uas'}.get(auto_label, auto_label)
        except Exception:
            unit = 'mas'

    unit_label, unit_factor = AnchoredOffsetFormatter._UNIT_LABELS.get(
        unit, ('mas', 3600e3))

    # Compute cos(latitude) for longitude → angular offset correction.
    # Applies to any spherical frame (equatorial, galactic, ecliptic, ...).
    cos_dec = 1.0
    if hasattr(ax, 'wcs'):
        try:
            cx = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2.0
            cy = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2.0
            _w = ax.wcs.pixel_to_world_values(cx, cy)
            cos_dec = np.cos(np.radians(float(_w[1])))
        except Exception:
            cos_dec = 1.0

    # Deterministic, round tick spacing on BOTH axes. Earlier this forced only
    # the lon axis to FOV/5 and left lat (and compact-mode lon) to astropy's
    # auto-locator — which picks values "nice" in RA seconds-of-time, i.e.
    # non-round in mas and dependent on the axes pixel size. Instead pick one
    # nice 1/2/5/10 spacing in the offset unit and lay symmetric ticks about
    # the field center, so the increments are round and (by default) identical
    # for lon and lat. ``spacing=`` overrides this per axis.
    lon_spacing_off, lat_spacing_off = _resolve_anchored_offset_spacing(
        ax, spacing, unit_factor, cos_dec)

    for coord_idx in (0, 1):
        is_ra = (coord_idx == 0 and is_equatorial)
        spacing_off = lon_spacing_off if coord_idx == 0 else lat_spacing_off

        # Force a draw so the world↔pixel transform and axes limits are valid.
        try:
            fig = ax.get_figure()
            fig.canvas.draw()
        except Exception:
            pass

        coord = ax.coords[coord_idx]

        # Field center + half-extent for this axis, then symmetric round ticks
        # at the resolved offset spacing (converted to world degrees; lon also
        # divides by cos(dec) since the offset is the angular Δα·cosδ).
        try:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            cx = (xlim[0] + xlim[1]) / 2.0
            cy = (ylim[0] + ylim[1]) / 2.0
            center_world = float(ax.wcs.pixel_to_world_values(cx, cy)[coord_idx])
            if coord_idx == 0:
                wa = ax.wcs.pixel_to_world_values(xlim[0], cy)
                wb = ax.wcs.pixel_to_world_values(xlim[1], cy)
                half_world = abs(float(wb[0]) - float(wa[0])) / 2.0
                spacing_world = spacing_off / unit_factor / max(cos_dec, 1e-9)
            else:
                wa = ax.wcs.pixel_to_world_values(cx, ylim[0])
                wb = ax.wcs.pixel_to_world_values(cx, ylim[1])
                half_world = abs(float(wb[1]) - float(wa[1])) / 2.0
                spacing_world = spacing_off / unit_factor
        except Exception:
            center_world = 0.0
            spacing_world = spacing_off / unit_factor
            half_world = spacing_world * 3.0

        if not np.isfinite(spacing_world) or spacing_world <= 0:
            continue

        # ~one tick past each edge; clamp the count so a tiny user spacing
        # can't enumerate a runaway number of ticks and freeze the draw.
        # Past the clamp the ticks stay correctly spaced but stop partway
        # across the axis, which looks deliberate rather than truncated --
        # so say so rather than letting the axis quietly go unticked.
        n_wanted = int(np.ceil(half_world / spacing_world)) + 1
        n = min(n_wanted, max_ticks)
        if n_wanted > max_ticks:
            # Name the axis: both coords are walked here, so an unlabeled
            # message would arrive twice looking like a duplicate rather than
            # like two separate axes each being truncated.
            _axis_name = 'lon' if coord_idx == 0 else 'lat'
            warnings.warn(
                f"apply_anchored_offset: the {_axis_name} axis needs "
                f"{n_wanted} ticks per side at spacing="
                f"{spacing_off:.3g} {unit} to span the field, above "
                f"max_ticks={max_ticks}; its ticks will stop partway across. "
                f"Use a coarser spacing, or raise max_ticks.", stacklevel=2)
        tick_vals = center_world + np.arange(-n, n + 1) * spacing_world
        try:
            coord.set_ticks(values=tick_vals * u.deg)
        except Exception:
            pass

        # Choose reference tick
        ref_idx: int
        if ref_tick == 'center':
            ref_idx = int(np.argmin(np.abs(tick_vals - center_world)))
        elif ref_tick == 'first':
            ref_idx = 0
        elif ref_tick == 'last':
            ref_idx = len(tick_vals) - 1
        elif isinstance(ref_tick, int):
            ref_idx = min(ref_tick, len(tick_vals) - 1)
        else:
            ref_idx = len(tick_vals) // 2

        ref_val = tick_vals[ref_idx]

        # Create formatter and apply. Offset precision auto-adapts to this
        # axis's (round) spacing unless the caller pinned it explicitly.
        cos_fac = cos_dec if coord_idx == 0 else 1.0
        axis_offset_precision = (offset_precision if offset_precision is not None
                                 else _offset_label_decimals(spacing_off))
        fmt = AnchoredOffsetFormatter(
            ref_value_deg=ref_val, is_ra=is_ra, unit=unit,
            ref_precision=ref_precision,
            offset_precision=axis_offset_precision,
            sep=sep, offsets_only=False, cos_factor=cos_fac,
            anchor_format=anchor_format
        )

        # Store reference string for axis labels
        ref_str = fmt._format_reference(ref_val)
        if not hasattr(ax, '_sph_anchored_ref_strings'):
            ax._sph_anchored_ref_strings = {}
        ax._sph_anchored_ref_strings[coord_idx] = ref_str

        # Override astropy's internal formatter method on the
        # _formatter_locator. This is the clean integration point —
        # it intercepts within astropy's render pipeline, so the hybrid
        # labels persist across redraws and savefig calls.
        fl = coord._formatter_locator
        original_formatter = fl.formatter  # save reference

        def _make_hybrid_formatter(anchored_fmt: AnchoredOffsetFormatter,
                                   orig_fmt: Any,
                                   center_val: float) -> Callable[..., list[str]]:
            """Create a closure that picks the reference tick dynamically
            from the actual values astropy passes, rather than relying
            on exact value matching."""
            def hybrid_formatter(values: Any, spacing: Any,
                                 format: str = 'auto') -> list[str]:
                vals_deg = [float(v.to(u.deg).value) if hasattr(v, 'to')
                            else float(v) for v in values]
                # Pick the tick closest to center as the reference
                if vals_deg:
                    ref_idx = int(np.argmin([abs(v - center_val)
                                            for v in vals_deg]))
                    anchored_fmt.ref_value_deg = vals_deg[ref_idx]
                return [anchored_fmt(v) for v in vals_deg]
            return hybrid_formatter

        fl.formatter = _make_hybrid_formatter(fmt, original_formatter,
                                                center_world)

        # Store formatter on the axes for reference
        if not hasattr(ax, '_sph_anchored_formatters'):
            ax._sph_anchored_formatters = {}
        ax._sph_anchored_formatters[coord_idx] = fmt

    # Store the unit for axis labels
    ax._sph_anchored_unit = unit

    # Set axis labels — adapt to coordinate frame
    if axis_labels is True:
        is_galactic = frame == 'galactic'
        is_supergalactic = frame == 'supergalactic'
        if is_equatorial:
            ax.coords[0].set_axislabel(
                f'RA (J2000) / \u0394\u03b1 cos \u03b4 ({unit_label})')
            ax.coords[1].set_axislabel(
                f'Dec (J2000) / \u0394\u03b4 ({unit_label})')
        elif is_galactic:
            ax.coords[0].set_axislabel(
                f'Gal. Lon / \u0394l cos b ({unit_label})')
            ax.coords[1].set_axislabel(
                f'Gal. Lat / \u0394b ({unit_label})')
        elif is_supergalactic:
            ax.coords[0].set_axislabel(
                f'SGL / \u0394SGL cos SGB ({unit_label})')
            ax.coords[1].set_axislabel(
                f'SGB / \u0394SGB ({unit_label})')
        else:
            ax.coords[0].set_axislabel(
                f'Longitude / \u0394\u03bb cos \u03b2 ({unit_label})')
            ax.coords[1].set_axislabel(
                f'Latitude / \u0394\u03b2 ({unit_label})')
    elif isinstance(axis_labels, dict):
        if 'lon' in axis_labels:
            ax.coords[0].set_axislabel(axis_labels['lon'])
        if 'lat' in axis_labels:
            ax.coords[1].set_axislabel(axis_labels['lat'])

    # Apply font styling + rotation.
    # `simplify=False`: offset labels can share a leading '-' or '+' between
    # adjacent ticks, which the sexagesimal simplifier would strip (see
    # note in apply_offset_ticks). Default `exclude_overlapping=True` is
    # kept — combined with 5 explicit RA positions above, astropy keeps
    # the non-overlapping subset (typically 3-5 visible) so the relative
    # scale stays readable without piling labels on top of each other.
    lon_kw: dict[str, Any] = {'simplify': False}
    lat_kw: dict[str, Any] = {'simplify': False}
    if fontsize is not None:
        lon_kw['size'] = fontsize
        lat_kw['size'] = fontsize
    if color is not None:
        lon_kw['color'] = color
        lat_kw['color'] = color
    pe = _stroke_path_effects(stroke_color, stroke_lw)
    if pe is not None:
        lon_kw['path_effects'] = pe
        lat_kw['path_effects'] = pe

    # Extend the tick-label color / stroke to the axis labels too, so a
    # recolored or stroked anchored-offset frame reads as one piece — parity
    # with apply_offset_ticks (882-896) and format_ticklabels, whose axis
    # labels are styled the same way. Re-set with the current text so nothing
    # else changes; no-op when neither color nor stroke was given.
    if axis_labels:
        _axislabel_style: dict[str, Any] = {}
        if color is not None:
            _axislabel_style['color'] = color
        if pe is not None:
            _axislabel_style['path_effects'] = pe
        if _axislabel_style:
            for _ci in (0, 1):
                ax.coords[_ci].set_axislabel(
                    ax.coords[_ci].get_axislabel(), **_axislabel_style)

    # Compact mode: rotate both axis labels +45° so each reads bottom-left to
    # top-right with the END (e.g. "mas") of the text at the tick mark. With
    # ha='right' + rotation_mode='anchor', the right edge of the unrotated
    # bbox is the rotation pivot and stays pinned to the tick. The body of
    # the text extends down-and-to-the-left of the pivot, which lands in
    # the figure margin (below for the bottom RA axis, to the left for the
    # left Dec axis). A negative-rotation lon would tilt the body up-and-
    # left into the plot area instead — wrong direction.
    if compact:
        lat_kw['rotation'] = 45
        lon_kw['rotation'] = 45
        lat_kw['ha'] = 'right'
        lon_kw['ha'] = 'right'
        lat_kw['rotation_mode'] = 'anchor'
        lon_kw['rotation_mode'] = 'anchor'
        # va: lat keeps 'center' for vertical labels; lon uses 'top' so
        # the bbox top aligns with the tick (label hangs in the margin).
        lat_kw['va'] = 'center'
        lon_kw['va'] = 'top'
        # Scale down fontsize in compact mode for cleaner spacing
        base_fs = fontsize or rcParams.get('xtick.labelsize', 10)
        if isinstance(base_fs, str):
            try:
                from matplotlib.font_manager import font_scalings
                base_fs = font_scalings.get(base_fs, 10) * rcParams['font.size']
            except Exception:
                base_fs = 10
        compact_fs = base_fs * 0.85
        lat_kw['size'] = compact_fs
        if 'size' not in lon_kw:
            lon_kw['size'] = compact_fs
        # Disable overlap exclusion so uniform offset labels aren't hidden
        # under the wider anchor in tight compact layouts.
        lat_kw['exclude_overlapping'] = False
        lon_kw['exclude_overlapping'] = False
    if lat_rotation is not None:
        lat_kw['rotation'] = lat_rotation
    if lon_rotation is not None:
        lon_kw['rotation'] = lon_rotation

    ax.coords[0].set_ticklabel(**_safe_ticklabel_kwargs(lon_kw))
    ax.coords[1].set_ticklabel(**_safe_ticklabel_kwargs(lat_kw))

    # Astropy's TickLabels._set_xy_alignments hard-codes ha/va per axis
    # side (bottom: ha='center' va='bottom'; left: ha='right' va='center';
    # etc.) and overrides any ha/va we pass through set_ticklabel. We need
    # two specific overrides for the hybrid layout to look right:
    #
    #   1. Multi-line RA anchor: the bottom-axis default va='bottom' puts
    #      the bbox bottom at the tick, which means the *top* line of the
    #      multi-line anchor extends UP into the plot area. Override the
    #      anchor tick to va='top' so its top line sits where a single-line
    #      label would and the seconds line hangs below.
    #   2. Compact-mode rotated labels: 'center' alignment puts the
    #      bbox center on the tick, so wide and narrow labels visually
    #      drift relative to their tick marks. ha='right' (lon) /
    #      ha='right' (lat) anchor the label END at the tick.
    #
    # Implemented by wrapping _set_xy_alignments to apply the override
    # after astropy's default alignment runs.
    _install_anchored_offset_alignment_override(ax, compact=compact)

    # Enable minor ticks on both coords. coords[0] (lon) has
    # ``values=`` set (explicit 5-tick RA layout) and needs the
    # interpolating-locator patch; coords[1] (lat) uses astropy's
    # default spacing-based locator, so the helper just enables
    # display there.
    if minor_ticks:
        for ci in (0, 1):
            _enable_minor_ticks_for_explicit_tick_values(
                ax.coords[ci], frequency=minor_frequency)

    # Re-trigger auto-fontsize on the lat coord (the lon coord has its
    # own compact-mode fontsize handling above). The original
    # make_wcs_frame fontsize was sized for the (long) HMS / DMS
    # defaults; the new mas / uas / arcsec offset labels are much
    # shorter, so a recompute lets fontsize grow back toward the
    # rcParams ceiling. Skipped when the caller pinned ``fontsize=``
    # explicitly or opted out via ``auto_fontsize=False``. The call is
    # try/excepted: auto-fontsize is a convenience, never a reason for
    # apply_anchored_offset to fail.
    if auto_fontsize and fontsize is None and not compact:
        from .autosize import auto_size_ticklabels
        try:
            ax.figure.canvas.draw()
        except Exception:
            pass
        try:
            auto_size_ticklabels(ax)
        except Exception as exc:
            warnings.warn(
                f"apply_anchored_offset: auto_fontsize failed "
                f"({type(exc).__name__}: {exc}); keeping the prior "
                f"fontsize. Pass auto_fontsize=False to suppress.",
                UserWarning, stacklevel=2)


def _install_anchored_offset_alignment_override(ax: Any, compact: bool = False) -> None:
    """Override astropy's hard-coded ha/va for anchored_offset tick labels.

    See ``apply_anchored_offset`` for rationale. Wraps each visible axis-side
    alignment with the required override (anchor tick → va='top'; in
    compact mode all rotated labels → ha='right').
    """
    for coord_idx, kind in ((0, 'lon'), (1, 'lat')):
        coord = ax.coords[coord_idx]
        tl = coord_ticklabels(coord)
        if getattr(tl, '_sph_anchored_align_patched', False):
            # Already patched on a previous apply_anchored_offset call; refresh
            # the compact flag and the per-tick override runs at draw time.
            tl._sph_anchored_compact = compact
            tl._sph_anchored_kind = kind
            continue
        original = tl._set_xy_alignments
        tl._sph_anchored_compact = compact
        tl._sph_anchored_kind = kind

        def patched(renderer: Any, _orig: Any = original,
                    _tl: Any = tl) -> None:
            _orig(renderer)
            # One text line height in display pixels — used to nudge the
            # multi-line anchor up so its top line occupies the same row as
            # the single-line offset labels.
            line_h_px = renderer.points_to_pixels(_tl.get_size()) * 1.2
            for axis_side in list(_tl.va.keys()):
                for i in list(_tl.va[axis_side].keys()):
                    txt = _tl.text[axis_side][i] if i < len(_tl.text[axis_side]) else ''
                    if txt and '\n' in txt:
                        # Multi-line anchor: pin the top line at the tick.
                        # Switching from astropy's va='bottom' to va='top'
                        # alone shifts the whole bbox down by line_height
                        # (since the bbox now extends below the position
                        # rather than above). Compensate by offsetting the
                        # y-position upward by one line height so the top
                        # row sits where the single-line offsets do.
                        n_extra_lines = txt.count('\n')
                        _tl.va[axis_side][i] = 'top'
                        x, y = _tl.xy[axis_side][i]
                        side = axis_side if isinstance(axis_side, str) else None
                        if side == 'b':
                            _tl.xy[axis_side][i] = (x, y + n_extra_lines * line_h_px)
                        elif side == 't':
                            _tl.xy[axis_side][i] = (x, y - n_extra_lines * line_h_px)
                    if _tl._sph_anchored_compact:
                        # Compact mode: end-align rotated labels with marks
                        _tl.ha[axis_side][i] = 'right'

        tl._set_xy_alignments = patched
        tl._sph_anchored_align_patched = True


def _auto_sexagesimal_format(ax: Any) -> tuple[str, str]:
    """Choose sexagesimal format precision appropriate for the field of view.

    Returns (lon_fmt, lat_fmt) strings with enough decimal places in the
    seconds field to resolve individual ticks across the field.
    """
    wcs = ax.wcs
    try:
        pix_scales = proj_plane_pixel_scales(wcs)
        ps = wcs.pixel_shape
        if ps is not None and len(ps) >= 2:
            nx, ny = ps[0], ps[1]
        else:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            nx = int(abs(xlim[1] - xlim[0]) + 1)
            ny = int(abs(ylim[1] - ylim[0]) + 1)
        fov_deg = max(pix_scales[0] * nx, pix_scales[1] * ny)
    except Exception:
        return 'hh:mm:ss.s', '+dd:mm:ss'

    fov_arcsec = fov_deg * 3600.0

    if fov_arcsec > 3600:
        return 'hh:mm:ss', '+dd:mm:ss'
    elif fov_arcsec > 60:
        return 'hh:mm:ss.s', '+dd:mm:ss.s'
    elif fov_arcsec > 5:
        return 'hh:mm:ss.ss', '+dd:mm:ss.ss'
    elif fov_arcsec > 0.5:
        return 'hh:mm:ss.sss', '+dd:mm:ss.sss'
    elif fov_arcsec > 0.05:
        return 'hh:mm:ss.ssss', '+dd:mm:ss.ssss'
    elif fov_arcsec > 0.005:
        return 'hh:mm:ss.sssss', '+dd:mm:ss.sssss'
    else:
        return 'hh:mm:ss.ssssss', '+dd:mm:ss.ssssss'


def _decimal_format_str(decimal_places: int, signed: bool) -> str:
    """An astropy decimal-degree format string with ``decimal_places`` digits.
    ``0`` → integer degrees (``'d'`` / ``'+d'``), no trailing dot."""
    base = '+d' if signed else 'd'
    if decimal_places <= 0:
        return base
    return base + '.' + 'd' * decimal_places


def _auto_decimal_places(coord: Any, cap: int = 6) -> int | None:
    """Decimal places matching an axis's explicit tick-value spacing.

    Returns ``None`` for a *spacing-based* locator — there astropy's own
    decimal auto-precision reads the spacing and is correct. For a
    *values-based* locator (ticks set via ``set_ticks(values=…)``, e.g. zoomed
    field frames) astropy has no spacing to read and would render full float
    precision (~10 dp ≈ μas on a few-degree field); this derives the precision
    from the spacing between the explicit values instead.
    """
    fl = coord._formatter_locator
    if getattr(fl, 'spacing', None) is not None:
        return None
    vals = getattr(fl, 'values', None)
    if vals is None or len(vals) < 2:
        return None
    arr = np.unique(np.asarray(vals.to_value(u.deg), dtype=float))
    diffs = np.diff(arr)
    diffs = diffs[diffs > 1e-12]
    if not len(diffs):
        return None
    step = float(diffs.min())
    return int(min(cap, max(0, np.ceil(-np.log10(step)))))


def _snap_compact_minute_ticks(coord: Any, unit_deg: float) -> None:
    """Thin a values-based locator to a whole-*unit* grid for the 'compact' style.

    The compact style truncates to minutes (``hh:mm`` / ``dd:mm``). On a
    values-based (zoomed field) locator whose tick spacing is *finer* than one
    minute, adjacent ticks round to the same minute; the duplicate label gets
    blanked and renders as an empty-mathtext ``'$'`` glyph. Re-grid the ticks
    onto whole ``unit_deg`` steps (a whole minute of time for RA / arc for Dec)
    so no two labels collide. No-op for a spacing-based (all-sky) locator —
    astropy already picks sane round steps there.
    """
    fl = coord._formatter_locator
    if getattr(fl, 'spacing', None) is not None:
        return
    vals = getattr(fl, 'values', None)
    if vals is None or len(vals) < 2:
        return
    arr = np.unique(np.asarray(vals.to_value(u.deg), dtype=float))
    diffs = np.diff(arr)
    diffs = diffs[diffs > 1e-9]
    if not len(diffs) or diffs.min() >= unit_deg * 0.999:
        return  # already at/above one minute — nothing to snap
    step = unit_deg * max(1, int(np.ceil(diffs.min() / unit_deg)))
    lo, hi = arr.min(), arr.max()
    start = np.ceil(lo / step - 1e-6) * step
    new = np.arange(start, hi + step * 1e-6, step)
    if len(new) >= 1:
        coord.set_ticks(values=new * u.deg)


def _restyle_overlay_ticklabels(ax: Any, color: Any = None,
                                path_effects: Any = None,
                                size: Any = None) -> None:
    """Apply tick-label visual props to skyplothelper *overlay* tick labels.

    On all-sky elliptical frames (AIT/MOL) the exterior latitude labels are
    drawn by the hybrid lat-overlay (``add_overlay_ticks``) as plain ``Text``
    artists tagged ``_sph_overlay_ticklabel`` — astropy's
    ``coords[1].set_ticklabel(...)`` can't reach them (its own lat labels are
    hidden on that path), so ``format_ticklabels``' color / stroke / size
    would otherwise miss them while the native longitude labels got styled.
    These overlay artists are static (not regenerated per draw, unlike
    astropy's native ticklabels), so setting the props directly here is
    robust. No-op when there are no overlay labels.
    """
    for txt in ax.texts:
        if not getattr(txt, '_sph_overlay_ticklabel', False):
            continue
        if color is not None:
            txt.set_color(color)
        if path_effects is not None:
            txt.set_path_effects(path_effects)
        if size is not None:
            txt.set_fontsize(size)


class _NoOpCoord:
    """Stand-in for the axis excluded by ``format_ticklabels(which=...)``.

    Swallows every per-axis setter call (``set_major_formatter``,
    ``set_separator``, ``set_ticklabel``, ``set_axislabel``, ...) so that axis
    is left exactly as it was, while the formatting code stays a single
    straight-line sequence over both coords. Only the *mutating* calls route
    through this; read-only probes (auto-precision) keep using the real coord.
    """

    def __getattr__(self, _name: str) -> Any:
        return lambda *args, **kwargs: None


def format_ticklabels(ax: Any, style: str | None = 'publication',
                      lon_fmt: str | None = None, lat_fmt: str | None = None,
                      lon_sep: str | tuple[str, ...] | None = None,
                      lat_sep: str | tuple[str, ...] | None = None,
                      simplify: bool = True,
                      which: str = 'both',
                      fontsize: float | None = None, color: str | None = None,
                      stroke_lw: float | None = None,
                      stroke_color: str | None = None,
                      exclude_overlapping: bool = True,
                      frame_defaults: bool = True,
                      rotation: float | None = None,
                      lon_rotation: float | None = None,
                      lat_rotation: float | None = None,
                      decimal_places: int | None = None,
                      axis_labels: bool | dict[str, Any] = True,
                      # Legacy aliases
                      ra_fmt: str | None = None, dec_fmt: str | None = None,
                      ra_sep: str | tuple[str, ...] | None = None,
                      dec_sep: str | tuple[str, ...] | None = None,
                      **kwargs: Any) -> None:
    """
    Apply coordinated tick label formatting to both axes of a WCSAxes.

    Automatically detects coordinate frame (ICRS, Galactic, Ecliptic,
    Supergalactic) and applies appropriate defaults — HMS/DMS for equatorial,
    decimal degrees with ° symbol for Galactic/Ecliptic/Supergalactic.

    This formats **absolute** celestial coordinates (sexagesimal or decimal
    degrees). For a zoomed field where you want ticks as **offsets** from a
    reference — arcsec / mas / μas relative labels, the VLBI / HST / Chandra
    convention — use :func:`apply_offset_ticks` instead. That is the right
    tool even on a compact ``TAN`` field centered on a real sky position (a
    milliarcsecond frame around Sgr A*, say): it keeps the celestial WCS but
    labels the ticks as offsets, so you do **not** need a synthetic
    linear-offset WCS header (which is also what makes this function raise a
    cryptic ``"Invalid format: hh:mm:ss"`` — it can only format a celestial
    frame). :func:`offset_figure` builds such a field in one call.

    Parameters
    ----------
    ax : WCSAxes
        The plot axis to format.
    style : str or None
        Preset style name:

        **Equatorial (HMS/DMS) styles:**

        - ``'publication'`` / ``'pub'`` : mathtext superscript separators
          (rendered as hᵐˢ / °′″ via ``$^\\mathregular{h}$`` etc.), format
          ``hh:mm:ss.s`` / ``+dd:mm:ss``. Mathtext rather than the Unicode
          modifier letters, which are missing from some common fonts
          (e.g. Arial) and render as tofu boxes there; pass
          ``lon_sep=SEPARATORS['hms_unicode']`` to opt into those.
        - ``'letter'`` : Plain ASCII separators (h,m,s / d,m,s)
        - ``'casa'`` : Colon-separated (CASA/CARTA convention), 2 decimal places
        - ``'latex'`` : LaTeX superscript symbols
        - ``'compact'`` : Publication separators with the seconds dropped
          (minute-truncating ``hh:mm`` / ``dd:mm`` format, on every astropy
          version). On astropy >= 7 it additionally applies ``simplify=True``
          to collapse unchanged leading fields (e.g. constant hours).
        - ``'minimal'`` : Minimal labels — hours only for RA, degrees only
          for DEC (useful for all-sky plots)

        **Decimal degree styles (all frames):**

        - ``'decimal'`` / ``'deg'`` : Decimal degrees with ° symbol, both axes
        - ``'decimal_plain'`` : Decimal degrees, no degree symbol

        **All-sky styles:**

        - ``'allsky_hours'`` / ``'allsky_h'`` : Hours for lon axis, degrees
          for lat axis (equatorial all-sky)
        - ``'allsky_deg'`` / ``'allsky_d'`` : Decimal degrees both axes
          (Galactic/Ecliptic all-sky)

        **Offset / relative coordinate styles:**

        - ``'offset'`` : Auto-unit relative offsets from image center
          (delegates to ``apply_offset_ticks()``)
        - ``'offset_arcsec'`` : Relative offsets in arcseconds (Δα″, Δδ″)
        - ``'offset_arcmin'`` : Relative offsets in arcminutes (Δα′, Δδ′)
        - ``'offset_mas'`` : Relative offsets in milliarcseconds (Δα mas, Δδ mas)
        - ``'offset_uas'`` : Relative offsets in microarcseconds (Δα μas, Δδ μas)

        **Sub-arcsec & anchored-offset styles:**

        - ``'vlbi'`` : Sub-arcsecond precision sexagesimal
          (``hh:mm:ss.sssss`` / ``+dd:mm:ss.ssss``) with simplify=True
          so only the varying sub-fields are shown
        - ``'anchored_offset'`` / ``'anchored_offset_mas'`` : One anchor tick
          in full sexagesimal + mas offsets from it for the other ticks
          (a marked reference position with relative offsets around it; the
          VLBI convention is the canonical use). Uses
          ``apply_anchored_offset()`` internally.
        - ``'anchored_offset_uas'`` : Same but with μas offsets
        - ``'anchored_offset_compact'`` : Same with rotated labels for
          tight multi-panel layouts

        - ``None`` : Skip presets, use lon_fmt/lat_fmt/lon_sep/lat_sep directly

    lon_fmt, lat_fmt : str, optional
        Format strings for set_major_formatter (e.g. 'hh:mm:ss.s', 'd.ddd').
        Aliases ``ra_fmt``/``dec_fmt`` also accepted.
    lon_sep, lat_sep : str or tuple, optional
        Separator strings for set_separator. Can be a key from SEPARATORS
        dict (e.g. 'hms_full') or a tuple of separator strings.
        Aliases ``ra_sep``/``dec_sep`` also accepted.
    simplify : bool
        Suppress unchanged leading fields (e.g. hours when all ticks share
        the same hour). Default True.
    which : str
        Which axis to (re)format. ``'both'`` (default; also ``'all'``) styles
        both coordinates as before; otherwise only the named axis is touched
        and the other's existing labels, separator, rotation, and axis label
        are left exactly as they are. Case-insensitive, and accepts the same
        aliases as :func:`~skyplothelper.highlight_gridline` — longitude:
        ``'lon'`` / ``'longitude'`` / ``'ra'`` / ``'l'`` / ...; latitude:
        ``'lat'`` / ``'latitude'`` / ``'dec'`` / ``'b'`` / ... Lets you restyle
        one axis without re-specifying the other — e.g. recolor only the
        longitude labels, or drop the ° from just the latitude. Every other
        argument still applies, but only to the selected axis (the ``lon_*``
        args are ignored under ``which='lat'`` and vice versa).
    fontsize : float, optional
    color : str, optional
    stroke_lw : float, optional
        Stroke (outline) linewidth for readability on busy backgrounds.
    stroke_color : str, optional
        Stroke color (typically 'w' for white outline on dark backgrounds).
    exclude_overlapping : bool
        Let astropy hide overlapping labels. Default True.
    rotation : float, optional
        Rotation angle (degrees) for all tick labels.
    lon_rotation, lat_rotation : float, optional
        Per-axis rotation override.
    decimal_places : int, optional
        Override decimal places for decimal styles.
    axis_labels : bool or dict
        If True (default), apply standard axis labels based on frame.
        If False, suppress axis labels. If a dict, set custom labels
        as ``{'lon': 'l (°)', 'lat': 'b (°)'}``.
    frame_defaults : bool
        Auto-detect frame and apply appropriate units. Default True.
    **kwargs
        Additional kwargs passed to set_ticklabel().

    Examples
    --------
    >>> format_ticklabels(ax)  # auto-detect frame, publication style
    >>> format_ticklabels(ax, style='casa')  # CASA colon-separated
    >>> format_ticklabels(ax, style='compact', fontsize=9)
    >>> format_ticklabels(ax, style='vlbi')  # sub-arcsecond precision
    >>> format_ticklabels(ax, style='offset_mas')  # mas offset labels
    >>> format_ticklabels(ax, style='decimal', decimal_places=1)
    >>> format_ticklabels(ax, style=None, lon_fmt='hh:mm', lat_fmt='dd:mm',
    ...                   lon_sep='hms_letter', lat_sep='dms_letter')

    Notes
    -----
    **Astropy version compatibility:** The ``simplify`` parameter (which
    suppresses unchanged leading sexagesimal fields) requires astropy ≥7.0.
    On older versions, ``simplify`` is silently ignored — labels still
    render, just without field suppression. The ``'compact'`` style does not
    rely on this for dropping seconds — it uses a minute-truncating format on
    every version, and only layers ``simplify`` on top where available.

    **Missing labels on Aitoff:** Astropy's ``exclude_overlapping`` can
    drop longitude labels (commonly 300° on Galactic Aitoff) when their
    bounding boxes collide with nearby latitude labels on the frame
    boundary. This is projection-geometry-dependent — Aitoff compresses
    meridians more at the edges than Mollweide, pushing labels closer
    together. Workarounds: use ``exclude_overlapping=False``, reduce
    ``fontsize``, or use ``apply_boundary_labels()`` which places labels
    at frame boundary intersections instead.
    """
    # Handle legacy aliases
    if ra_fmt is not None and lon_fmt is None:
        lon_fmt = ra_fmt
    if dec_fmt is not None and lat_fmt is None:
        lat_fmt = dec_fmt
    if ra_sep is not None and lon_sep is None:
        lon_sep = ra_sep
    if dec_sep is not None and lat_sep is None:
        lat_sep = dec_sep

    # Resolve separator names to tuples
    if isinstance(lon_sep, str) and lon_sep in SEPARATORS:
        lon_sep = SEPARATORS[lon_sep]
    if isinstance(lat_sep, str) and lat_sep in SEPARATORS:
        lat_sep = SEPARATORS[lat_sep]

    # Per-axis application gate. ``which`` restricts every mutation to the
    # selected coord; the excluded one gets a no-op stand-in so its existing
    # labels are left untouched. ``do_lon`` / ``do_lat`` gate the few
    # non-coord operations (overlay-label restyle, manual all-sky axis labels).
    # Case-insensitive, with the same lon/lat alias vocabulary as
    # skyplothelper.grid (so 'ra'/'dec'/'longitude'/'l'/'b'/... all work).
    _which = str(which).strip().lower()
    _lon_names = ('lon', 'longitude', 'meridian', 'ra', 'l',
                  'glon', 'slon', 'elon', '0', 'x')
    _lat_names = ('lat', 'latitude', 'parallel', 'dec', 'b',
                  'glat', 'slat', 'elat', '1', 'y')
    if _which in ('both', 'all'):
        do_lon = do_lat = True
    elif _which in _lon_names:
        do_lon, do_lat = True, False
    elif _which in _lat_names:
        do_lon, do_lat = False, True
    else:
        raise ValueError(
            f"format_ticklabels(): which must be 'both', 'lon', or 'lat' "
            f"(or an alias like 'ra'/'dec'/'longitude'/'latitude'), "
            f"got {which!r}.")
    lon_c = ax.coords[0] if do_lon else _NoOpCoord()
    lat_c = ax.coords[1] if do_lat else _NoOpCoord()

    # Detect frame for defaults
    frame = _detect_frame(ax) if frame_defaults else 'icrs'
    is_equatorial = frame in ('icrs', 'fk5', 'fk4')
    is_galactic_like = frame in ('galactic', 'supergalactic', 'ecliptic',
                                 'geocentrictrueecliptic',
                                 'heliocentrictrueecliptic')

    # Apply frame-based default units (before style, since styles may override)
    if frame_defaults:
        lon_unit, lat_unit = _FRAME_DEFAULT_UNITS.get(frame, (u.deg, u.deg))
        if is_galactic_like:
            lon_c.set_format_unit(lon_unit, decimal=True)
            lat_c.set_format_unit(lat_unit, decimal=True)
        else:
            lon_c.set_format_unit(lon_unit)
            lat_c.set_format_unit(lat_unit)

    # --- Apply style presets ---
    if style is not None:
        style_lower = style.lower().replace('-', '_').replace(' ', '_')

        # Equatorial sexagesimal styles
        if style_lower in ('publication', 'pub'):
            if is_equatorial:
                lon_sep = lon_sep or SEPARATORS['hms_full']
                lat_sep = lat_sep or SEPARATORS['dms_full']
                auto_lon, auto_lat = _auto_sexagesimal_format(ax)
                lon_fmt = lon_fmt or auto_lon
                lat_fmt = lat_fmt or auto_lat
            elif is_galactic_like:
                # Galactic: decimal degrees with ° symbol
                lat_sep = lat_sep or SEPARATORS['deg_symbol']

        elif style_lower == 'letter':
            if is_equatorial:
                lon_sep = lon_sep or SEPARATORS['hms_letter']
                lat_sep = lat_sep or SEPARATORS['dms_letter']
                auto_lon, auto_lat = _auto_sexagesimal_format(ax)
                lon_fmt = lon_fmt or auto_lon
                lat_fmt = lat_fmt or auto_lat

        elif style_lower == 'casa':
            if is_equatorial:
                lon_sep = lon_sep or SEPARATORS['hms_colon']
                lat_sep = lat_sep or SEPARATORS['dms_colon']
                auto_lon, auto_lat = _auto_sexagesimal_format(ax)
                lon_fmt = lon_fmt or auto_lon
                lat_fmt = lat_fmt or auto_lat

        elif style_lower == 'latex':
            if is_equatorial:
                lon_sep = lon_sep or SEPARATORS['hms_latex']
                lat_sep = lat_sep or SEPARATORS['dms_latex']
                auto_lon, auto_lat = _auto_sexagesimal_format(ax)
                lon_fmt = lon_fmt or auto_lon
                lat_fmt = lat_fmt or auto_lat

        elif style_lower == 'compact':
            # "Publication with the seconds dropped." A minute-truncating
            # format does the actual seconds-drop — on EVERY astropy version.
            # (simplify alone does NOT drop seconds: it only suppresses
            # redundant *leading* fields, and on astropy 6 it's ignored
            # entirely — so relying on it left the seconds in, with the
            # default format even adding decimal places on astropy 7.)
            if is_equatorial:
                lon_sep = lon_sep or SEPARATORS['hms_full']
                lat_sep = lat_sep or SEPARATORS['dms_full']
                lon_fmt = lon_fmt or 'hh:mm'
                lat_fmt = lat_fmt or 'dd:mm'
            else:
                lon_fmt = lon_fmt or 'dd:mm'
                lat_fmt = lat_fmt or 'dd:mm'
            # On astropy >= 7, simplify additionally collapses redundant
            # leading fields (e.g. the constant hours) on top of the
            # truncation; ignored on < 7 (the truncation already suffices).
            if _ASTROPY_GE_7:
                simplify = True
            # If a zoomed field frame's auto ticks are finer than one minute,
            # thin them to a whole-minute grid so the minute-truncating format
            # doesn't emit duplicate (blanked → literal '$') labels.
            _lon_unit = 0.25 if is_equatorial else 1.0 / 60.0  # RA: 1 time-min
            _snap_compact_minute_ticks(lon_c, _lon_unit)
            _snap_compact_minute_ticks(lat_c, 1.0 / 60.0)  # Dec: 1 arcmin

        elif style_lower == 'minimal':
            if is_equatorial:
                lon_fmt = lon_fmt or 'hh'
                lat_fmt = lat_fmt or 'dd'
                lon_sep = lon_sep or SEPARATORS['hms_full']
                lat_sep = lat_sep or SEPARATORS['dms_full']

        elif style_lower in ('allsky_hours', 'allsky_h'):
            if is_equatorial:
                lon_fmt = lon_fmt or 'hh'
                lat_fmt = lat_fmt or 'dd'
                lon_sep = lon_sep or SEPARATORS['hms_full']
                lat_sep = lat_sep or SEPARATORS['dms_full']

        elif style_lower in ('allsky_deg', 'allsky_d'):
            lon_c.set_format_unit(u.deg, decimal=True)
            lat_c.set_format_unit(u.deg, decimal=True)

        elif style_lower in ('decimal', 'deg'):
            lon_c.set_format_unit(u.deg, decimal=True)
            lat_c.set_format_unit(u.deg, decimal=True)
            # decimal_places=None: derive precision from the tick spacing for
            # values-based axes (field frames), where astropy's decimal auto-
            # precision has no spacing to read and would render ~10 dp. Spacing-
            # based axes (all-sky) return None here and keep astropy's auto.
            lon_dp = (decimal_places if decimal_places is not None
                      else _auto_decimal_places(ax.coords[0]))
            lat_dp = (decimal_places if decimal_places is not None
                      else _auto_decimal_places(ax.coords[1]))
            if lon_dp is not None:
                lon_fmt = lon_fmt or _decimal_format_str(lon_dp, signed=False)
            if lat_dp is not None:
                lat_fmt = lat_fmt or _decimal_format_str(lat_dp, signed=True)

        elif style_lower == 'decimal_plain':
            # ``show_decimal_unit=False`` drops the ° symbol astropy appends
            # in decimal mode — this is what distinguishes ``decimal_plain``
            # from ``decimal`` (which keeps the °). An explicit format string
            # sets precision without re-adding the ° (verified), so the same
            # spacing-derived precision applies.
            lon_c.set_format_unit(u.deg, decimal=True,
                                  show_decimal_unit=False)
            lat_c.set_format_unit(u.deg, decimal=True,
                                  show_decimal_unit=False)
            lon_dp = (decimal_places if decimal_places is not None
                      else _auto_decimal_places(ax.coords[0]))
            lat_dp = (decimal_places if decimal_places is not None
                      else _auto_decimal_places(ax.coords[1]))
            if lon_dp is not None:
                lon_fmt = lon_fmt or _decimal_format_str(lon_dp, signed=False)
            if lat_dp is not None:
                lat_fmt = lat_fmt or _decimal_format_str(lat_dp, signed=True)

        elif style_lower == 'offset':
            apply_offset_ticks(ax, unit='auto', fontsize=fontsize,
                               color=color, stroke_lw=stroke_lw,
                               stroke_color=stroke_color,
                               axis_labels=axis_labels)
            return

        elif style_lower == 'offset_arcsec':
            apply_offset_ticks(ax, unit='arcsec', fontsize=fontsize,
                               color=color, stroke_lw=stroke_lw,
                               stroke_color=stroke_color,
                               axis_labels=axis_labels)
            return

        elif style_lower == 'offset_arcmin':
            apply_offset_ticks(ax, unit='arcmin', fontsize=fontsize,
                               color=color, stroke_lw=stroke_lw,
                               stroke_color=stroke_color,
                               axis_labels=axis_labels)
            return

        elif style_lower == 'offset_mas':
            apply_offset_ticks(ax, unit='mas', fontsize=fontsize,
                               color=color, stroke_lw=stroke_lw,
                               stroke_color=stroke_color,
                               axis_labels=axis_labels)
            return

        elif style_lower == 'offset_uas':
            apply_offset_ticks(ax, unit='uas', fontsize=fontsize,
                               color=color, stroke_lw=stroke_lw,
                               stroke_color=stroke_color,
                               axis_labels=axis_labels)
            return

        elif style_lower == 'vlbi':
            if is_equatorial:
                lon_sep = lon_sep or SEPARATORS['hms_full']
                lat_sep = lat_sep or SEPARATORS['dms_full']
                lon_fmt = lon_fmt or 'hh:mm:ss.sssss'
                lat_fmt = lat_fmt or '+dd:mm:ss.ssss'
                simplify = True

        elif style_lower in ('anchored_offset', 'anchored_offset_mas',
                              'anchored_offset_uas', 'anchored_offset_compact'):
            # Delegate to apply_anchored_offset — this sets up the formatter
            # and registers a draw callback for label replacement.
            # First apply high-precision sexagesimal as base format.
            if is_equatorial:
                lon_sep = lon_sep or SEPARATORS['hms_full']
                lat_sep = lat_sep or SEPARATORS['dms_full']
                lon_fmt = lon_fmt or 'hh:mm:ss.sssss'
                lat_fmt = lat_fmt or '+dd:mm:ss.sssss'
                simplify = True
            anchored_unit = 'uas' if style_lower == 'anchored_offset_uas' else 'mas'
            anchored_compact = style_lower == 'anchored_offset_compact'
            # Store for deferred application (after main formatting is done)
            ax._sph_anchored_offset_pending = (anchored_unit, anchored_compact)

        else:
            warnings.warn(f"Unknown style '{style}', ignoring. Available: "
                          "publication, letter, casa, latex, compact, minimal, "
                          "allsky_hours, allsky_deg, decimal, decimal_plain, "
                          "offset, offset_arcsec, offset_arcmin, offset_mas, "
                          "offset_uas, vlbi, anchored_offset, anchored_offset_mas, "
                          "anchored_offset_uas, anchored_offset_compact")

    # Apply format strings (strip leading '+' on astropy < 7.0)
    if lon_fmt is not None:
        lon_c.set_major_formatter(_compat_format(lon_fmt))
    if lat_fmt is not None:
        lat_c.set_major_formatter(_compat_format(lat_fmt))

    # ``decimal_plain`` drops the degree symbol via ``show_decimal_unit=False``.
    # On astropy >= 8 ``set_major_formatter`` resets that flag back to True, so
    # re-assert it after the formatter (idempotent on older astropy, which kept
    # the flag through the formatter). Precision from the format string above is
    # preserved by this call.
    if style is not None and style_lower == 'decimal_plain':
        lon_c.set_format_unit(u.deg, decimal=True, show_decimal_unit=False)
        lat_c.set_format_unit(u.deg, decimal=True, show_decimal_unit=False)

    # Apply separators
    if lon_sep is not None:
        lon_c.set_separator(lon_sep)
    if lat_sep is not None:
        lat_c.set_separator(lat_sep)

    # Build ticklabel kwargs
    ticklabel_kwargs: dict[str, Any] = {'simplify': simplify,
                                        'exclude_overlapping': exclude_overlapping}
    if fontsize is not None:
        ticklabel_kwargs['size'] = fontsize
    if color is not None:
        ticklabel_kwargs['color'] = color

    # Stroke effects
    pe = _stroke_path_effects(stroke_color, stroke_lw)
    if pe is not None:
        ticklabel_kwargs['path_effects'] = pe

    ticklabel_kwargs.update(kwargs)

    # Per-axis rotation
    lon_kw = dict(ticklabel_kwargs)
    lat_kw = dict(ticklabel_kwargs)
    if rotation is not None:
        lon_kw['rotation'] = rotation
        lat_kw['rotation'] = rotation
    if lon_rotation is not None:
        lon_kw['rotation'] = lon_rotation
    if lat_rotation is not None:
        lat_kw['rotation'] = lat_rotation

    # On elliptical frames (AIT, MOL) the pole is a singular point on
    # the curved boundary where astropy may place duplicate ticks at
    # ±90°. With simplify=True that pair gets reduced to a bare '00″'
    # because the second label is identical to the first. Disable
    # simplify on the latitude coord for these frames so pole labels
    # either render in full or are naturally suppressed by astropy's
    # overlap detection.
    try:
        _is_elliptical = isinstance(ax.coords.frame, EllipticalFrame)
    except Exception:
        _is_elliptical = False
    if _is_elliptical:
        lat_kw['simplify'] = False

    lon_c.set_ticklabel(**_safe_ticklabel_kwargs(lon_kw))
    lat_c.set_ticklabel(**_safe_ticklabel_kwargs(lat_kw))
    # Reach the hybrid lat-overlay's exterior labels (AIT/MOL all-sky), which
    # the coords API can't touch — apply the same color / stroke / size so the
    # latitude labels match the natively-styled longitude ones. Skipped when
    # ``which='lon'`` (those overlay artists are the latitude labels).
    if do_lat:
        _restyle_overlay_ticklabels(
            ax, color=ticklabel_kwargs.get('color'),
            path_effects=ticklabel_kwargs.get('path_effects'),
            size=ticklabel_kwargs.get('size'))
    # Detect if we have a custom all-sky frame (with chv spines) and pin
    # tick labels / axis labels to the correct spines to avoid overlap.
    _has_custom_frame = isinstance(
        getattr(ax, 'frame', None), _AllSkyCustomFrame
    ) if '_AllSkyCustomFrame' in dir() else False

    # Also check via frame_class on the coords
    if not _has_custom_frame:
        try:
            frame_obj = ax.coords.frame
            _has_custom_frame = (hasattr(frame_obj, 'spine_names') and
                                 'h' in frame_obj.spine_names and
                                 'v' in frame_obj.spine_names and
                                 not isinstance(frame_obj, EllipticalFrame))
        except Exception:
            pass

    if _has_custom_frame:
        # RA/lon: tick labels along equator (h-spine) + boundary (c)
        lon_c.set_ticks_position('ch')
        lon_c.set_ticklabel_position('h')
        # Dec/lat: tick labels along boundary (c-spine) only
        lat_c.set_ticks_position('cv')
        lat_c.set_ticklabel_position('c')

    # Axis labels get the same color / stroke the tick labels received, so a
    # recolored or stroked frame reads as one piece (parity with
    # apply_offset_ticks, which already styles its offset axis labels — this
    # was the gap that left absolute-frame titles unstroked). `pe` is the same
    # tick-label stroke computed above; both keys are omitted when unset, so
    # this is a no-op for an unstyled call.
    _axislabel_style: dict[str, Any] = {}
    if color is not None:
        _axislabel_style['color'] = color
    if pe is not None:
        _axislabel_style['path_effects'] = pe

    # Axis labels
    if axis_labels is True:
        _FRAME_AXIS_LABELS = {
            'icrs':            ('RA (J2000)', 'Dec (J2000)'),
            'fk5':             ('RA (FK5)', 'Dec (FK5)'),
            'fk4':             ('RA (FK4)', 'Dec (FK4)'),
            'galactic':        ('Galactic Longitude (°)', 'Galactic Latitude (°)'),
            'supergalactic':   ('Supergalactic Longitude (°)',
                                'Supergalactic Latitude (°)'),
            'ecliptic':        ('Ecliptic Longitude (°)',
                                'Ecliptic Latitude (°)'),
            'geocentrictrueecliptic': ('Ecliptic Longitude (°)',
                                       'Ecliptic Latitude (°)'),
            'heliocentrictrueecliptic': ('Ecliptic Longitude (°)',
                                         'Ecliptic Latitude (°)'),
        }
        lon_label, lat_label = _FRAME_AXIS_LABELS.get(frame,
                                                       ('Longitude', 'Latitude'))

        if _has_custom_frame:
            # For custom all-sky frames, suppress astropy's auto labels
            # (they overlap at cusps) and place manually. Each manual label is
            # tagged with its axis kind so ``which`` can rebuild one without
            # disturbing the other's existing label.
            lon_c.set_axislabel('')
            lat_c.set_axislabel('')
            kept = []
            for art in getattr(ax, '_sph_manual_axis_labels', []):
                k = getattr(art, '_sph_axis_kind', None)
                if (k == 'lon' and do_lon) or (k == 'lat' and do_lat) or k is None:
                    art.remove()
                else:
                    kept.append(art)
            if do_lon:
                # RA label below equator center
                t1 = ax.text(0.5, -0.02, lon_label, transform=ax.transAxes,
                        ha='center', va='top',
                        fontsize=rcParams.get('axes.labelsize', 12),
                        **_axislabel_style)
                t1._sph_axis_kind = 'lon'
                kept.append(t1)
            if do_lat:
                # Dec label at left side, rotated
                t2 = ax.text(-0.02, 0.5, lat_label, transform=ax.transAxes,
                        ha='right', va='center', rotation=90,
                        fontsize=rcParams.get('axes.labelsize', 12),
                        **_axislabel_style)
                t2._sph_axis_kind = 'lat'
                kept.append(t2)
            ax._sph_manual_axis_labels = kept
        else:
            lon_c.set_axislabel(lon_label, **_axislabel_style)
            lat_c.set_axislabel(lat_label, **_axislabel_style)
    elif axis_labels is False:
        lon_c.set_axislabel('')
        lat_c.set_axislabel('')
        # Also remove manually placed labels from prior calls (only the
        # axis/axes ``which`` selects; untagged legacy ones always go).
        kept = []
        for art in getattr(ax, '_sph_manual_axis_labels', []):
            k = getattr(art, '_sph_axis_kind', None)
            if (k == 'lon' and do_lon) or (k == 'lat' and do_lat) or k is None:
                art.remove()
            else:
                kept.append(art)
        ax._sph_manual_axis_labels = kept
    elif isinstance(axis_labels, dict):
        if 'lon' in axis_labels:
            lon_c.set_axislabel(axis_labels['lon'], **_axislabel_style)
        if 'lat' in axis_labels:
            lat_c.set_axislabel(axis_labels['lat'], **_axislabel_style)

    # Deferred VLBI hybrid application (after all other formatting)
    anchored_pending = getattr(ax, '_sph_anchored_offset_pending', None)
    if anchored_pending is not None:
        del ax._sph_anchored_offset_pending
        anchored_unit, anchored_compact = anchored_pending
        apply_anchored_offset(ax, unit=anchored_unit, sep='super',
                          fontsize=fontsize, color=color,
                          stroke_lw=stroke_lw, stroke_color=stroke_color,
                          axis_labels=axis_labels, compact=anchored_compact)




# ===== Auxiliary helpers (offset unit picker, rotation patch) =====

def _auto_offset_unit(fov_deg: float) -> tuple[float, str]:
    """Choose appropriate offset unit and scale from field of view in degrees."""
    fov_arcsec = fov_deg * 3600
    if fov_arcsec > 7200:    # > 2 deg → deg
        return 1/3600., 'deg'
    elif fov_arcsec > 600:   # > 10 arcmin → arcmin
        return 1/60., 'arcmin'
    elif fov_arcsec > 2:     # > 2 arcsec → arcsec
        return 1., 'arcsec'
    elif fov_arcsec > 0.002: # > 2 mas → mas
        return 1e3, 'mas'
    else:                    # sub-mas → μas
        return 1e6, 'μas'

def _patch_rotated_label_centering(lat_coord: Any) -> None:
    """
    Fix the vertical centering of 90°-rotated Dec tick labels in WCSAxes.

    RectangularFrame hardcodes ``dy = -text_size * 0.5`` assuming unrotated
    text.  After rotation the offset no longer centers correctly; this
    monkey-patches ``_set_xy_alignments`` to compensate.
    """
    tls = coord_ticklabels(lat_coord)   # the TickLabels artist (astropy-7 safe)
    original = tls._set_xy_alignments

    def patched(renderer: Any) -> None:
        original(renderer)
        for axis in tls.xy:
            for i in tls.xy[axis]:
                label_text = tls.text[axis][i]
                if label_text == '':
                    continue
                tls.set_text(label_text)
                tls.set_rotation(90)
                bb = tls.get_window_extent(renderer)
                x, y = tls.xy[axis][i]
                tls.xy[axis][i] = (x, y - bb.height * 0.3)

    tls._set_xy_alignments = patched


# ============================================================
# Curve-following longitude tick labels
# ============================================================


_ZENITHAL_FITS_CODES = frozenset(
    ('SIN', 'TAN', 'STG', 'ARC', 'ZEA', 'AZP', 'SZP', 'AIR'))


def _ax_is_zenithal(ax: Any) -> bool:
    """True if ``ax``'s WCS uses a zenithal projection (back hemisphere
    is meaningful and worth filtering)."""
    try:
        ctype1 = str(ax.wcs.wcs.ctype[0])
    except Exception:
        return False
    return ctype1.split('-')[-1].upper() in _ZENITHAL_FITS_CODES


def _orthographic_visibility(lons: npt.ArrayLike, lats: npt.ArrayLike,
                             lon_0: float, lat_0: float) -> np.ndarray:
    """Boolean mask: True where (lon, lat) is in the *front* hemisphere
    of an orthographic globe centered at (lon_0, lat_0).

    Inlined here (rather than importing from
    ``skyplothelper.globe.boundaries``) so the helper has no
    circular-import surface and works for any zenithal WCS."""
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    lon_r = np.radians(lons - lon_0)
    lat_r = np.radians(lats)
    lat0_r = np.radians(lat_0)
    cos_c = (np.sin(lat0_r) * np.sin(lat_r)
             + np.cos(lat0_r) * np.cos(lat_r) * np.cos(lon_r))
    return cos_c >= 0


_HOUR_SEP = {'super': r'$^\mathregular{h}$', 'unicode': '\u02b0',
             'plain': 'h', 'letter': 'h', 'none': ''}


def _resolve_hour_sep(sep: str) -> str:
    """Resolve a longitude-hour separator spec to its suffix string.

    Accepts a preset name ('super' = mathtext superscript, the
    default; 'unicode'; 'plain'/'letter'; 'none') or any literal
    suffix string.
    """
    return _HOUR_SEP.get(sep, sep)


def _format_lon_label(lon_deg: float, frame: str,
                      fmt: str | Callable[[float], str],
                      sep: str = 'super') -> str:
    """Render a tick label for a longitude value.

    ``fmt`` may be a callable, ``'auto'`` (hours for ICRS, degrees for
    galactic / ecliptic / supergalactic), ``'hour'``, or ``'deg'``.
    """
    if callable(fmt):
        return fmt(lon_deg)
    if fmt == 'auto':
        fmt = 'hour' if frame == 'icrs' else 'deg'
    if fmt == 'hour':
        suffix = _resolve_hour_sep(sep)
        h = (float(lon_deg) / 15.0) % 24
        # Use integer hours if the value rounds clean, else 1 decimal.
        if abs(h - round(h)) < 1e-3:
            return f"{int(round(h)) % 24:d}{suffix}"
        return f"{h:.1f}{suffix}"
    if fmt == 'deg':
        # Absolute longitude convention is [0, 360) (galactic, ecliptic,
        # supergalactic) — don't fold >180° to a signed value (l=182° would
        # read as -178°). Mirrors coord_overlay._format_tick_label.
        d = float(lon_deg) % 360.0
        return f"{int(round(d)) % 360:d}°"
    raise ValueError(f"Unknown lon-tick format {fmt!r}")


def add_curved_lon_ticks(ax: Any, tick_lat: float = 0.,
                         lon_spacing: float = 30.,
                         lon_ticks: npt.ArrayLike | None = None,
                         hide_back: bool | None = None, eps_deg: float = 0.5,
                         offset_points: float = 0.,
                         fontsize: float | None = None,
                         color: Any = None,
                         fmt: str | Callable[[float], str] = 'auto',
                         sep: str = 'super',
                         frame: str | None = None,
                         suppress_default: bool = True,
                         show_ticks: bool = True,
                         tick_length_points: float = 5.,
                         tick_color: Any = None, tick_lw: float = 1.0,
                         **text_kwargs: Any) -> list[Any]:
    """Place longitude tick labels along a chosen latitude curve.

    For each tick longitude, the label is anchored at the projected
    pixel position of (lon, ``tick_lat``) and rotated to align with
    the local meridian's tangent direction (so labels read "naturally"
    along the curve rather than as floating horizontal text).

    Designed for SIN-projection globes — where meridians fan out
    visibly around the equator — but works on any WCSAxes whose
    longitude axis traces a curve in pixel space (TAN, STG, ARC, ZEA,
    AZP, SZP, AIR, plus the all-sky pseudocylindrical / pseudoconic
    frames). For zenithal projections the back hemisphere is filtered
    automatically (``hide_back=True`` by default for those, ``False``
    elsewhere).

    Parameters
    ----------
    ax : WCSAxes
        The axes to decorate. Must expose ``ax.wcs`` with a celestial
        WCS (``ctype = 'RA---<proj>'`` / ``'DEC--<proj>'`` or one of
        the supported alternative frames).
    tick_lat : float, optional
        Latitude in degrees along which to place the labels. Default
        ``0`` (the equator). For zenithal globes, choose a latitude
        within the visible hemisphere; otherwise no ticks are drawn.
    lon_spacing : float, optional
        Spacing between auto-generated tick longitudes in degrees.
        Ignored if ``lon_ticks`` is supplied. Default ``30``.
    lon_ticks : array-like, optional
        Explicit list of tick longitudes in degrees. Overrides
        ``lon_spacing``.
    hide_back : bool or None, optional
        If True, skip ticks whose meridian doesn't intersect the
        front hemisphere at ``tick_lat``. ``None`` (default)
        auto-selects: ``True`` for zenithal projections,
        ``False`` for everything else.
    eps_deg : float, optional
        Latitude offset (degrees) used for the numerical tangent
        derivative. Default ``0.5``; smaller values give more
        accurate rotations but risk nearly-degenerate WCS round-off.
    offset_points : float, optional
        Offset of the label *along the meridian tangent direction*
        (i.e. perpendicular to the lat curve), in points. Default
        ``0`` (label sits on the curve). Positive values push the
        label outward away from the projection center — the same
        side the tick mark extends — keeping the label visually in
        line with its tick (above/below the tick for a roughly-
        horizontal lat curve). Negative values pull the label in
        the opposite direction. Matches the placement convention
        used by :func:`~skyplothelper.coord_overlay.add_overlay_ticks`
        and the ``make_wcs_frame`` auto-trigger.
    fontsize : float or None, optional
        Font size; ``None`` uses ``rcParams['xtick.labelsize']``.
    color : matplotlib color or None, optional
        Text color; ``None`` uses ``rcParams['xtick.color']``.
    fmt : {'auto', 'hour', 'deg'} or callable, optional
        Label format. ``'auto'`` picks hours for ICRS, degrees
        elsewhere. A callable receives the lon value in degrees and
        returns the label string.
    sep : str, optional
        Hour-suffix style for hour labels: ``'super'`` (mathtext
        superscript, the default — renders in any font), ``'unicode'``
        (``ʰ``), ``'plain'`` / ``'letter'`` (``h``), ``'none'``, or any
        literal suffix string. Ignored for degree labels.
    frame : str or None, optional
        Coordinate frame name (``'icrs'``, ``'galactic'``, ...).
        ``None`` reads from ``ax._sph_frame`` if present, else falls
        back to ``'icrs'``.
    suppress_default : bool, optional
        If True (default), hide the existing default longitude labels so the
        curved labels are the only ones drawn — BOTH astropy's native ticks
        (via ``ax.coords[0].set_tick*_visible(False)``) AND the frame's auto
        in-frame overlay labels (the ``_sph_auto_overlay`` 'lon' set that
        ``make_globe_frame`` / all-sky frames draw, removed like
        :func:`add_overlay_ticks` does). Without the latter, the curved labels
        would double up on top of the frame's defaults. Set False to keep the
        existing labels too.
    show_ticks : bool, optional
        Draw a short tick mark at each label position, oriented along
        the local meridian (perpendicular to the constant-lat curve).
        Default ``True``.
    tick_length_points : float, optional
        Total length of each tick mark in points (the line is centered
        on the curve, so it extends ``tick_length_points/2`` to each
        side). Default ``5``.
    tick_color : matplotlib color or None, optional
        Color for the tick marks. ``None`` (default) reuses ``color``.
    tick_lw : float, optional
        Line width for tick marks. Default ``1.0``.
    **text_kwargs
        Additional kwargs passed to ``ax.text`` for each label
        (``fontweight``, ``path_effects``, ``alpha``, ...).

    Returns
    -------
    artists : list of matplotlib.text.Text
        The text artists drawn, in tick-longitude order.

    Notes
    -----
    The grid lines drawn by ``ax.coords.grid()`` are not affected.
    Only the tick labels (and their default mark glyphs) are
    suppressed by ``suppress_default=True``.
    """
    wcs = ax.wcs

    def _suppress_lon_defaults() -> None:
        # Hide BOTH astropy's native ticks AND the frame's auto in-frame
        # overlay 'lon' labels (the ``_sph_auto_overlay`` set that
        # ``make_*_frame`` draws). ``set_tick*_visible`` only reaches the
        # native ticks — it can't touch the overlay Text artists — so without
        # the overlay removal this function's labels would double up on top of
        # the frame's defaults on a ``make_globe_frame`` / all-sky frame.
        # Mirrors :func:`add_overlay_ticks`'s ``_remove_auto_overlay_ticks``
        # for the kind it draws ('lon').
        from .coord_overlay import _remove_auto_overlay_ticks
        ax.coords[0].set_ticks_visible(False)
        ax.coords[0].set_ticklabel_visible(False)
        _remove_auto_overlay_ticks(ax, {'lon'})

    if frame is None:
        # Try ax._sph_frame (set by cartopy backend), else detect from
        # the WCS CTYPE prefix.
        frame = getattr(ax, '_sph_frame', None)
        if frame is None:
            try:
                ctype1 = str(wcs.wcs.ctype[0]).upper()
            except Exception:
                ctype1 = ''
            if ctype1.startswith('GLON'):
                frame = 'galactic'
            elif ctype1.startswith('SLON') or ctype1.startswith('SGLON'):
                frame = 'supergalactic'
            elif ctype1.startswith('ELON'):
                frame = 'ecliptic'
            elif ctype1.startswith('HLON'):
                frame = 'helioecliptic'
            else:
                frame = 'icrs'
    frame = frame.lower() if isinstance(frame, str) else 'icrs'

    if hide_back is None:
        hide_back = _ax_is_zenithal(ax)

    lon_center = float(wcs.wcs.crval[0])
    lat_center = float(wcs.wcs.crval[1])

    # Tick longitudes
    if lon_ticks is None:
        lon_ticks = np.arange(-180., 180., float(lon_spacing))
    lons = np.asarray(lon_ticks, dtype=float)
    # Wrap into a single 360° band centered on lon_center
    lons = ((lons - lon_center + 180.0) % 360.0) - 180.0 + lon_center

    if hide_back:
        vis = _orthographic_visibility(lons, tick_lat, lon_center, lat_center)
        lons = lons[vis]

    if len(lons) == 0:
        if suppress_default:
            _suppress_lon_defaults()
        return []

    # Anchor pixel coords at (lon, tick_lat) and tangent direction via
    # finite-difference along the meridian (lat ± eps_deg).
    lats0 = np.full_like(lons, float(tick_lat))
    lats_p = np.full_like(lons, float(tick_lat) + float(eps_deg))
    lats_m = np.full_like(lons, float(tick_lat) - float(eps_deg))
    x0, y0 = wcs.world_to_pixel_values(lons, lats0)
    x_p, y_p = wcs.world_to_pixel_values(lons, lats_p)
    x_m, y_m = wcs.world_to_pixel_values(lons, lats_m)

    valid = (np.isfinite(x0) & np.isfinite(y0) &
             np.isfinite(x_p) & np.isfinite(y_p) &
             np.isfinite(x_m) & np.isfinite(y_m))
    lons = lons[valid]
    x0, y0 = x0[valid], y0[valid]
    x_p, y_p = x_p[valid], y_p[valid]
    x_m, y_m = x_m[valid], y_m[valid]
    if len(lons) == 0:
        if suppress_default:
            _suppress_lon_defaults()
        return []

    # Meridian-tangent direction in pixel space
    dx_m = x_p - x_m
    dy_m = y_p - y_m
    # Tick label runs along the constant-lat curve (perpendicular to
    # the meridian) — rotate by -90 from the meridian direction.
    rotations_deg = np.degrees(np.arctan2(dy_m, dx_m)) - 90.0
    # Always-readable: keep within (-90, 90]
    rotations_deg = ((rotations_deg + 90.0) % 180.0) - 90.0

    if suppress_default:
        _suppress_lon_defaults()

    # Label styling defaults — honor an auto-fontsize value cached on
    # the axes by make_wcs_frame / make_globe_frame's auto_fontsize
    # hook, so add_curved_lon_ticks's Text overlays pick up the same
    # sizing the frame builder already chose. Same precedence model as
    # add_overlay_ticks: explicit ``fontsize=`` wins; otherwise cache;
    # otherwise rcParams default.
    if fontsize is None:
        fontsize = getattr(ax, '_sph_auto_label_fontsize', None)
    if fontsize is None:
        fontsize = rcParams.get('xtick.labelsize', 10)
    if color is None:
        color = rcParams.get('xtick.color', '0.2')

    # Perpendicular offset: ``offset_points`` along the outward normal
    # **of the lat curve** at each tick — i.e. *along the meridian
    # tangent*, in the same direction the tick mark extends, away from
    # the projection center. This keeps the label visually in line
    # with its tick (above or below for a roughly-horizontal lat curve)
    # matching the convention :func:`add_overlay_ticks` and the
    # ``make_wcs_frame`` auto-trigger use, rather than perpendicular
    # to the meridian (which sat the label to the *side* of the tick).
    if offset_points != 0.0:
        norm = np.hypot(dx_m, dy_m)
        norm[norm == 0] = 1.0
        # Perpendicular-to-the-lat-curve = along the meridian-tangent
        # direction (the gridline direction at each tick).
        nx = dx_m / norm
        ny = dy_m / norm
        # Pick a single global outward direction for the whole curve
        # instead of a per-tick sign. Per-tick worked for curves well
        # outside the projection center but flipped individual ticks
        # when the curve passes through or near the center —
        # ``(x0 - cx, y0 - cy) ≈ 0`` and floating-point noise in
        # ``dx_m`` decides the sign. Global average tangent
        # (rotated to point outward from the curve's centroid
        # relative to the projection center) is direction-stable
        # across the whole curve.
        cx = 0.5 * (ax.get_xlim()[0] + ax.get_xlim()[1])
        cy = 0.5 * (ax.get_ylim()[0] + ax.get_ylim()[1])
        valid = np.isfinite(nx) & np.isfinite(ny)
        if valid.any():
            avg_nx = float(np.mean(nx[valid]))
            avg_ny = float(np.mean(ny[valid]))
            mean_x = float(np.mean(x0[valid])) - cx
            mean_y = float(np.mean(y0[valid])) - cy
            global_sign = np.sign(avg_nx * mean_x + avg_ny * mean_y)
            if global_sign == 0:
                global_sign = 1.0
        else:
            global_sign = 1.0
        nx = nx * global_sign
        ny = ny * global_sign
    else:
        nx = ny = np.zeros_like(lons)

    # Tick marks (short line segments along the meridian-tangent
    # direction, centered on the curve). Implemented in display-space
    # so the length stays a fixed pixel size regardless of axes data
    # range — convert each anchor to display coords, add ±half-length
    # along the (unit) meridian-tangent vector in pixels, project back
    # to data coords, and draw the line.
    if show_ticks and tick_length_points > 0:
        tnorm = np.hypot(dx_m, dy_m)
        tnorm[tnorm == 0] = 1.0
        tx = dx_m / tnorm
        ty = dy_m / tnorm
        L_px = float(tick_length_points) * ax.figure.dpi / 72.0
        half = 0.5 * L_px
        data_to_disp = ax.transData.transform
        disp_to_data = ax.transData.inverted().transform
        tcolor = tick_color if tick_color is not None else color
        for x, y, ux, uy in zip(x0, y0, tx, ty):
            anchor_disp = data_to_disp((x, y))
            a = disp_to_data(anchor_disp + np.array([ux, uy]) * half)
            b = disp_to_data(anchor_disp - np.array([ux, uy]) * half)
            ax.plot([a[0], b[0]], [a[1], b[1]],
                    color=tcolor, lw=tick_lw, solid_capstyle='butt',
                    zorder=5)

    # Honor caller-supplied ha / va; otherwise auto-pick near-edge
    # anchoring so ``offset_points`` is the actual visible gap
    # between the curve and the label, independent of font size or
    # rotation. This aligns the helper with the same
    # convention the Ruler and CoordinateOverlay.render_labels use.
    ha_kw = text_kwargs.pop('ha', 'auto')
    va_kw = text_kwargs.pop('va', 'auto')

    artists = []
    for lon, x, y, rot, ox, oy in zip(lons, x0, y0, rotations_deg, nx, ny):
        label = _format_lon_label(lon, frame=frame, fmt=fmt, sep=sep)
        if ha_kw == 'auto' or va_kw == 'auto':
            ha_auto, va_auto = _resolve_text_anchor(rot, 1, ox, oy)
            ha_final = ha_auto if ha_kw == 'auto' else ha_kw
            va_final = va_auto if va_kw == 'auto' else va_kw
        else:
            ha_final, va_final = ha_kw, va_kw
        if offset_points != 0.0:
            t = ax.annotate(
                label, xy=(x, y), xycoords='data',
                xytext=(ox * offset_points, oy * offset_points),
                textcoords='offset points',
                rotation=rot, ha=ha_final, va=va_final,
                fontsize=fontsize, color=color, **text_kwargs,
            )
        else:
            t = ax.text(
                x, y, label, rotation=rot, ha=ha_final, va=va_final,
                fontsize=fontsize, color=color, **text_kwargs,
            )
        artists.append(t)

    return artists

