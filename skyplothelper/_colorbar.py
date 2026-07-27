"""Shared colorbar-tick helpers.

The adaptive minor-tick logic was born in :mod:`skyplothelper.images.quicklook`
and is factored out here so the general-purpose
:func:`skyplothelper.add_colorbar` can use the same behavior — one home, so the
two cannot drift (the frame-short-label triplication that had to be collapsed
is the cautionary tale for three copies of one idea).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.ticker import (
    AutoMinorLocator,
    FixedLocator,
    FormatStrFormatter,
    FuncFormatter,
    NullLocator,
    StrMethodFormatter,
)

__all__ = ['decade_minor_ticks', 'norm_is_compressed', 'apply_minor_ticks',
           'tick_text', 'apply_adaptive_format', 'str_formatter']


def decade_minor_ticks(lo: float, hi: float) -> list[float]:
    """Minor-tick positions at 1/2/3/5 x 10^k spanning ``[lo, hi]``.

    The generalization of a hard-coded ``[1, 2, 3, 5, ... 5000]``: the same
    multiples, but over whatever decades the data occupies rather than
    assuming values of order 1-1000.
    """
    hi_abs = max(abs(float(lo)), abs(float(hi)))
    if not np.isfinite(hi_abs) or hi_abs <= 0:
        return []
    top = int(np.floor(np.log10(hi_abs)))
    ticks: list[float] = []
    for k in range(top - 4, top + 1):
        for m in (1, 2, 3, 5):
            v = m * (10.0 ** k)
            if lo <= v <= hi or lo <= -v <= hi:
                ticks.append(v if lo <= v <= hi else -v)
    return sorted(set(ticks))


# Minimum display separation between kept minor ticks, as a fraction of the
# bar's height. A fraction of the *normalized* bar, so it is size-independent:
# "keep ticks at least ~1.2% of bar height apart, drop those closer." Two ticks
# nearer than this overprint and read as a smudge rather than as subdivisions.
_MIN_TICK_FRAC = 0.012


def decrowd_display(positions: list[float], norm: Any,
                    min_frac: float = _MIN_TICK_FRAC) -> list[float]:
    """Drop decade ticks the *norm* does not visibly separate on the bar.

    :func:`decade_minor_ticks` enumerates 1/2/3/5 x 10^k by DATA-value decade,
    blind to how the norm maps them onto the display. On an asinh / symlog bar
    whose range sits in its linear regime, the low decades map to nearly one
    spot and the ticks pile up into a smudge at the base. Mapping each candidate
    through the norm and keeping only those at least *min_frac* of bar height
    apart de-crowds it in display space, where the pile-up actually happens.

    Candidates within *min_frac* of either end are also dropped: they collide
    with the major end-ticks, and the lowest is usually just a sub-linthresh
    value that reads as "approximately zero".
    """
    kept: list[float] = []
    last = -np.inf
    for v in sorted(positions):
        try:
            p = float(norm(v))
        except Exception:
            continue
        if not np.isfinite(p) or p < min_frac or p > 1.0 - min_frac:
            continue
        if p - last >= min_frac:
            kept.append(v)
            last = p
    return kept


def norm_is_compressed(norm: Any) -> bool:
    """True for a log / asinh / symlog / power / sqrt bar.

    Decade multiples only make sense on a compressed bar; on a linear one they
    pile up against zero, so the caller wants an even subdivision instead.
    astropy's ``ImageNormalize`` WRAPS a stretch, so the norm's own class name
    says nothing on its own — inspect the stretch it holds as well.
    """
    name = type(norm).__name__.lower()
    stretch = getattr(norm, 'stretch', None)
    if stretch is not None:
        name += ' ' + type(stretch).__name__.lower()
    return any(k in name for k in ('log', 'asinh', 'sinh', 'power', 'sqrt'))


def apply_minor_ticks(cbar: Any, minor_ticks: Any,
                      lo: float | None = None,
                      hi: float | None = None) -> None:
    """Set *cbar*'s minor-tick locator per the *minor_ticks* contract.

    ``minor_ticks``:
      * ``'auto'`` / ``None`` — adaptive: an even subdivision on a linear bar,
        1/2/3/5 x 10^k across the occupied decades on a compressed one.
      * ``False`` — no minor ticks.
      * a Locator — used directly.
      * a sequence — those exact positions.

    *lo* / *hi* default to the mappable's clim; they only matter for the
    compressed-bar decade positions.
    """
    # The tick axis follows the bar's orientation (vertical -> y, horizontal
    # -> x); a Colorbar reports its own orientation.
    axis = (cbar.ax.yaxis if getattr(cbar, 'orientation', 'vertical')
            == 'vertical' else cbar.ax.xaxis)

    if minor_ticks is False:
        axis.set_minor_locator(NullLocator())
        return
    if minor_ticks is None or (isinstance(minor_ticks, str)
                               and minor_ticks == 'auto'):
        norm = getattr(cbar, 'norm', None)
        if norm_is_compressed(norm):
            if lo is None or hi is None:
                try:
                    lo, hi = cbar.mappable.get_clim()
                except Exception:
                    lo, hi = None, None
            positions = ([] if lo is None or hi is None
                         else decade_minor_ticks(lo, hi))
            # Enumerated by value decade, so de-crowd in display space where an
            # asinh/symlog bar in its linear regime piles the low decades up.
            if norm is not None and positions:
                positions = decrowd_display(positions, norm)
            axis.set_minor_locator(FixedLocator(positions))
        else:
            axis.set_minor_locator(AutoMinorLocator(5))
        return
    if hasattr(minor_ticks, 'tick_values'):        # a Locator
        axis.set_minor_locator(minor_ticks)
        return
    axis.set_minor_locator(FixedLocator(list(minor_ticks)))


def tick_text(value: float, lo: float, hi: float) -> str:
    """Colorbar tick text with precision matched to the displayed range.

    A fixed ``'.0f'`` is fine for a 0-5000 mJy bar and useless for a 0-3 Jy
    one, where every tick rounds to the same integer.
    """
    span = abs(float(hi) - float(lo))
    if not np.isfinite(span) or span == 0:
        return f'{value:g}'
    # Aim for ~5 distinguishable steps across the bar.
    step = span / 5.0
    decimals = int(np.clip(np.ceil(-np.log10(step)) + 1, 0, 6))
    text = f'{value:.{decimals}f}'
    # Trim a pointless trailing '.000' the clip may produce.
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return text or '0'


def str_formatter(spec: str) -> Any:
    """A Formatter from a format string, accepting EITHER style.

    ``'%.3f'`` (printf / old) and ``'{x:.3f}'`` (str.format / new) are both in
    wide use, so detect rather than assume: a brace means new-style. Picking
    one silently prints the other verbatim, which is the trap this avoids.
    """
    if '{' in spec:
        return StrMethodFormatter(spec)
    return FormatStrFormatter(spec)


def apply_adaptive_format(cbar: Any, lo: float | None = None,
                          hi: float | None = None) -> None:
    """Set *cbar*'s MAJOR-tick formatter to :func:`tick_text` precision.

    Unlike the minor-tick default, this is opt-in: it rewrites every major
    label rather than adding to the bar, so the caller asks for it explicitly.
    *lo* / *hi* default to the mappable's clim.
    """
    if lo is None or hi is None:
        try:
            lo, hi = cbar.mappable.get_clim()
        except Exception:
            return
    axis = (cbar.ax.yaxis if getattr(cbar, 'orientation', 'vertical')
            == 'vertical' else cbar.ax.xaxis)
    axis.set_major_formatter(FuncFormatter(lambda x, _: tick_text(x, lo, hi)))
