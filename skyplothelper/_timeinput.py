"""Normalize any time-like input to an :class:`~astropy.time.Time`.

The time-side counterpart of :func:`~skyplothelper.plotting.to_lonlat`: one
place that decides what a user's time argument means, so every entry point in
the package accepts the same set of types and agrees on the answer.

The package leans on astropy for everything else, so the astropy-native type
should work wherever the stdlib one does -- and vice versa.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ['to_time']


def _is_plain_number(value: Any) -> bool:
    """True for a bare int/float (or array of them) carrying no epoch.

    Checked *after* Time and datetime, both of which would otherwise be
    coerced to an object-dtype array here.
    """
    try:
        arr = np.asarray(value)
    except Exception:
        return False
    return arr.dtype.kind in 'iuf'


def to_time(value: Any, *, scale: str | None = None,
            plain_format: str | None = None, _caller: str = 'sph.to_time',
            ) -> Any:
    """Normalize any time input to an :class:`~astropy.time.Time`.

    The supported one-liner for "I have a time in some form -- give me the
    astropy object". Accepts what the rest of the package accepts, so the
    result can go to astropy, to a non-skyplothelper library, or into your
    own math.

    Parameters
    ----------
    value : Time, datetime, str, or number
        An :class:`~astropy.time.Time` (scalar or array), a
        :class:`datetime.datetime`, an ISO/FITS date string, or a bare number
        together with *plain_format*.
    scale : str, optional
        Convert into this time scale (``'utc'``, ``'tt'``, ``'tdb'``, ...).
        Default ``None`` -- **the time comes back in its own scale,
        unconverted**.
    plain_format : str, optional
        How to read a *bare number*: ``'mjd'``, ``'jd'``, ``'decimalyear'``,
        and the other astropy time formats. Required for numeric input, and
        ignored for every other type, which carries its own epoch already.

    Returns
    -------
    astropy.time.Time

    Raises
    ------
    TypeError
        For a bare number with no *plain_format*. ``60000`` could be an MJD, a
        JD, or a decimal year, and the three differ by millions of days; the
        error names the fix rather than guessing.

    Notes
    -----
    The default deliberately does **not** convert to UTC. A bare
    ``to_time(t)`` is a lossless read-out of what you already have; silently
    coercing the scale is the failure mode this API exists to avoid -- a TT
    timestamp written into a header card that FITS defines as UTC is off by
    ~69 s, and nothing about the result looks wrong. Ask for a conversion
    explicitly with *scale*, or read ``.utc`` at the point of use.

    *scale* means "convert **into**", not "interpret as". To declare that a
    string or number is already on a non-UTC scale, build the Time yourself:
    ``Time('2026-07-19', scale='tt')``.

    Examples
    --------
    >>> t = sph.to_time('2026-07-19T12:00:00')       # ISO string
    >>> t = sph.to_time(datetime.datetime.utcnow())  # stdlib datetime
    >>> t = sph.to_time(60000, plain_format='mjd')   # bare number
    >>> t = sph.to_time(some_time, scale='utc')      # convert explicitly
    """
    from astropy.time import Time

    if _is_plain_number(value):
        if plain_format is None:
            raise TypeError(
                f'{_caller}: ambiguous numeric time {value!r} -- it could '
                'be an MJD, a JD, or a decimal year. Pass plain_format, e.g. '
                f"plain_format='mjd', or give an ISO string / datetime / "
                'astropy Time instead.')
        t = Time(value, format=plain_format)
    elif isinstance(value, Time):
        t = value
    else:
        # datetime, date, ISO or FITS string, or anything else astropy parses.
        # Let astropy raise its own error for the unparseable cases -- it
        # names the offending value more precisely than we could here.
        t = Time(value)

    if scale is not None:
        t = getattr(t, scale)
    return t


def _to_datetime(value: Any, *, caller: str) -> Any:
    """Coerce a time input to a stdlib :class:`datetime.datetime`.

    For handing off to libraries that duck-type on datetime attributes rather
    than accepting an astropy Time -- cartopy's ``Nightshade`` calls
    ``date.utcoffset()``, so a Time or an ISO string reaches it as an
    AttributeError. Converting through UTC here is correct rather than
    merely convenient: the consumer has no scale of its own to convert into.
    """
    import datetime as _dt

    if isinstance(value, _dt.datetime):
        return value
    return to_time(value, _caller=caller).utc.datetime
