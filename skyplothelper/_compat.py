"""Astropy version flags and compatibility helpers.

These are internal — not re-exported from the package ``__init__``.
"""

from __future__ import annotations

from typing import Any

import astropy

# --- Astropy version detection for feature gating ---
try:
    _ASTROPY_VERSION = tuple(int(x) for x in
                             astropy.__version__.split('.')[:2]
                             if x.isdigit())
    if len(_ASTROPY_VERSION) < 2:
        _ASTROPY_VERSION = (0, 0)
except Exception:
    _ASTROPY_VERSION = (0, 0)
_ASTROPY_GE_7 = _ASTROPY_VERSION >= (7, 0)  # simplify kwarg in set_ticklabel


def coord_ticks(coord: Any) -> Any:
    """A coordinate helper's ``Ticks`` artist, without tripping astropy 7's
    deprecation of direct ``.ticks`` access.

    Astropy 7 moved the artist to the private ``._ticks`` and made ``.ticks`` a
    deprecated property (``CoordinateHelper.ticks should not be accessed
    directly``); astropy < 7 exposes it as a plain ``.ticks`` attribute. Use
    this for the low-level operations that have no public equivalent
    (``set_tick_out``, ``set_minor_ticksize``, …).

    Oriented for the newer astropy by default: prefer the ``._ticks`` attribute
    and fall back to ``.ticks`` on older releases (duck-typed rather than
    version-gated, so it stays correct even if version detection is off)."""
    ticks = getattr(coord, '_ticks', None)
    return ticks if ticks is not None else coord.ticks


def coord_ticklabels(coord: Any) -> Any:
    """A coordinate helper's ``TickLabels`` artist, without tripping astropy 7's
    deprecation of direct ``.ticklabels`` access (moved to ``._ticklabels``).
    Prefers the newer ``._ticklabels`` and falls back to ``.ticklabels`` on
    older astropy. See :func:`coord_ticks`."""
    tls = getattr(coord, '_ticklabels', None)
    return tls if tls is not None else coord.ticklabels


# lazily populated by _safe_ticklabel_kwargs: (accepted-kwarg-names | None,
# has_varkw). ``None`` for the names signals "unknown -> permissive".
_SET_TICKLABEL_SIG: tuple[set[str] | None, bool] | None = None


def _safe_ticklabel_kwargs(kw: dict[str, Any]) -> dict[str, Any]:
    """
    Filter set_ticklabel kwargs for astropy version compatibility.

    Strips any kwarg the installed astropy's ``CoordinateHelper.set_ticklabel``
    does not accept. The first problem case was ``simplify`` (astropy >=7.0);
    this introspection-based approach auto-adapts to future changes.

    If signature introspection fails (very old astropy, monkey-patched
    methods, etc.), falls back to the hard-coded ``simplify`` gate.
    """
    global _SET_TICKLABEL_SIG
    if _SET_TICKLABEL_SIG is None:
        try:
            import inspect

            from astropy.visualization.wcsaxes import CoordinateHelper
            sig = inspect.signature(CoordinateHelper.set_ticklabel)
            params = sig.parameters
            # If **kwargs is present, any kwarg is accepted - no filtering needed.
            has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD
                            for p in params.values())
            _SET_TICKLABEL_SIG = (set(params.keys()), has_varkw)
        except Exception:
            _SET_TICKLABEL_SIG = (None, True)  # unknown -> permissive
    accepted, has_varkw = _SET_TICKLABEL_SIG
    if has_varkw or accepted is None:
        # Permissive path - but still strip simplify on known-old astropy as
        # a final safety net since the hard-coded check is free.
        if not _ASTROPY_GE_7 and 'simplify' in kw:
            kw = {k: v for k, v in kw.items() if k != 'simplify'}
        return kw
    # Drop any kwarg the signature doesn't accept.
    bad = [k for k in kw if k not in accepted]
    if not bad:
        return kw
    return {k: v for k, v in kw.items() if k not in bad}


def _compat_format(fmt: str | None) -> str | None:
    """
    Strip leading ``+`` from format strings for astropy < 7.0 compatibility.

    Astropy >=7.0 supports a ``+`` prefix in angle format strings (e.g.
    ``+dd:mm:ss``) to force display of the sign for positive values.
    Earlier versions raise ``ValueError`` for this syntax because the
    DMS/DDEC regexes don't match it, and the partial failure leaves the
    ``AngleFormatterLocator`` without ``_fields`` set.
    """
    if not _ASTROPY_GE_7 and fmt is not None and fmt.startswith('+'):
        return fmt[1:]
    return fmt
