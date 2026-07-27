"""Image clipping, stretching, normalization, and stretch registry.

Wraps astropy.visualization stretch/interval classes with thin convenience
helpers (``clip_percentile``, ``clip_sigma``, ``clip_zscale``, ``rescale_image``,
``make_norm``, ``auto_stretch``, ...) and an internal ``_STRETCH_REGISTRY``
of inline stretch functions for cases where astropy isn't required.
"""

from __future__ import annotations

import warnings
from typing import Any

import astropy.units as u  # noqa: F401  (used in describe_image / make_norm docstrings)
import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt
from astropy.visualization import (
    AsymmetricPercentileInterval,
    ImageNormalize,
    ManualInterval,
    PercentileInterval,
    ZScaleInterval,
)


def clip_percentile(data: npt.ArrayLike, plo: float = 0.5,
                    phi: float = 99.5) -> tuple[float, float]:
    """
    Determine display limits from percentile clipping.

    Parameters
    ----------
    data : ndarray
        Input image (NaN/Inf-safe).
    plo, phi : float
        Lower and upper percentiles (0–100).

    Returns
    -------
    vmin, vmax : float
    """
    valid = np.asarray(data)[np.isfinite(data)]
    if len(valid) == 0:
        return 0.0, 1.0
    # One percentile call (one partition pass) instead of two over the full image.
    lo, hi = np.percentile(valid, [plo, phi])
    return float(lo), float(hi)


def clip_sigma(data: npt.ArrayLike, sigma_lo: float = 3,
               sigma_hi: float | None = None,
               niter: int = 5) -> tuple[float, float]:
    """
    Determine display limits from iterative sigma clipping.

    Parameters
    ----------
    data : ndarray
    sigma_lo : float
        Lower clip threshold (in σ).
    sigma_hi : float or None
        Upper clip threshold. If None, uses ``sigma_lo``.
    niter : int
        Number of clipping iterations.

    Returns
    -------
    vmin, vmax : float
    """
    if sigma_hi is None:
        sigma_hi = sigma_lo
    arr = np.asarray(data, dtype=float).ravel()
    mask = np.isfinite(arr)
    for _ in range(niter):
        sub = arr[mask]
        if len(sub) < 3:
            break
        med = np.median(sub)
        std = np.std(sub)
        if std == 0:
            break
        mask = mask & (arr >= med - sigma_lo * std) & (arr <= med + sigma_hi * std)
    sub = arr[mask]
    if len(sub) == 0:
        return float(np.nanmin(arr)), float(np.nanmax(arr))
    return float(np.min(sub)), float(np.max(sub))


def clip_zscale(data: npt.ArrayLike, contrast: float = 0.25,
                nsamples: int = 1000) -> tuple[float, float]:
    """
    Determine display limits using the ZScale algorithm (IRAF/DS9 style).

    Uses ``astropy.visualization.ZScaleInterval`` when available, falling
    back to a percentile approximation otherwise.

    Parameters
    ----------
    data : ndarray
    contrast : float
        ZScale contrast parameter (0 = full range, 1 = minimal range).
    nsamples : int
        Number of pixels to sample.

    Returns
    -------
    vmin, vmax : float
    """
    try:
        from astropy.visualization import ZScaleInterval
        interval = ZScaleInterval(contrast=contrast, n_samples=nsamples)
        vmin, vmax = interval.get_limits(np.asarray(data, dtype=float))
        return float(vmin), float(vmax)
    except ImportError:
        # Fallback: percentile approximation
        p = max(1, 100 * (1 - contrast) / 2)
        return clip_percentile(data, p, 100 - p)


def auto_interval(data: npt.ArrayLike, method: str = 'percentile',
                  **kwargs: Any) -> tuple[float, float]:
    """
    Unified interval (vmin, vmax) detection.

    Parameters
    ----------
    data : ndarray
    method : str
        'percentile' (default), 'sigma', 'zscale', 'minmax', or 'manual'.
    **kwargs
        Passed to the underlying clip function:

        - percentile: ``plo``, ``phi``
        - sigma: ``sigma_lo``, ``sigma_hi``, ``niter``
        - zscale: ``contrast``, ``nsamples``
        - minmax: (no kwargs)
        - manual: ``vmin``, ``vmax`` (required)

    Returns
    -------
    vmin, vmax : float
    """
    method = method.lower()
    if method == 'percentile':
        return clip_percentile(data,
                               kwargs.get('plo', 0.5),
                               kwargs.get('phi', 99.5))
    elif method == 'sigma':
        return clip_sigma(data,
                          kwargs.get('sigma_lo', 3),
                          kwargs.get('sigma_hi', None),
                          kwargs.get('niter', 5))
    elif method == 'zscale':
        return clip_zscale(data,
                           kwargs.get('contrast', 0.25),
                           kwargs.get('nsamples', 1000))
    elif method == 'minmax':
        valid = np.asarray(data)[np.isfinite(data)]
        if len(valid) == 0:
            return 0.0, 1.0
        return float(np.min(valid)), float(np.max(valid))
    elif method == 'manual':
        return float(kwargs['vmin']), float(kwargs['vmax'])
    else:
        raise ValueError(
            f"Unknown interval method '{method}'. "
            "Available: percentile, sigma, zscale, minmax, manual")


# ---- Stretch functions ----

# Registry mapping stretch names → (astropy class, fallback numpy func).
# Fallback is used when astropy.visualization is unavailable or for
# standalone operation.

def _stretch_linear(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return x

def _stretch_sqrt(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.sqrt(np.clip(x, 0, None))

def _stretch_squared(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return x ** 2

def _stretch_log(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    a = 1000  # log scaling parameter
    return np.log(a * np.clip(x, 0, None) + 1) / np.log(a + 1)

def _stretch_asinh(x: npt.NDArray[np.float64],
                   a: float = 0.1) -> npt.NDArray[np.float64]:
    return np.arcsinh(x / a) / np.arcsinh(1.0 / a)

def _stretch_sinh(x: npt.NDArray[np.float64],
                  a: float = 0.3333) -> npt.NDArray[np.float64]:
    return np.sinh(a * x) / np.sinh(a)

def _stretch_power(x: npt.NDArray[np.float64],
                   a: float = 3) -> npt.NDArray[np.float64]:
    return x ** a

_STRETCH_REGISTRY = {
    'linear':  _stretch_linear,
    'sqrt':    _stretch_sqrt,
    'squared': _stretch_squared,
    'log':     _stretch_log,
    'asinh':   _stretch_asinh,
    'sinh':    _stretch_sinh,
    'power':   _stretch_power,
}


def _get_stretch_func(name: str, **kwargs: Any) -> Any:
    """Look up stretch function by name, trying astropy first."""
    name_lower = name.lower()

    # Try astropy.visualization stretch classes (more robust)
    try:
        from astropy.visualization import (
            AsinhStretch,
            HistEqStretch,
            LinearStretch,
            LogStretch,
            PowerStretch,
            SinhStretch,
            SqrtStretch,
            SquaredStretch,
        )
        _astropy_stretches = {
            'linear': LinearStretch,
            'sqrt': SqrtStretch,
            'squared': SquaredStretch,
            'log': lambda: LogStretch(a=kwargs.get('a', 1000)),
            'asinh': lambda: AsinhStretch(a=kwargs.get('a', 0.1)),
            'sinh': lambda: SinhStretch(a=kwargs.get('a', 1/3)),
            'power': lambda: PowerStretch(a=kwargs.get('a', 3)),
            'histeq': lambda: HistEqStretch(
                kwargs['_data']) if '_data' in kwargs else None,
        }
        if name_lower in _astropy_stretches:
            cls = _astropy_stretches[name_lower]
            try:
                stretch_obj = cls() if isinstance(cls, type) else cls()
            except TypeError:
                # histeq without data — fall through to error or fallback
                if name_lower == 'histeq':
                    raise ValueError(
                        "HistEqStretch requires data. Pass data via "
                        "rescale_image() which handles this automatically.")
                raise
            if stretch_obj is not None:
                return stretch_obj
    except ImportError:
        pass

    # Fallback to numpy implementations
    if name_lower in _STRETCH_REGISTRY:
        fn = _STRETCH_REGISTRY[name_lower]
        return fn

    raise ValueError(
        f"Unknown stretch '{name}'. Available: "
        + ", ".join(sorted(set(list(_STRETCH_REGISTRY.keys())
                               + ['histeq', 'symlog', 'symmetric_log']))))


def rescale_image(data: npt.ArrayLike, stretch: str = 'linear',
                  clip: str = 'percentile',
                  plo: float = 0.5, phi: float = 99.5,
                  vmin: float | None = None, vmax: float | None = None,
                  sigma: float = 3, contrast: float = 0.25,
                  a: float | None = None,
                  fill_nan: float = 0.0) -> npt.NDArray[np.float64]:
    """
    Rescale image data to [0, 1] with clipping and stretch.

    This is the main convenience function for preparing image data
    for display. It combines interval detection (clipping) with
    a nonlinear stretch in a single call.

    Parameters
    ----------
    data : ndarray
        Input 2D image array.
    stretch : str
        Stretch function: 'linear', 'sqrt', 'squared', 'log', 'asinh',
        'sinh', 'power', 'histeq', 'symlog', 'symmetric_log'. Default
        'linear'. ``'symlog'`` (matplotlib ``SymLogNorm``) and
        ``'symmetric_log'`` (the C¹-continuous pysymlog variant; requires the
        optional ``pysymlog`` extra) are signed-data transforms — linear near
        zero, logarithmic in the wings — for residual / velocity / Q-U maps.
    clip : str
        Clipping method: 'percentile', 'sigma', 'zscale', 'minmax',
        'manual'. Default 'percentile'.
    plo, phi : float
        Percentile bounds (used when clip='percentile').
    vmin, vmax : float, optional
        Manual display limits. If provided, overrides clip method.
    sigma : float
        Sigma threshold (used when clip='sigma').
    contrast : float
        ZScale contrast (used when clip='zscale').
    a : float, optional
        Stretch parameter (meaning depends on stretch type): asinh/sinh
        linear-width, log steepness, power exponent, or — for
        ``symlog`` / ``symmetric_log`` — the ``linthresh`` / ``shift`` in
        data units (defaults: 1% / 0.1% of the display range).
    fill_nan : float
        Value to substitute for NaN/Inf pixels. Default 0.

    Returns
    -------
    scaled : ndarray
        Image rescaled to [0, 1], same shape as input.

    Examples
    --------
    >>> scaled = rescale_image(data, stretch='asinh', clip='zscale')
    >>> scaled = rescale_image(data, stretch='sqrt', plo=1, phi=99)
    >>> scaled = rescale_image(data, stretch='log', vmin=0.001, vmax=10)
    """
    arr = np.asarray(data, dtype=float).copy()

    # Determine interval
    if vmin is not None and vmax is not None:
        lo, hi = float(vmin), float(vmax)
    else:
        clip_kw = {}
        if clip == 'percentile':
            clip_kw = {'plo': plo, 'phi': phi}
        elif clip == 'sigma':
            clip_kw = {'sigma_lo': sigma, 'sigma_hi': sigma}
        elif clip == 'zscale':
            clip_kw = {'contrast': contrast}
        lo, hi = auto_interval(arr, method=clip, **clip_kw)
        # Allow explicit override of one end
        if vmin is not None:
            lo = float(vmin)
        if vmax is not None:
            hi = float(vmax)

    # Clip and normalize to [0, 1]
    if hi <= lo:
        hi = lo + 1.0
    arr = np.nan_to_num(arr, nan=fill_nan, posinf=hi, neginf=lo)

    # symlog / symmetric_log are *signed-data* transforms: they need the data
    # on its native scale (the linear pre-normalize below would collapse the
    # sign and zero-point a symmetric-log relies on). Route them straight
    # through the matplotlib ``SymLogNorm`` / pysymlog ``SymmetricLogarithmNorm``
    # (which map data → [0, 1] directly), reusing ``make_norm`` so the
    # ``linthresh`` / ``shift`` defaults match the same-named mpl-side stretch.
    # ``a`` sets linthresh (symlog) / shift (symmetric_log), both in data units.
    if stretch.lower() in ('symlog', 'symmetric_log'):
        norm = make_norm(stretch, vmin=lo, vmax=hi, a=a)
        out = np.asarray(norm(arr), dtype=float)
        return np.clip(np.nan_to_num(out, nan=fill_nan), 0.0, 1.0)

    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0).copy()  # writeable for astropy stretches

    # Apply stretch
    stretch_kw = {}
    if a is not None:
        stretch_kw['a'] = a
    if stretch.lower() == 'histeq':
        stretch_kw['_data'] = arr[np.isfinite(arr)]
    stretch_fn = _get_stretch_func(stretch, **stretch_kw)

    # astropy stretch objects are callable on arrays
    if hasattr(stretch_fn, '__call__'):
        arr = np.asarray(stretch_fn(arr), dtype=float)

    return np.clip(arr, 0.0, 1.0)


def make_norm(stretch: Any = 'linear', vmin: float | None = None,
              vmax: float | None = None, data: npt.ArrayLike | None = None,
              clip: str = 'percentile', plo: float = 0.5, phi: float = 99.5,
              sigma: float = 3, contrast: float = 0.25,
              a: float | None = None,
              interval: Any = None) -> Any:
    """
    Create a matplotlib Normalize object from stretch/interval names.

    Internally builds an ``astropy.visualization.ImageNormalize`` (which
    is itself a subclass of ``matplotlib.colors.Normalize``), composing
    an interval × stretch. The string-name API matches the rest of the
    package; advanced users may pass astropy ``BaseStretch`` /
    ``BaseInterval`` objects directly for full composability.

    Useful for passing directly to ``imshow(norm=...)``.  If ``data``
    is provided and ``vmin``/``vmax`` are not, the interval is computed
    automatically.

    Parameters
    ----------
    stretch : str or astropy.visualization.BaseStretch
        Stretch name or a stretch object. Names: 'linear', 'sqrt',
        'squared'/'power2', 'power', 'log', 'asinh', 'sinh', 'histeq',
        'symlog', 'symmetric_log'. Stretches compose with ``+``, e.g.
        ``AsinhStretch(a=0.05) + SqrtStretch()``.

        ``'symlog'`` is special: astropy ships no SymLogStretch, so
        this returns a matplotlib ``SymLogNorm`` directly. All other
        astropy-backed names route through ``ImageNormalize``.

        ``'symmetric_log'`` requires the optional ``pysymlog``
        dependency and returns a ``pysymlog.SymmetricLogarithmNorm``
        — a C¹-continuous variant of symlog (``log((x+shift)/shift)``)
        with no visible "kink" at the linear-to-log boundary. The
        smoother gradient near zero is the right call for signed-
        residual maps, velocity fields, and polarization Q/U. Install
        with ``pip install pysymlog`` or
        ``pip install skyplothelper[pysymlog]``. The ``a`` kwarg maps
        to pysymlog's ``shift`` (~ 1/10 of mpl's ``linthresh`` for a
        visually-similar transition scale).
    vmin, vmax : float, optional
        Display limits. Auto-computed from ``data`` if None.
    data : ndarray, optional
        Image data for auto-interval computation. Required for
        ``stretch='histeq'``.
    clip : str
        Interval method when auto-computing from data: ``'percentile'``,
        ``'zscale'``, ``'sigma'``, ``'minmax'``, ``'manual'``. Ignored
        when ``interval`` is given explicitly.
    plo, phi : float
        Percentile bounds for ``clip='percentile'``.
    sigma : float
        Sigma threshold for ``clip='sigma'``.
    contrast : float
        ZScale contrast for ``clip='zscale'``.
    a : float, optional
        Stretch parameter (meaning depends on stretch type — log
        steepness, asinh linear-width fraction, power exponent, etc.).
    interval : astropy.visualization.BaseInterval, optional
        Escape hatch for full astropy interval objects, e.g.
        ``AsymmetricPercentileInterval(0.5, 99.5)`` or
        ``ZScaleInterval(contrast=0.1)``. If given, overrides ``clip``.

    Returns
    -------
    norm : matplotlib.colors.Normalize
        An ``ImageNormalize`` subclass instance (or ``SymLogNorm`` for
        ``stretch='symlog'``). Drop-in for ``imshow(norm=...)``.

    Examples
    --------
    >>> norm = make_norm('asinh', data=image, clip='zscale')
    >>> ax.imshow(image, norm=norm, cmap='viridis')

    >>> norm = make_norm('log', vmin=0.001, vmax=100)

    Compose stretches and pass an explicit interval:

    >>> from astropy.visualization import (AsinhStretch, SqrtStretch,
    ...     AsymmetricPercentileInterval)
    >>> norm = make_norm(stretch=AsinhStretch(0.05) + SqrtStretch(),
    ...                  interval=AsymmetricPercentileInterval(0.1, 99.9),
    ...                  data=image)
    """
    from astropy.visualization import (
        AsinhStretch,
        BaseInterval,
        BaseStretch,
        HistEqStretch,
        LinearStretch,
        LogStretch,
        MinMaxInterval,
        PowerStretch,
        SinhStretch,
        SqrtStretch,
        SquaredStretch,
    )

    # --- Resolve stretch ---
    if isinstance(stretch, BaseStretch):
        stretch_obj = stretch
        s = None
    elif isinstance(stretch, str):
        s = stretch.lower()
        if s in ('symlog', 'symmetric_log'):
            # No astropy equivalent — keep matplotlib (or pysymlog) path.
            # Compute interval below, then return SymLogNorm /
            # SymmetricLogarithmNorm directly.
            stretch_obj = None
        elif s == 'linear':
            stretch_obj = LinearStretch()
        elif s == 'sqrt':
            stretch_obj = SqrtStretch()
        elif s in ('squared', 'power2'):
            stretch_obj = SquaredStretch()
        elif s == 'power':
            stretch_obj = PowerStretch(a=a if a is not None else 3)
        elif s == 'log':
            # astropy's LogStretch parameter `a` controls steepness
            # (default 1000). Higher = more compression of bright end.
            stretch_obj = LogStretch(a=a if a is not None else 1000)
        elif s == 'asinh':
            # astropy's `a` is the linear-width fraction (0–1).
            # Default 0.1 matches astropy's own default.
            stretch_obj = AsinhStretch(a=a if a is not None else 0.1)
        elif s == 'sinh':
            stretch_obj = SinhStretch(a=a if a is not None else 1/3.)
        elif s == 'histeq':
            if data is None:
                raise ValueError(
                    "stretch='histeq' requires data= for histogram-"
                    "equalization computation.")
            _arr = np.asarray(data)
            stretch_obj = HistEqStretch(_arr[np.isfinite(_arr)])
        else:
            warnings.warn(
                f"Unknown stretch '{stretch}' for make_norm. "
                "Available: linear, sqrt, squared, power2, power, "
                "log, asinh, sinh, histeq, symlog, symmetric_log. "
                "Returning linear.")
            stretch_obj = LinearStretch()
            s = 'linear'
    else:
        raise TypeError(
            f"stretch must be a string or BaseStretch instance, "
            f"got {type(stretch).__name__}")

    # --- Resolve interval ---
    if interval is not None and not isinstance(interval, BaseInterval):
        raise TypeError(
            "interval must be an astropy.visualization.BaseInterval "
            f"instance, got {type(interval).__name__}")

    if interval is None:
        if vmin is not None and vmax is not None:
            interval_obj = ManualInterval(vmin=vmin, vmax=vmax)
        elif clip == 'percentile':
            # Symmetric case: plo and (100 - phi) are equal — i.e.
            # the same fraction trimmed off both ends. Use the simpler
            # PercentileInterval; otherwise use the asymmetric form.
            if abs(plo - (100 - phi)) < 1e-9:
                interval_obj = PercentileInterval(phi - plo)
            else:
                interval_obj = AsymmetricPercentileInterval(plo, phi)
        elif clip == 'zscale':
            interval_obj = ZScaleInterval(contrast=contrast)
        elif clip == 'minmax':
            interval_obj = MinMaxInterval()
        elif clip == 'sigma':
            # No astropy SigmaInterval; reuse our clip_sigma helper
            if data is None:
                interval_obj = ManualInterval(
                    vmin=vmin if vmin is not None else 0.0,
                    vmax=vmax if vmax is not None else 1.0)
            else:
                lo, hi = clip_sigma(data, sigma_lo=sigma, sigma_hi=sigma)
                interval_obj = ManualInterval(vmin=lo, vmax=hi)
        elif clip in ('manual', None):
            interval_obj = ManualInterval(
                vmin=vmin if vmin is not None else 0.0,
                vmax=vmax if vmax is not None else 1.0)
        else:
            warnings.warn(
                f"Unknown clip method '{clip}' for make_norm. Falling "
                "back to MinMaxInterval. Available: percentile, sigma, "
                "zscale, minmax, manual.")
            interval_obj = MinMaxInterval()
    else:
        interval_obj = interval

    # --- Special case: symlog / symmetric_log route outside ImageNormalize ---
    if s in ('symlog', 'symmetric_log'):
        # Use the interval to determine vmin/vmax
        if data is not None:
            try:
                _vlo, _vhi = interval_obj.get_limits(np.asarray(data))
            except Exception:
                _vlo = vmin if vmin is not None else 0.0
                _vhi = vmax if vmax is not None else 1.0
        else:
            _vlo = vmin if vmin is not None else 0.0
            _vhi = vmax if vmax is not None else 1.0
        if vmin is not None:
            _vlo = vmin
        if vmax is not None:
            _vhi = vmax

        if s == 'symmetric_log':
            try:
                import pysymlog
            except ImportError as exc:
                raise ImportError(
                    "stretch='symmetric_log' requires the optional "
                    "`pysymlog` package. Install with `pip install "
                    "pysymlog` or `pip install skyplothelper[pysymlog]`. "
                    "For the matplotlib-shipped piecewise variant, use "
                    "stretch='symlog' instead."
                ) from exc
            # ``SymmetricLogarithmNorm`` is attached to the pysymlog
            # module only after the (idempotent) matplotlib registration.
            pysymlog.register_mpl()
            SymmetricLogarithmNorm = pysymlog.SymmetricLogarithmNorm
            # pysymlog's `shift` is documented as ~1/10 of mpl's
            # `linthresh` for visually-similar transitions, so honor
            # the same heuristic: default shift = 0.1% of range.
            shift = a if a is not None else max(abs(_vhi - _vlo) * 0.001,
                                                1e-10)
            return SymmetricLogarithmNorm(shift=shift,
                                           vmin=_vlo, vmax=_vhi)

        linthresh = a if a is not None else max(abs(_vhi - _vlo) * 0.01,
                                                1e-10)
        return mcolors.SymLogNorm(linthresh=linthresh,
                                  vmin=_vlo, vmax=_vhi)

    # --- Compose into ImageNormalize ---
    # If user gave both data and explicit vmin/vmax, ImageNormalize
    # uses vmin/vmax; if only data, it uses interval; if neither, the
    # ManualInterval(0,1) fallback above takes over.
    #
    # Defensive guard: if `data` was passed but contains no finite
    # values, astropy's interval.get_limits() can raise IndexError
    # (e.g. ZScaleInterval crashes on all-NaN input). Catch this and
    # fall back to ManualInterval so callers don't have to special-case.
    if data is not None and (vmin is None or vmax is None):
        try:
            _arr = np.asarray(data)
            _has_finite = bool(np.any(np.isfinite(_arr)))
        except Exception:
            _has_finite = True
        if not _has_finite:
            warnings.warn(
                "make_norm: input data has no finite values; falling "
                "back to ManualInterval(vmin=0, vmax=1).")
            interval_obj = ManualInterval(
                vmin=vmin if vmin is not None else 0.0,
                vmax=vmax if vmax is not None else 1.0)
            data = None  # avoid astropy re-trying with the same bad input

    return ImageNormalize(data=data, interval=interval_obj,
                          stretch=stretch_obj,
                          vmin=vmin, vmax=vmax)


def adjust_gamma(data: npt.ArrayLike, gamma: float) -> npt.NDArray[np.float64]:
    """
    Apply gamma correction to image data. NaN-safe.

    Parameters
    ----------
    data : ndarray
        Image array (should be non-negative for meaningful results).
    gamma : float
        Gamma exponent. Values < 1 brighten; > 1 darken.

    Returns
    -------
    corrected : ndarray
    """
    return np.where(np.isfinite(data),
                    np.clip(data, 0, None) ** float(gamma),
                    data)


def auto_stretch(data: npt.ArrayLike,
                 verbose: bool = False) -> tuple[str, str]:
    """
    Recommend a stretch function based on image data statistics.

    Examines the dynamic range, skewness, zero-fraction, and sign
    distribution to suggest the most appropriate stretch.  The
    returned name can be passed directly to ``rescale_image()`` or
    ``make_norm()``.

    Parameters
    ----------
    data : ndarray
        Input 2D image array.
    verbose : bool
        If True, print the reasoning behind the recommendation.

    Returns
    -------
    stretch : str
        Recommended stretch name.
    reason : str
        One-line explanation.

    Examples
    --------
    >>> stretch, reason = auto_stretch(image_data)
    >>> scaled = rescale_image(image_data, stretch=stretch)
    """
    arr = np.asarray(data, dtype=float)
    valid = arr[np.isfinite(arr)]

    if len(valid) < 10:
        return 'linear', 'Too few valid pixels for analysis'

    vmin, vmax = float(np.min(valid)), float(np.max(valid))  # noqa: F841 (vmax kept paired)
    med = float(np.median(valid))  # noqa: F841 (kept for downstream extension)
    mean = float(np.mean(valid))
    std = float(np.std(valid))

    # Percentile-based range (robust to outliers)
    p01, p99 = np.percentile(valid, [1, 99])
    robust_range = p99 - p01

    # Dynamic range (ratio of max to noise floor)
    if robust_range > 0:
        dynamic_range = (p99 - p01) / max(std, 1e-30)
    else:
        return 'linear', 'Constant image'

    # Fraction of pixels near zero (within 1σ of zero)
    near_zero = np.sum(np.abs(valid) < std) / len(valid) if std > 0 else 0

    # Skewness (positive = tail toward bright)
    if std > 0:
        skew = float(np.mean(((valid - mean) / std) ** 3))
    else:
        skew = 0.0

    # Has significant negative values? (radio continuum pattern)
    has_negative = vmin < -3 * std and np.sum(valid < -std) > 0.01 * len(valid)

    # --- Decision tree ---
    if has_negative:
        stretch = 'linear'
        reason = (f'Signed data (min={vmin:.2g}), negative fraction '
                  f'> 1% → linear (use make_norm("symlog") for display)')
    elif dynamic_range > 100 and skew > 3:
        stretch = 'log'
        reason = (f'High dynamic range ({dynamic_range:.0f}σ) with strong '
                  f'positive skew ({skew:.1f}) → log')
    elif dynamic_range > 30 and skew > 1.5:
        stretch = 'asinh'
        reason = (f'Moderate-high dynamic range ({dynamic_range:.0f}σ) '
                  f'with skew {skew:.1f} → asinh (smooth near zero)')
    elif near_zero > 0.7 and skew > 1:
        stretch = 'sqrt'
        reason = (f'{near_zero:.0%} pixels near zero, positive skew '
                  f'→ sqrt (enhances faint structure)')
    elif 0.5 < skew <= 1.5:
        stretch = 'sqrt'
        reason = f'Mild positive skew ({skew:.1f}) → sqrt'
    elif abs(skew) <= 0.5 and dynamic_range < 10:
        stretch = 'linear'
        reason = (f'Low dynamic range ({dynamic_range:.1f}σ), '
                  f'symmetric → linear')
    elif abs(skew) <= 0.5 and dynamic_range >= 10:
        stretch = 'histeq'
        reason = (f'Symmetric but wide range ({dynamic_range:.0f}σ) '
                  f'→ histeq (maximize contrast)')
    else:
        stretch = 'asinh'
        reason = f'General case (skew={skew:.1f}, DR={dynamic_range:.0f}σ) → asinh'

    if verbose:
        print(f"  Recommended stretch: {stretch}")
        print(f"  Reason: {reason}")

    return stretch, reason


def describe_image(data: npt.ArrayLike,
                   name: str = 'Image') -> dict[str, Any]:
    """
    Print a diagnostic summary of an image array.

    Includes shape, data type, value statistics, dynamic range,
    NaN/Inf pixel counts, and a recommended stretch function.

    Parameters
    ----------
    data : ndarray
        Input image array (2D or higher).
    name : str
        Label for the printout header.

    Returns
    -------
    info : dict
        Dictionary of computed statistics (for programmatic use).

    Examples
    --------
    >>> info = describe_image(fits_data, name='NGC1275_Xband')
    Image: NGC1275_Xband
      Shape:       (512, 512)
      Dtype:       float64
      Valid px:    261,887 / 262,144 (99.9%)
      Min / Max:   -0.00234 / 1.847
      Mean ± Std:  0.0156 ± 0.0891
      Median:      0.00312
      Percentiles: 1%=-0.0019  25%=0.0004  75%=0.0089  99%=0.387
      Dynamic range: 42.3σ (p1–p99)
      Recommended stretch: asinh
        Reason: Moderate-high dynamic range (42σ) with skew 8.3 → asinh
    """
    arr = np.asarray(data, dtype=float)
    total_px = arr.size
    finite_mask = np.isfinite(arr)
    n_valid = int(np.sum(finite_mask))
    n_nan = int(np.sum(np.isnan(arr)))
    n_inf = int(np.sum(np.isinf(arr)))
    valid = arr[finite_mask]

    info = {
        'shape': arr.shape,
        'dtype': arr.dtype,
        'total_pixels': total_px,
        'valid_pixels': n_valid,
        'nan_pixels': n_nan,
        'inf_pixels': n_inf,
    }

    print(f"Image: {name}")
    print(f"  Shape:       {arr.shape}")
    print(f"  Dtype:       {arr.dtype}")
    pct_valid = 100 * n_valid / max(total_px, 1)
    print(f"  Valid px:    {n_valid:,} / {total_px:,} ({pct_valid:.1f}%)")

    if n_nan > 0 or n_inf > 0:
        parts = []
        if n_nan > 0:
            parts.append(f"{n_nan:,} NaN")
        if n_inf > 0:
            parts.append(f"{n_inf:,} Inf")
        print(f"  Bad pixels:  {', '.join(parts)}")

    if n_valid == 0:
        print("  No valid pixels — cannot compute statistics.")
        info['recommended_stretch'] = 'linear'
        info['stretch_reason'] = 'No valid data'
        return info

    vmin, vmax = float(np.min(valid)), float(np.max(valid))
    vmean, vstd = float(np.mean(valid)), float(np.std(valid))
    vmed = float(np.median(valid))
    p01, p25, p75, p99 = [float(v) for v in np.percentile(valid, [1, 25, 75, 99])]

    info.update({
        'min': vmin, 'max': vmax,
        'mean': vmean, 'std': vstd, 'median': vmed,
        'p01': p01, 'p25': p25, 'p75': p75, 'p99': p99,
    })

    print(f"  Min / Max:   {vmin:.4g} / {vmax:.4g}")
    print(f"  Mean ± Std:  {vmean:.4g} ± {vstd:.4g}")
    print(f"  Median:      {vmed:.4g}")
    print(f"  Percentiles: 1%={p01:.4g}  25%={p25:.4g}  "
          f"75%={p75:.4g}  99%={p99:.4g}")

    if vstd > 0:
        dr = (p99 - p01) / vstd
        print(f"  Dynamic range: {dr:.1f}σ (p1–p99)")
        info['dynamic_range_sigma'] = dr

    # Recommended stretch
    stretch, reason = auto_stretch(data)
    info['recommended_stretch'] = stretch
    info['stretch_reason'] = reason
    print(f"  Recommended stretch: {stretch}")
    print(f"    Reason: {reason}")

    # For signed data, also suggest make_norm
    if n_valid > 0 and vmin < -3 * vstd:
        info['norm_hint'] = 'symlog'
        print("  Norm hint: make_norm('symlog', data=...) for signed display")

    return info


def rescale_percentile(data: npt.ArrayLike, plo: float = 0.5,
                       phi: float = 99.5, stretch: str = 'linear',
                       fill_nan: float = 0.0,
                       **kwargs: Any) -> npt.NDArray[np.float64]:
    """
    Rescale image to [0, 1] using percentile clipping. Convenience alias.

    Equivalent to ``rescale_image(data, stretch=stretch, clip='percentile',
    plo=plo, phi=phi)``, but with a shorter name for the most common
    use case.

    Parameters
    ----------
    data : ndarray
    plo, phi : float
        Lower and upper percentiles.
    stretch : str
        Stretch function (default 'linear').
    fill_nan : float
        NaN replacement value.
    **kwargs
        Passed to ``rescale_image``.

    Returns
    -------
    scaled : ndarray
        Array in [0, 1].
    """
    return rescale_image(data, stretch=stretch, clip='percentile',
                         plo=plo, phi=phi, fill_nan=fill_nan, **kwargs)


def list_stretches() -> None:
    """Print available stretch function names."""
    stretches = [
        ('linear',   'Identity (no stretch)'),
        ('sqrt',     'Square root'),
        ('squared',  'Squared (power 2)'),
        ('log',      'Logarithmic'),
        ('asinh',    'Inverse hyperbolic sine'),
        ('sinh',     'Hyperbolic sine'),
        ('power',    'Power law (configurable exponent via a=)'),
        ('histeq',   'Histogram equalization (astropy only)'),
        ('symlog',   'Symmetric log (signed data; matplotlib SymLogNorm)'),
        ('symmetric_log',
                     'C¹-continuous symmetric log (requires pysymlog extra)'),
    ]
    print(f"{'Name':<12s} Description")
    print("-" * 50)
    for name, desc in stretches:
        print(f"{name:<12s} {desc}")
