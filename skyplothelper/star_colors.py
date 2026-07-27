"""Perceived star colors from temperature or color index.

:func:`teff_to_rgb`, :func:`bv_to_rgb`, :func:`bp_rp_to_rgb`, and the general
:func:`color_index_to_rgb` convert a stellar effective temperature (K) or a
photometric color index to the RGB color a human eye actually *perceives* —
useful for coloring a star catalog by spectral type so hot stars read
blue-white and cool stars orange-red, the way they look to the eye.

Every color index resolves to an effective temperature and then defers to
:func:`teff_to_rgb`. Johnson ``B-V`` uses the Ballesteros (2012) analytic
relation; the modern survey indices a catalog most often hands you — Gaia
``BP-RP``, SDSS ``g-r``, 2MASS ``J-K`` — are interpolated against the
empirical Pecaut & Mamajek (2013) dwarf color/Teff sequence, so a star's
implied temperature agrees whichever color you arrive with. Reach for
:func:`color_index_to_rgb` when you want to name the index explicitly
(``index='BP-RP'``); :func:`bv_to_rgb` / :func:`bp_rp_to_rgb` are thin
wrappers for the two most common cases.

This is a **tristimulus** conversion (integrating the color-matching functions
via the Planckian locus), not a Wien-peak mapping — so a Sun-temperature star
comes out white, not green. The pipeline:

1. ``B-V`` → effective temperature via the Ballesteros (2012) relation.
2. Temperature → CIE ``(x, y)`` chromaticity via the Kim et al. (2002)
   Planckian-locus polynomials (valid 1667–25000 K; they encode the CMF
   integral).
3. ``(x, y)`` → XYZ (unit luminance) → linear sRGB (D65) → clip → normalize so
   the brightest channel is 1 (color independent of brightness) → desaturate
   toward white → sRGB gamma.

A ``saturation`` knob (default ``0.55``) mixes toward white, since pure
chromaticity is more vivid than real starlight looks. Brightness is deliberately
*not* encoded — the returned colors sit at a common max channel — so you can map
magnitude to marker size / alpha independently.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# CIE XYZ (D65) → linear sRGB.
_XYZ_TO_SRGB = np.array([
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570],
])

# Kim et al. (2002) validity range for the Planckian-locus polynomials.
_TEFF_MIN = 1667.0
_TEFF_MAX = 25000.0


def _planckian_xy(t: npt.NDArray[np.float64],
                  ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """CIE ``(x, y)`` chromaticity on the Planckian locus (Kim et al. 2002)."""
    inv = 1.0 / t
    x = np.where(
        t <= 4000.0,
        (-0.2661239e9 * inv ** 3 - 0.2343589e6 * inv ** 2
         + 0.8776956e3 * inv + 0.179910),
        (-3.0258469e9 * inv ** 3 + 2.1070379e6 * inv ** 2
         + 0.2226347e3 * inv + 0.240390),
    )
    y = np.select(
        [t <= 2222.0, t <= 4000.0],
        [(-1.1063814 * x ** 3 - 1.34811020 * x ** 2
          + 2.18555832 * x - 0.20219683),
         (-0.9549476 * x ** 3 - 1.37418593 * x ** 2
          + 2.09137015 * x - 0.16748867)],
        default=(3.0817580 * x ** 3 - 5.87338670 * x ** 2
                 + 3.75112997 * x - 0.37001483),
    )
    return x, y


def _srgb_gamma(c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Linear → sRGB companding (gamma)."""
    c = np.clip(c, 0.0, None)
    out: npt.NDArray[np.float64] = np.where(
        c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)
    return out


def teff_to_rgb(teff: npt.ArrayLike,
                saturation: float = 0.55) -> npt.NDArray[np.float64]:
    """Perceived RGB color of a blackbody at effective temperature *teff* (K).

    Parameters
    ----------
    teff : float or array_like
        Effective temperature(s) in kelvin. Values outside the
        1667–25000 K validity range are clipped to it.
    saturation : float
        Chromaticity fraction in ``[0, 1]``; the color is mixed
        ``saturation`` of the way from white to full chromaticity (default
        ``0.55``, since real starlight is less vivid than pure locus colors).
        ``1`` = full chromaticity, ``0`` = white.

    Returns
    -------
    numpy.ndarray
        ``(3,)`` for a scalar *teff*, else ``(N, 3)`` of floats in ``[0, 1]``
        (RGB), ready for ``scatter(..., c=...)`` / ``color=``.

    Examples
    --------
    >>> teff_to_rgb(5778)                 # the Sun → white
    array([...])
    >>> teff_to_rgb([10000, 3500])        # blue-white, orange
    array([[...], [...]])
    """
    t = np.asarray(teff, dtype=float)
    scalar = t.ndim == 0
    t = np.clip(np.atleast_1d(t), _TEFF_MIN, _TEFF_MAX)

    x, y = _planckian_xy(t)
    y_safe = np.where(np.abs(y) < 1e-9, 1e-9, y)
    xyz = np.stack([x / y_safe, np.ones_like(x), (1.0 - x - y) / y_safe],
                   axis=-1)                                   # unit-luminance XYZ
    lin = np.clip(xyz @ _XYZ_TO_SRGB.T, 0.0, None)            # (N, 3) linear sRGB

    # Normalize to a common brightness (color carries hue only, not magnitude).
    mx = lin.max(axis=-1, keepdims=True)
    lin = np.divide(lin, mx, out=np.zeros_like(lin), where=mx > 0)

    # Apply the sRGB gamma, THEN desaturate toward white — the order of the
    # validated reference (desaturating in sRGB space, not linear).
    rgb = _srgb_gamma(lin)
    s = float(np.clip(saturation, 0.0, 1.0))
    rgb = np.clip(s * rgb + (1.0 - s), 0.0, 1.0)
    return rgb[0] if scalar else rgb


def bv_to_rgb(bv: npt.ArrayLike,
              saturation: float = 0.55) -> npt.NDArray[np.float64]:
    """Perceived RGB color from a Johnson ``B-V`` color index.

    Converts ``B-V`` to effective temperature via the Ballesteros (2012)
    relation, then defers to :func:`teff_to_rgb`. Vectorized; same return
    shape convention.

    Parameters
    ----------
    bv : float or array_like
        Johnson ``B-V`` color index (bluer/hotter is more negative).
    saturation : float
        Passed through to :func:`teff_to_rgb`.

    Examples
    --------
    >>> bv_to_rgb(0.65)     # Sun-like → white
    array([...])
    >>> bv_to_rgb(1.85)     # Betelgeuse-like → orange
    array([...])
    """
    b = np.asarray(bv, dtype=float)
    # Ballesteros (2012): a two-blackbody fit to the B and V bands.
    teff = 4600.0 * (1.0 / (0.92 * b + 1.7) + 1.0 / (0.92 * b + 0.62))
    return teff_to_rgb(teff, saturation=saturation)


# ---------------------------------------------------------------------------
# Color-index → Teff via the Pecaut & Mamajek (2013) dwarf sequence
# ---------------------------------------------------------------------------
# Johnson B-V keeps the Ballesteros closed form above (a tidy two-blackbody
# analytic fit). Gaia BP-RP, SDSS g-r, and 2MASS J-K have no such clean
# formula, so they are interpolated against the empirical main-sequence
# color/Teff table of Pecaut & Mamajek (2013, ApJS 208, 9; "A Modern Mean
# Dwarf Stellar Color and Effective Temperature Sequence", online version).
# The three indices share one self-consistent sequence, so the implied Teff
# agrees whichever color the user arrives with — and it agrees with the
# Ballesteros B-V path too (the Sun lands at ~5770 K by all four).
#
# Columns: Teff(K), BP-RP, g-r, J-K. `nan` = not tabulated at that type.
# Rows run hot (O5V) to cool (L8V), i.e. Teff descending.
_MAMAJEK_TABLE = """
41400    nan  -0.620    nan
40500    nan  -0.620    nan
39500    nan  -0.620    nan
38300    nan  -0.620    nan
37100    nan  -0.620    nan
36100    nan  -0.620    nan
35100    nan  -0.620    nan
34300    nan  -0.620    nan
33300    nan  -0.620  -0.235
31900    nan  -0.605  -0.230
31400    nan  -0.590  -0.226
29000    nan  -0.540  -0.216
26000    nan  -0.490  -0.207
24500    nan  -0.483  -0.179
20600    nan  -0.475  -0.145
18500    nan  -0.468  -0.131
17000    nan  -0.460  -0.119
16400    nan  -0.437  -0.108
15700    nan  -0.413  -0.094
14500    nan  -0.390  -0.088
14000    nan  -0.360  -0.081
12300    nan  -0.330  -0.064
10700 -0.120  -0.280  -0.034
10400 -0.087  -0.265  -0.023
 9700 -0.037  -0.250  -0.004
 9300  0.005  -0.210   0.005
 8800  0.068  -0.170   0.022
 8600  0.110  -0.140   0.038
 8250  0.166  -0.090   0.059
 8100  0.194  -0.070   0.069
 7910  0.222  -0.050   0.082
 7760  0.263  -0.020   0.095
 7590  0.320   0.030   0.112
 7400  0.327   0.060   0.130
 7220  0.377   0.100   0.141
 7020  0.434   0.140   0.164
 6820  0.490   0.190   0.190
 6750  0.518   0.210   0.199
 6670  0.546   0.240   0.211
 6550  0.587   0.280   0.227
 6350  0.640   0.290   0.256
 6280  0.670   0.320   0.273
 6180  0.694   0.360   0.286
 6050  0.719   0.380   0.299
 5990  0.767   0.390   0.314
 5930  0.784   0.420   0.329
 5860  0.803   0.450   0.349
 5770  0.823   0.476   0.366
 5720  0.832   0.480   0.373
 5680  0.841   0.490   0.379
 5660  0.850   0.500   0.386
 5600  0.869   0.520   0.403
 5550  0.880   0.540   0.407
 5480  0.900   0.570   0.424
 5380  0.950   0.600   0.451
 5270  0.983   0.620   0.478
 5170  1.010   0.690   0.498
 5100  1.100   0.740   0.525
 4830  1.210   0.850   0.596
 4600  1.340   0.990   0.662
 4440  1.430   1.040   0.700
 4300  1.530   1.140   0.749
 4100  1.700   1.260   0.783
 3990  1.730   1.290   0.797
 3930  1.790   1.320   0.811
 3850  1.840    nan    0.817
 3770  1.970    nan    0.828
 3660  2.090    nan    0.836
 3620  2.130    nan    0.835
 3560  2.230    nan    0.834
 3470  2.390    nan    0.833
 3430  2.500    nan    0.831
 3270  2.780    nan    0.827
 3210  2.940    nan    0.837
 3110  3.160    nan    0.865
 3060  3.350    nan    0.892
 2930  3.710    nan    0.917
 2810  4.160    nan    0.957
 2740  4.500    nan    0.969
 2680  4.650    nan    1.003
 2630  4.720    nan    1.072
 2570  4.860    nan    1.127
 2420  5.100    nan    1.155
 2380  4.780    nan    1.190
 2350  4.860    nan    1.245
 2270    nan    nan    1.270
 2160    nan    nan    1.360
 2060    nan    nan    1.470
 1920    nan    nan    1.580
 1870    nan    nan    1.630
 1710    nan    nan    1.668
 1550    nan    nan    1.714
 1530    nan    nan    1.775
 1420    nan    nan    1.785
"""

# Canonical index key ← accepted user spellings (case / separator insensitive;
# the survey name is accepted as a friendly alias).
_COLOR_INDEX_ALIASES = {
    'b-v': 'b-v', 'bv': 'b-v', 'johnson': 'b-v',
    'bp-rp': 'bp-rp', 'bprp': 'bp-rp', 'gaia': 'bp-rp',
    'g-r': 'g-r', 'gr': 'g-r', 'sdss': 'g-r',
    'j-k': 'j-k', 'jk': 'j-k', 'j-ks': 'j-k', 'jks': 'j-k', '2mass': 'j-k',
}
_SUPPORTED_INDICES = ('B-V', 'BP-RP', 'g-r', 'J-K')


def _parse_mamajek() -> tuple[npt.NDArray[np.float64],
                              dict[str, npt.NDArray[np.float64]]]:
    """Parse the embedded table into a Teff column and per-index color columns."""
    teff: list[float] = []
    cols: dict[str, list[float]] = {'bp-rp': [], 'g-r': [], 'j-k': []}
    for line in _MAMAJEK_TABLE.strip().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        t, bp, gr, jk = (float(x) for x in parts[:4])
        teff.append(t)
        cols['bp-rp'].append(bp)
        cols['g-r'].append(gr)
        cols['j-k'].append(jk)
    return (np.asarray(teff),
            {k: np.asarray(v) for k, v in cols.items()})


def _build_anchor(teff: npt.NDArray[np.float64],
                  color: npt.NDArray[np.float64],
                  ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Strictly-increasing ``color`` → ``Teff`` anchors for ``np.interp``.

    Each index is monotonic only over its stellar branch — BP-RP folds past
    the M/L transition, J-K turns blueward through the T dwarfs (methane), and
    g-r plateaus across the O stars. Walking the table hot→cool and keeping a
    row only when its color strictly exceeds the last kept one collapses those
    plateaus and truncates each fold, leaving a clean monotonic map.
    """
    kept_c: list[float] = []
    kept_t: list[float] = []
    last = -np.inf
    for t, c in zip(teff, color):
        if np.isfinite(c) and c > last:
            kept_c.append(float(c))
            kept_t.append(float(t))
            last = c
    return np.asarray(kept_c), np.asarray(kept_t)


_TEFF_COL, _COLOR_COLS = _parse_mamajek()
# index key → (colors ascending, matching Teff) monotonic anchor arrays.
_INDEX_ANCHORS = {k: _build_anchor(_TEFF_COL, v) for k, v in _COLOR_COLS.items()}


def _canon_index(index: str) -> str:
    """Normalize a user-supplied color-index name to its canonical key."""
    key = str(index).strip().lower().replace(' ', '').replace('_', '-')
    if key in _COLOR_INDEX_ALIASES:
        return _COLOR_INDEX_ALIASES[key]
    raise ValueError(
        f"unknown color index {index!r}; supported: "
        + ', '.join(repr(s) for s in _SUPPORTED_INDICES))


def color_index_to_rgb(value: npt.ArrayLike, index: str = 'B-V',
                       saturation: float = 0.55) -> npt.NDArray[np.float64]:
    """Perceived RGB color from a named photometric color index.

    The one entry point for coloring a catalog by whichever color index it
    carries. Each index resolves to an effective temperature and defers to
    :func:`teff_to_rgb`.

    Parameters
    ----------
    value : float or array_like
        Color-index value(s). Bluer/hotter is more negative.
    index : str
        Which index ``value`` is. One of ``'B-V'`` (Johnson, via the
        Ballesteros relation), ``'BP-RP'`` (Gaia), ``'g-r'`` (SDSS/PS1), or
        ``'J-K'`` (2MASS) — the survey name (``'gaia'``, ``'sdss'``,
        ``'2mass'``) is accepted as an alias, and matching is case- and
        separator-insensitive. The three survey indices are interpolated
        against the Pecaut & Mamajek (2013) empirical dwarf sequence; values
        outside a given index's tabulated range are clipped to it.
    saturation : float
        Chromaticity fraction passed through to :func:`teff_to_rgb`.

    Returns
    -------
    numpy.ndarray
        ``(3,)`` for a scalar *value*, else ``(N, 3)`` of floats in ``[0, 1]``.
        A non-finite input color yields a non-finite (masked) row.

    Examples
    --------
    >>> color_index_to_rgb(0.82, index='BP-RP')   # a Gaia solar analog → white
    array([...])
    >>> color_index_to_rgb([0.0, 2.5], index='BP-RP')  # blue-white, deep orange
    array([[...], [...]])

    See Also
    --------
    bp_rp_to_rgb : thin wrapper for ``index='BP-RP'``.
    bv_to_rgb : thin wrapper for ``index='B-V'``.
    teff_to_rgb : the temperature → color primitive all indices defer to.
    """
    canon = _canon_index(index)
    if canon == 'b-v':
        # B-V keeps its dedicated closed form (agrees with the table at the Sun).
        return bv_to_rgb(value, saturation=saturation)

    colors, teffs = _INDEX_ANCHORS[canon]
    v = np.asarray(value, dtype=float)
    scalar = v.ndim == 0
    va = np.atleast_1d(v)
    # Clip into the tabulated color range (outside it the color saturates
    # anyway); np.interp needs an ascending xp, which ``colors`` is by build.
    clipped = np.clip(va, colors[0], colors[-1])
    teff = np.interp(clipped, colors, teffs)
    rgb = teff_to_rgb(teff, saturation=saturation)
    # Mask rows whose input color was non-finite (missing photometry) so
    # callers can spot them — teff_to_rgb would otherwise fold NaN to gray.
    bad = ~np.isfinite(va)
    if bad.any():
        rgb = np.asarray(rgb, dtype=float).copy()
        rgb[bad] = np.nan
    return rgb[0] if scalar else rgb


def bp_rp_to_rgb(bp_rp: npt.ArrayLike,
                 saturation: float = 0.55) -> npt.NDArray[np.float64]:
    """Perceived RGB color from a Gaia ``BP-RP`` color index.

    Thin wrapper for :func:`color_index_to_rgb` with ``index='BP-RP'`` —
    interpolates the Pecaut & Mamajek (2013) dwarf sequence to a temperature,
    then defers to :func:`teff_to_rgb`. Gaia ``BP-RP`` spans a wider range
    than Johnson ``B-V`` (roughly ``-0.1`` to ``5``), so this is *not* the
    same as ``bv_to_rgb(bp_rp)`` — the proper transform avoids over-reddening.

    Parameters
    ----------
    bp_rp : float or array_like
        Gaia ``BP-RP`` (``G_BP - G_RP``) color index.
    saturation : float
        Passed through to :func:`teff_to_rgb`.

    Examples
    --------
    >>> bp_rp_to_rgb(0.82)     # Gaia solar analog → white
    array([...])
    >>> bp_rp_to_rgb(2.5)      # mid-M dwarf → orange-red
    array([...])
    """
    return color_index_to_rgb(bp_rp, index='BP-RP', saturation=saturation)
