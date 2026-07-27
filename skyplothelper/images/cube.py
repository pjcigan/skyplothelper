"""Backend-agnostic spectral data-cube holder shared across cube tools.

:class:`DataCube` is a thin *(data + WCS)* wrapper — never a plotting object —
that carries the FITS-cube plumbing every cube tool otherwise re-implements:
load + squeeze degenerate axes to ``(channel, y, x)``, split the celestial and
1-D spectral sub-WCS, classify the spectral axis (velocity / frequency /
wavelength), and expose the per-channel spectral world values with velocity ↔
frequency ↔ redshift conversion. It follows the same shared-core philosophy as
:class:`~skyplothelper.geometry._projector.Projector`: a vetted, generalized
core that the *backends* (matplotlib :func:`~skyplothelper.channel_map`, the
plotly cube viewer, future moment/PV tools) draw from — rendering stays in the
backends.

It deliberately stays thin: no reprojection, no unit-aware spectral
reprojection, no masking machinery. Those are :mod:`spectral_cube`'s job; a
``spectral_cube.SpectralCube`` is *accepted* as input (duck-typed, never
imported) but never required.

Transforms (:meth:`~DataCube.spectral_bin`, :meth:`~DataCube.spatial_downsample`,
:meth:`~DataCube.smooth`) return a *new* :class:`DataCube`, so pipelines chain::

    cube = sph.DataCube.from_fits("co.fits").spatial_downsample(2).smooth(5)
"""

from __future__ import annotations

import warnings
from typing import Any, NamedTuple

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.constants import c as _C_LIGHT
from astropy.io import fits as pyfits
from astropy.wcs import WCS, FITSFixedWarning

# CTYPE prefixes → spectral axis kind, for label auto-detection.
_VELOCITY_CTYPES = ("VRAD", "VOPT", "VELO", "FELO", "VELOCITY")
_FREQ_CTYPES = ("FREQ",)
_WAVE_CTYPES = ("WAVE", "AWAV")

_DEFAULT_TARGET_UNIT = {"velocity": "km/s", "frequency": "GHz",
                        "wavelength": "um"}
_DEFAULT_FMT = {"velocity": "{:.0f}", "frequency": "{:.3f}",
                "wavelength": "{:.3f}", "redshift": "{:.4f}"}


# ---------------------------------------------------------------------------
# Loading + spectral classification
# ---------------------------------------------------------------------------

def _load_cube(cube: Any,
               header: pyfits.Header | None) -> tuple[Any, pyfits.Header | None]:
    """Resolve ``cube`` (ndarray / HDU / HDUList / path / SpectralCube) to
    ``(data3d, header)``.

    Degenerate axes (a Stokes or single-plane axis wrapping the spectral one,
    as many radio cubes carry on disk) are squeezed away so the result is a
    plain ``(channel, y, x)`` array. An explicit ``header`` overrides whatever
    the input carried.
    """
    # A spectral_cube.SpectralCube (or anything exposing the same .hdu +
    # .spectral_axis surface) — read its HDU, never import spectral_cube.
    if hasattr(cube, "hdu") and hasattr(cube, "spectral_axis"):
        cube = cube.hdu

    loaded_hdr: pyfits.Header | None = None
    if isinstance(cube, str):
        with pyfits.open(cube) as hdul:
            data = np.asarray(hdul[0].data, dtype=float)
            loaded_hdr = hdul[0].header
    elif isinstance(cube, pyfits.HDUList):
        data = np.asarray(cube[0].data, dtype=float)
        loaded_hdr = cube[0].header
    elif isinstance(cube, (pyfits.PrimaryHDU, pyfits.ImageHDU)):
        data = np.asarray(cube.data, dtype=float)
        loaded_hdr = cube.header
    else:
        data = np.asarray(cube, dtype=float)

    hdr = header if header is not None else loaded_hdr
    data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(
            f"a spectral cube must be 3-D (channel, y, x); got shape "
            f"{data.shape} after squeezing degenerate axes.")
    return data, hdr


def _classify_spectral(ctype: str) -> str:
    """Map a spectral CTYPE (e.g. ``'VRAD-LSR'``) to a label kind."""
    c = ctype.upper()
    if c.startswith(_VELOCITY_CTYPES):
        return "velocity"
    if c.startswith(_FREQ_CTYPES):
        return "frequency"
    if c.startswith(_WAVE_CTYPES):
        return "wavelength"
    return "unknown"


# ---------------------------------------------------------------------------
# Spectral / spatial transforms (module-level; reused by DataCube + channel_map)
# ---------------------------------------------------------------------------

def _hanning_smooth(data: npt.NDArray, width: int) -> npt.NDArray:
    """Smooth along the spectral (0th) axis with a normalized Hanning window."""
    w = int(width)
    if w < 3:
        return data
    win = np.hanning(w)
    win = win / win.sum()
    out = np.empty_like(data)
    flat = data.reshape(data.shape[0], -1)
    of = out.reshape(out.shape[0], -1)
    for j in range(flat.shape[1]):
        of[:, j] = np.convolve(flat[:, j], win, mode="same")
    return out


def _block_average(data: npt.NDArray, world: npt.NDArray | None,
                   n: int) -> tuple[npt.NDArray, npt.NDArray | None]:
    """Average consecutive groups of *n* channels (drops any short remainder)."""
    n = int(n)
    nblk = data.shape[0] // n
    if nblk < 1:
        return data, world
    trimmed = data[: nblk * n]
    avg = np.nanmean(trimmed.reshape(nblk, n, *data.shape[1:]), axis=1)
    wout = None
    if world is not None:
        wout = world[: nblk * n].reshape(nblk, n).mean(axis=1)
    return avg, wout


def _block_average_spatial(data: npt.NDArray, factor: int) -> npt.NDArray:
    """Block-mean each plane over ``(y, x)`` by an integer *factor*.

    Trailing rows/columns that don't fill a full block are dropped (so the
    result is exactly ``(nchan, ny // factor, nx // factor)``).
    """
    f = int(factor)
    nchan, ny, nx = data.shape
    ny2, nx2 = ny // f, nx // f
    if ny2 < 1 or nx2 < 1:
        raise ValueError(
            f"spatial_downsample factor {f} is larger than the {ny}×{nx} "
            f"image.")
    trimmed = data[:, : ny2 * f, : nx2 * f]
    return np.nanmean(trimmed.reshape(nchan, ny2, f, nx2, f), axis=(2, 4))


def _downsample_celestial(wcs: Any, factor: int) -> Any:
    """Celestial WCS for a *factor*× block-averaged image (matched to
    :func:`_block_average_spatial`: new pixel *i* centers on old-pixel
    ``i*f + (f-1)/2``)."""
    if wcs is None:
        return None
    f = int(factor)
    new = wcs.deepcopy()
    # FITS CRPIX is 1-based: old_center(p_new) = f*(p_new - 1) + 1 + (f-1)/2.
    new.wcs.crpix = (wcs.wcs.crpix - 1.0 - (f - 1) / 2.0) / f + 1.0
    if wcs.wcs.has_cd():
        new.wcs.cd = wcs.wcs.cd * f
    else:
        new.wcs.cdelt = wcs.wcs.cdelt * f
    return new


def _sync_header_spatial(header: pyfits.Header | None, factor: int,
                         ny: int, nx: int) -> pyfits.Header | None:
    """Best-effort copy of *header* with its celestial scale/reference updated
    for a *factor*× spatial block-average (keeps beam / scale-bar provenance
    consistent). ``.celestial_wcs`` is the authoritative post-transform WCS."""
    if header is None:
        return None
    f = int(factor)
    h = header.copy()
    for ax in ("1", "2"):
        k = "CRPIX" + ax
        if k in h:
            h[k] = (float(h[k]) - 1.0 - (f - 1) / 2.0) / f + 1.0
    cd_keys = [f"CD{i}_{j}" for i in "12" for j in "12" if f"CD{i}_{j}" in h]
    if cd_keys:
        for k in cd_keys:
            h[k] = float(h[k]) * f
    else:
        for k in ("CDELT1", "CDELT2"):
            if k in h:
                h[k] = float(h[k]) * f
    if "NAXIS1" in h:
        h["NAXIS1"] = int(nx)
    if "NAXIS2" in h:
        h["NAXIS2"] = int(ny)
    return h


# ---------------------------------------------------------------------------
# Spectral label formatting
# ---------------------------------------------------------------------------

def _tidy_unit(unit: Any) -> str:
    """A compact unit string for labels (``'km / s'`` → ``'km/s'``)."""
    return str(unit).replace(" / ", "/").replace(" ", "")


def _clean_unit(unit_str: str) -> str:
    """Tidy a raw FITS unit string (``'m.s**-1'`` → ``'m/s'``)."""
    try:
        return _tidy_unit(u.Unit(unit_str))
    except Exception:
        return unit_str


def _spectral_label_value(world: float, world_unit: str, axis_kind: str,
                          mode: str, target_unit: str | None,
                          restfreq: Any, vsys: float | None,
                          ) -> tuple[float, str] | None:
    """Convert one spectral world value to ``(value, unit_str)`` for a label.

    Handles velocity / frequency / wavelength / redshift, converting between
    representations via astropy equivalencies (a rest frequency is needed to
    turn a frequency axis into velocity or redshift).
    """
    if not world_unit:
        return None
    q = world * u.Unit(world_unit)
    kind = axis_kind if mode == "auto" else mode

    try:
        if kind == "velocity":
            tu = u.Unit(target_unit or "km/s")
            if q.unit.is_equivalent(tu):
                val = q.to(tu).value
            elif restfreq is not None:
                val = q.to(tu, equivalencies=u.doppler_radio(restfreq)).value
            else:
                return None
            if vsys is not None:
                val = val - vsys
            return float(val), _tidy_unit(tu)
        if kind == "frequency":
            tu = u.Unit(target_unit or "GHz")
            return float(q.to(tu).value), _tidy_unit(tu)
        if kind == "wavelength":
            tu = u.Unit(target_unit or "um")
            return float(q.to(tu).value), _tidy_unit(tu)
        if kind == "redshift":
            if q.unit.is_equivalent(u.km / u.s):
                z = (q / _C_LIGHT).to(u.dimensionless_unscaled).value
            elif restfreq is not None:
                z = float((restfreq / q).to(u.dimensionless_unscaled).value) - 1.0
            else:
                return None
            return float(z), ""
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# The public cube holder
# ---------------------------------------------------------------------------

class _Unset:
    """Sentinel for :meth:`DataCube._wrap` "keep the template's value" — distinct
    from a real ``None``, which is a meaningful override (e.g. a transform that
    drops the header)."""

    def __repr__(self) -> str:
        return "<unset>"


_UNSET: Any = _Unset()


# Order-appropriate default colormaps: sequential for integrated intensity /
# dispersion, a blue↔orange diverging map for the velocity field (the
# blueshift/redshift convention, symmetric about the systemic velocity).
_MOMENT_CMAP = {0: "sph.deepsky", 1: "sph.diff_blueorange", 2: "sph.dusk"}
_MOMENT_NAME = {0: "integrated intensity", 1: "velocity field",
                2: "velocity dispersion"}
# Non-WCS keys grafted from the source header onto a written moment map (the
# celestial WCS keys come from wcs.to_header(); these carry beam + provenance).
_PROVENANCE_KEYS = ("BMAJ", "BMIN", "BPA", "OBJECT", "TELESCOP", "INSTRUME",
                    "RESTFRQ", "RESTFREQ", "DATE-OBS")


class MomentMap(NamedTuple):
    """A 2-D moment map collapsed from a cube (see :meth:`DataCube.moment`).

    Order ``0`` is the integrated intensity (``∫ I dv``), ``1`` the
    intensity-weighted mean velocity (the velocity field), ``2`` the velocity
    dispersion. A moment map is not a cube, so it's its own light record rather
    than a :class:`DataCube`: the collapsed image, its units string (for a
    colorbar label), the celestial WCS for plotting, the moment order, and the
    source header (kept so :meth:`plot` can draw the beam / scale bar and so
    ``BMAJ`` / ``OBJECT`` / provenance survive a round-trip).

    Construct one from :meth:`DataCube.moment`, or wrap a moment map you
    already have (your own or a pipeline product) with :meth:`from_fits` — then
    :meth:`plot` renders it with order-appropriate defaults.

    Examples
    --------
    >>> import skyplothelper as sph
    >>> cube = sph.DataCube.from_fits('cube.fits')
    >>> m1 = cube.moment1(threshold=3 * rms)   # velocity field (needs a cut!)
    >>> res = m1.plot()                         # sph frame, diverging cmap, beam
    >>> sph.MomentMap.from_fits('mom1.fits', order=1).plot()   # your own map
    """
    data:   npt.NDArray   # (ny, nx) moment image
    units:  str | None    # units (BUNIT×velocity for m0; velocity for m1/m2)
    wcs:    Any           # celestial WCS (None if the cube had no WCS)
    order:  int           # moment order (0 / 1 / 2)
    header: Any = None    # source FITS header (beam / OBJECT / provenance)

    @property
    def name(self) -> str:
        """Human label, e.g. ``'moment 1 (velocity field)'`` — handy for titles."""
        kind = _MOMENT_NAME.get(self.order, "")
        return f"moment {self.order} ({kind})".strip() if kind else \
            f"moment {self.order}"

    @classmethod
    def from_fits(cls, path: str, order: int | None = None, *,
                  units: str | None = None, hdu: int = 0) -> MomentMap:
        """Wrap an existing 2-D moment-map FITS file for plotting.

        For a moment map you already have (computed elsewhere, or a pipeline
        product) — reads the image, celestial WCS, and full header, and tags it
        with *order* so :meth:`plot` picks the right defaults. *units* defaults
        to ``BUNIT``; the retained header lets :meth:`plot` draw the beam.

        *order* may be omitted for a file written by :meth:`to_fits` (it carries
        a ``MOMORDER`` keyword); otherwise pass ``0`` / ``1`` / ``2``.
        """
        with pyfits.open(path) as hdul:
            arr = np.squeeze(np.asarray(hdul[hdu].data, dtype=float))
            hdr = hdul[hdu].header
        if arr.ndim != 2:
            raise ValueError(
                f"a moment map must be 2-D; got shape {arr.shape}.")
        if order is None:
            order = hdr.get("MOMORDER")
            if order is None:
                raise ValueError(
                    "no order given and the file has no MOMORDER keyword; "
                    "pass order=0, 1, or 2.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            try:
                wcs = WCS(hdr).celestial
            except Exception:
                wcs = None
        return cls(arr, units if units is not None else hdr.get("BUNIT"),
                   wcs, int(order), hdr)

    def to_fits(self, path: str, *, overwrite: bool = False) -> None:
        """Write the moment map to a 2-D FITS file.

        The header is the celestial WCS plus the units (``BUNIT``), a
        ``MOMORDER`` keyword (so :meth:`from_fits` can recover the order), a
        descriptive ``BTYPE``, and any beam / ``OBJECT`` / instrument
        provenance grafted from the source header.
        """
        if self.wcs is not None:
            hdr = self.wcs.to_header()
        else:
            hdr = pyfits.Header()
        if self.header is not None:
            for k in _PROVENANCE_KEYS:
                if k in self.header and k not in hdr:
                    hdr[k] = self.header[k]
        if self.units:
            hdr["BUNIT"] = self.units
        hdr["MOMORDER"] = (self.order, "moment order (skyplothelper)")
        hdr["BTYPE"] = self.name
        pyfits.writeto(path, self.data, hdr, overwrite=overwrite)

    def plot(self, ax: Any = None, *, cmap: str | None = None,
             colorbar: bool = True, symmetric: bool | None = None,
             center: float | None = None, title: str | None = None,
             beam: bool = True, scalebar: float | None = None,
             scalebar_label: str | None = None,
             scalebar_kwargs: dict[str, Any] | None = None,
             **quicklook_kwargs: Any) -> Any:
        """Plot the map on an sph frame with order-appropriate defaults.

        Delegates to :func:`~skyplothelper.quicklook_plot`, which recognizes a
        :class:`MomentMap` and applies the moment defaults (order-based
        colormap, a *"Moment N"* corner label, the header beam, a centered
        diverging range for the velocity field). This method adds the
        convenience overrides below and an optional scale bar.

        Parameters
        ----------
        ax : WCSAxes, optional
            Target axes. When ``None``, ``quicklook_plot`` builds an sph frame
            from this map's WCS.
        cmap : str, optional
            Override the order-based default colormap.
        colorbar : bool
            Draw a colorbar labeled with :attr:`units` (default True).
        symmetric : bool, optional
            Force a symmetric diverging range on / off. Default (``None``) is on
            for order 1, off otherwise.
        center : float, optional
            Center value for the symmetric range (e.g. the systemic velocity for
            a mom-1 map in absolute velocity). Default is the map's median.
        title : str, optional
            Plot title. Default is the header ``OBJECT``.
        beam : bool
            Draw the header beam (default True; needs ``BMAJ``/``BMIN``/``BPA``).
        scalebar, scalebar_label, scalebar_kwargs : float, str, dict
            Draw a scale bar of *scalebar* arcsec via
            :func:`~skyplothelper.add_sizebar_asec` (needs the header).
        **quicklook_kwargs
            Forwarded to :func:`~skyplothelper.quicklook_plot` (e.g.
            ``beam_style``, ``mpl_style``, ``grid``, ``vmin`` / ``vmax``).
        """
        from .quicklook import quicklook_plot

        vmin = quicklook_kwargs.pop("vmin", None)
        vmax = quicklook_kwargs.pop("vmax", None)
        # Honor an explicit center / symmetric override here; otherwise leave
        # vmin/vmax None and let quicklook_plot apply its moment-1 default.
        want_sym = (self.order == 1) if symmetric is None else bool(symmetric)
        if (vmin is None and vmax is None and "norm" not in quicklook_kwargs
                and (center is not None or (want_sym and self.order == 1))):
            finite = self.data[np.isfinite(self.data)]
            if finite.size:
                c = float(np.median(finite)) if center is None else float(center)
                half = float(np.percentile(np.abs(finite - c), 99)) or 1.0
                vmin, vmax = c - half, c + half

        result = quicklook_plot(
            self, ax=ax, image=True, contours=False, colorbar=colorbar,
            colormap=cmap if cmap is not None else "sph.deepsky",
            source_name=title, vmin=vmin, vmax=vmax,
            beam_maj=(None if beam else 0.0), **quicklook_kwargs)

        if scalebar is not None and self.header is not None:
            from ..overlays.annotations import add_sizebar_asec
            lbl = (scalebar_label if scalebar_label is not None
                   else f"{scalebar:g}″")
            add_sizebar_asec(result.ax, self.header, float(scalebar), lbl,
                             **(scalebar_kwargs or {}))
        return result


class DataCube:
    """A spectral data cube: ``(channel, y, x)`` data + its celestial/spectral WCS.

    Construct from anything :func:`channel_map`-style tools accept — an ndarray,
    an HDU / HDUList, a FITS path, or a ``spectral_cube.SpectralCube`` — with an
    optional ``header`` override. Degenerate (Stokes / single-plane) axes are
    squeezed away.

    Parameters
    ----------
    data : ndarray, HDU, HDUList, str, or SpectralCube
        The cube (or a path to a FITS file). A 4-D cube with a degenerate
        Stokes/spectral axis is squeezed to ``(channel, y, x)``.
    header : astropy.io.fits.Header or astropy.wcs.WCS, optional
        Header (or WCS) describing the cube; overrides one carried by *data*.
        Needed for the celestial + spectral WCS, spectral labels, and beam /
        scale-bar furniture downstream.

    Attributes
    ----------
    data : ndarray
        The ``(nchan, ny, nx)`` cube.
    nchan, shape : int, tuple
        Channel count and full array shape.
    celestial_wcs : astropy.wcs.WCS or None
        The 2-D celestial sub-WCS (for imshow panels).
    spectral : astropy.wcs.WCS or None
        The 1-D spectral sub-WCS. After a spectral transform, :attr:`world`
        is authoritative and this is the *pre-transform* axis.
    axis_kind : str or None
        ``'velocity'`` / ``'frequency'`` / ``'wavelength'`` / ``'unknown'``.
    world : ndarray or None
        Per-channel spectral world value, in :attr:`world_unit`.
    world_unit : str or None
        Native unit of :attr:`world`.
    bunit : str or None
        The cube's ``BUNIT`` (intensity unit).
    restfreq : astropy.units.Quantity or None
        Rest frequency (``RESTFRQ``), enabling frequency ↔ velocity labels.
    header : astropy.io.fits.Header or None
        The resolved header (best-effort updated by spatial transforms).

    Examples
    --------
    >>> cube = DataCube.from_fits("co.fits")
    >>> small = cube.spatial_downsample(2).spectral_bin(3)
    >>> small.channel(0).shape          # a single 2-D plane
    (128, 128)
    >>> small.spectral_label(0)         # e.g. '-120 km/s'
    """

    # __init__ parses input; transforms build siblings via _wrap (no re-parse),
    # so the load/squeeze/WCS-split cost is paid once per source cube.
    def __init__(self, data: Any, header: pyfits.Header | None = None) -> None:
        if isinstance(header, WCS):
            header = header.to_header()
        arr, hdr = _load_cube(data, header)
        celestial, spectral, world, world_unit, axis_kind, bunit, restfreq = \
            _split_wcs(arr, hdr)
        self._init(arr, hdr, celestial, spectral, world, world_unit,
                   axis_kind, bunit, restfreq)

    def _init(self, data: npt.NDArray, header: pyfits.Header | None,
              celestial: Any, spectral: Any, world: npt.NDArray | None,
              world_unit: str | None, axis_kind: str | None,
              bunit: str | None, restfreq: Any) -> None:
        self.data = data
        self.header = header
        self.celestial_wcs = celestial
        self.spectral = spectral
        self.world = world
        self.world_unit = world_unit
        self.axis_kind = axis_kind
        self.bunit = bunit
        self.restfreq = restfreq
        self.nchan = int(data.shape[0])
        self._vlim_cache: dict[tuple[float, float], tuple[float, float]] = {}

    @classmethod
    def _wrap(cls, template: DataCube, *, data: npt.NDArray,
              header: pyfits.Header | None = _UNSET,
              celestial: Any = _UNSET,
              world: npt.NDArray | None = _UNSET) -> DataCube:
        """Build a sibling cube from *template*, overriding only what changed."""
        obj = cls.__new__(cls)
        obj._init(
            data,
            template.header if header is _UNSET else header,
            template.celestial_wcs if celestial is _UNSET else celestial,
            template.spectral,
            template.world if world is _UNSET else world,
            template.world_unit, template.axis_kind, template.bunit,
            template.restfreq,
        )
        return obj

    @classmethod
    def from_fits(cls, path: str, header: pyfits.Header | None = None,
                  ) -> DataCube:
        """Load a :class:`DataCube` from a FITS file *path*."""
        return cls(path, header=header)

    @property
    def shape(self) -> tuple[int, ...]:
        """The ``(nchan, ny, nx)`` shape of :attr:`data`."""
        return tuple(self.data.shape)

    def __len__(self) -> int:
        return self.nchan

    def __repr__(self) -> str:
        nchan, ny, nx = self.data.shape
        kind = self.axis_kind or "no-spectral-wcs"
        return f"DataCube(nchan={nchan}, ny={ny}, nx={nx}, axis={kind})"

    # -- channel access ----------------------------------------------------
    def channel(self, i: int) -> npt.NDArray:
        """The 2-D plane for channel *i* (supports negative indexing)."""
        return self.data[int(i)]

    def spectral_label(self, i: int, unit: str | None = None,
                       mode: str = "auto", vsys: float | None = None,
                       restfreq: Any = None) -> str | None:
        """Formatted spectral-coordinate label for channel *i* (or ``None``).

        Parameters
        ----------
        i : int
            Channel index.
        unit : str, optional
            Target unit (default per kind: ``km/s`` / ``GHz`` / ``um``).
        mode : str
            ``'auto'`` (use the axis's own kind) or one of ``'velocity'`` /
            ``'frequency'`` / ``'wavelength'`` / ``'redshift'`` to reinterpret.
        vsys : float, optional
            Systemic velocity to subtract (velocity mode).
        restfreq : Quantity, optional
            Rest frequency override (for frequency ↔ velocity/redshift).
        """
        if self.world is None or self.axis_kind is None:
            return None
        kind = self.axis_kind if mode == "auto" else mode
        pair = _spectral_label_value(
            float(self.world[int(i)]), self.world_unit or "", self.axis_kind,
            mode, unit, restfreq if restfreq is not None else self.restfreq,
            vsys)
        if pair is None:
            return None
        val, ustr = pair
        fmt = _DEFAULT_FMT.get(kind, "{:.3g}")
        return f"{fmt.format(val)} {ustr}".strip()

    # -- transforms (return a new DataCube) --------------------------------
    def spectral_bin(self, n: int) -> DataCube:
        """Average every *n* consecutive channels (a new, coarser cube).

        :attr:`world` is re-derived by averaging; :attr:`spectral` stays the
        pre-transform sub-WCS (labels read the authoritative :attr:`world`).
        """
        if int(n) <= 1:
            return self
        data, world = _block_average(self.data, self.world, int(n))
        return DataCube._wrap(self, data=data, world=world)

    def spatial_downsample(self, factor: int) -> DataCube:
        """Block-average each plane over ``(y, x)`` by an integer *factor*.

        The celestial WCS (and, best-effort, the header) are rescaled to match,
        so panels and overlays stay registered.
        """
        if int(factor) <= 1:
            return self
        data = _block_average_spatial(self.data, int(factor))
        celestial = _downsample_celestial(self.celestial_wcs, int(factor))
        header = _sync_header_spatial(self.header, int(factor),
                                      data.shape[1], data.shape[2])
        return DataCube._wrap(self, data=data, header=header,
                              celestial=celestial)

    def smooth(self, width: int, kind: str = "hanning") -> DataCube:
        """Smooth along the spectral axis (a new cube; :attr:`world` unchanged)."""
        if kind != "hanning":
            raise ValueError(f"smooth kind must be 'hanning', got {kind!r}.")
        data = _hanning_smooth(self.data, int(width))
        return DataCube._wrap(self, data=data)

    # -- reductions --------------------------------------------------------
    def moment(self, order: int = 0, *, unit: str | None = None,
               threshold: float | None = None,
               vsys: float | None = None) -> MomentMap:
        """Collapse the cube to a 2-D moment map of the given *order*.

        - ``0`` — integrated intensity ``∫ I dv``. Over a velocity axis with a
          resolvable step the sum is multiplied by ``|Δv|`` (units
          ``BUNIT × velocity``); otherwise a plain channel sum (``BUNIT``).
        - ``1`` — intensity-weighted mean velocity ``∫ I v dv / ∫ I dv``
          (the velocity field), in the spectral unit.
        - ``2`` — velocity dispersion ``√(∫ I (v − M₁)² dv / ∫ I dv)``.

        Parameters
        ----------
        order : int
            Moment order (``0``, ``1``, or ``2``).
        unit : str, optional
            Spectral unit for the moment (e.g. ``'km/s'``); default is the
            cube's native spectral unit.
        threshold : float, optional
            Exclude voxels with intensity below this value before the weighted
            sums. **Moments 1 and 2 are noise-dominated without a cut** (the
            ``∫ I`` denominator approaches zero over noise), so pass a level a
            few × the RMS. This is a plain scalar cut, not a mask — for real
            masking (per-channel masks, morphology) use ``spectral_cube``.
        vsys : float, optional
            Systemic velocity subtracted from a moment-1 field (in *unit*).

        Returns
        -------
        MomentMap
            The 2-D image, its units string, the celestial WCS, and *order*.

        Raises
        ------
        ValueError
            For an unsupported *order*, or a moment 1/2 on a cube with no
            spectral world axis.
        """
        order = int(order)
        if order not in (0, 1, 2):
            raise ValueError(f"moment order must be 0, 1, or 2; got {order}.")

        data = self.data
        if threshold is not None:
            data = np.where(data >= float(threshold), data, np.nan)

        # Spectral world values in the requested unit (else native).
        w: npt.NDArray | None = None
        vel_unit = self.world_unit
        if self.world is not None and self.world_unit:
            if unit is not None:
                try:
                    w = (self.world * u.Unit(self.world_unit)
                         ).to(u.Unit(unit)).value
                    vel_unit = unit
                except Exception:                 # non-convertible → native
                    w = np.asarray(self.world, dtype=float)
            else:
                w = np.asarray(self.world, dtype=float)

        if order == 0:
            dv = 1.0
            integrated = False
            if w is not None and self.axis_kind == "velocity" and len(w) > 1:
                dv = float(np.abs(np.mean(np.diff(w))))
                integrated = True
            img = np.nansum(data, axis=0) * dv
            if self.bunit and integrated:
                units: str | None = f"{self.bunit} {_clean_unit(vel_unit or '')}"
            else:
                units = self.bunit
            return MomentMap(img, units, self.celestial_wcs, 0, self.header)

        # Moments 1 and 2 are intensity-weighted over the spectral axis.
        if w is None:
            raise ValueError(
                f"moment {order} needs a spectral world axis (velocity / "
                f"frequency / wavelength); this cube has none.")
        wcube = w[:, None, None]
        isum = np.nansum(data, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            m1 = np.nansum(data * wcube, axis=0) / isum   # weighted mean
            if order == 1:
                img = m1 - (vsys if vsys is not None else 0.0)
                return MomentMap(img, _clean_unit(vel_unit or ""),
                                 self.celestial_wcs, 1, self.header)
            var = np.nansum(data * (wcube - m1) ** 2, axis=0) / isum
            img = np.sqrt(np.clip(var, 0.0, None))        # dispersion (≥ 0)
        return MomentMap(img, _clean_unit(vel_unit or ""), self.celestial_wcs,
                         2, self.header)

    def moment0(self, unit: str | None = None, threshold: float | None = None,
                vsys: float | None = None) -> MomentMap:
        """Integrated-intensity map — :meth:`moment` with ``order=0``."""
        return self.moment(0, unit=unit, threshold=threshold, vsys=vsys)

    def moment1(self, unit: str | None = None, threshold: float | None = None,
                vsys: float | None = None) -> MomentMap:
        """Velocity-field map — :meth:`moment` with ``order=1``."""
        return self.moment(1, unit=unit, threshold=threshold, vsys=vsys)

    def moment2(self, unit: str | None = None,
                threshold: float | None = None) -> MomentMap:
        """Velocity-dispersion map — :meth:`moment` with ``order=2``."""
        return self.moment(2, unit=unit, threshold=threshold)

    # -- shared normalization ---------------------------------------------
    def vlimits(self, plo: float = 0.5, phi: float = 99.5,
                ) -> tuple[float, float]:
        """Cached global ``(vmin, vmax)`` percentiles over all finite voxels.

        Lets multiple tools (a channel grid and an interactive viewer) share
        one normalization across the whole cube. Cached per ``(plo, phi)``.
        """
        key = (float(plo), float(phi))
        if key not in self._vlim_cache:
            finite = self.data[np.isfinite(self.data)]
            if finite.size == 0:
                self._vlim_cache[key] = (0.0, 1.0)
            else:
                lo, hi = np.percentile(finite, [plo, phi])
                self._vlim_cache[key] = (float(lo), float(hi))
        return self._vlim_cache[key]


def _split_wcs(data: npt.NDArray, header: pyfits.Header | None,
               ) -> tuple[Any, Any, npt.NDArray | None, str | None,
                          str | None, str | None, Any]:
    """From ``(data, header)`` derive the celestial + spectral WCS, the
    per-channel spectral world array, its unit + kind, BUNIT, and rest freq."""
    celestial: Any = None
    spectral: Any = None
    world: npt.NDArray | None = None
    world_unit: str | None = None
    axis_kind: str | None = None
    bunit: str | None = None
    restfreq: Any = None
    if header is None:
        return celestial, spectral, world, world_unit, axis_kind, bunit, restfreq

    nchan = int(data.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        full = WCS(header)
        try:
            celestial = full.celestial
        except Exception:
            celestial = None
        try:
            sp = full.spectral
            spectral = sp if sp.world_n_dim == 1 else None
        except Exception:
            spectral = None
        if spectral is not None:
            try:
                vals = spectral.pixel_to_world_values(np.arange(nchan))
                world = np.asarray(vals, dtype=float)
                world_unit = (spectral.world_axis_units[0] or "").strip()
                axis_kind = _classify_spectral(spectral.wcs.ctype[0])
            except Exception:
                world = None
    bunit = header.get("BUNIT")
    rf = header.get("RESTFRQ", header.get("RESTFREQ"))
    restfreq = float(rf) * u.Hz if rf else None
    return celestial, spectral, world, world_unit, axis_kind, bunit, restfreq
