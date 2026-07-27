"""First-class :class:`Beam` patch for radio beam / optical PSF rendering.

A :class:`Beam` subclasses :class:`matplotlib.patches.Ellipse`, so all
standard patch attributes work as usual (``fc``, ``ec``, ``alpha``,
``lw``, ``zorder``, ``path_effects``, ...). On top of that it adds:

* Beam-specific accessors with semantic names: :attr:`bmaj_pix`,
  :attr:`bmin_pix`, :attr:`bpa_deg`.
* A :attr:`style` switcher — one of ``'ellipse'`` (plain outline),
  ``'crosshair'`` (outline plus short lines along the major and minor
  axes — the publication standard for radio maps), ``'crosshairgrid'``
  (crosshair plus a fine hatch grid for visibility on busy
  backgrounds), ``'hatch'`` (outline with a diagonal hatch fill),
  ``'filled'`` (solid fill), ``'filledgrid'`` (solid fill plus the
  hatch grid). Switching the style rebuilds the crosshair child
  artists and updates the hatch pattern on the parent ellipse in
  place — geometry changes propagate the same way.
* Independent control over the crosshair lines and the grid hatch
  density / marker (see :meth:`set_crosshair` and :meth:`set_grid`).
* Factory classmethods for the common construction paths:
  :meth:`from_header` (reads ``BMAJ`` / ``BMIN`` in degrees and
  ``BPA`` in degrees from a FITS header or :class:`~astropy.wcs.WCS`)
  and :meth:`from_arcsec` (caller provides BMAJ/BMIN in arcsec plus
  the pixel scale directly). :meth:`set_size_arcsec` is the matching
  in-place mutator for the arcsec workflow.

The :class:`BeamStack` helper bundles multiple co-located beams
(e.g. ALMA 12 m + 7 m, VLA A/B/C/D configurations) into a single
add / remove / move unit while keeping every member beam fully
independent.

Position-angle convention
-------------------------
``bpa_deg`` is the FITS ``BPA``: degrees east of north. The class
chooses ``width=bmin_pix``, ``height=bmaj_pix``, ``angle=bpa_deg``
on the underlying :class:`~matplotlib.patches.Ellipse` so that, on
an N-up / E-left image (the standard convention with ``CDELT1 < 0``),
``bpa_deg=0`` puts the major axis along the +y axis (north) and
``bpa_deg=90`` rotates it CCW into −x (east). Callers thinking in
matplotlib's native angle (CCW from +x) can pass ``pa_convention=
'plot'`` to convert at construction time, and can read the
matplotlib-style angle back via :attr:`bpa_plot`.

Hatch / grid colors
-------------------
Matplotlib couples the hatch line color to the patch's edge color
(``rcParams['hatch.color']`` if set, otherwise ``ec``) with no
public per-instance override. So the grid hatch on a Beam inherits
its color from ``ec``; if you need a differently-colored grid you
have to drive ``rcParams['hatch.color']`` globally (or accept
matching ec).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from .._stroke import _stroke_path_effects

_BEAM_STYLES = frozenset({
    'ellipse',
    'crosshair',
    'crosshairgrid',
    'hatch',
    'filled',
    'filledgrid',
})

_PA_CONVENTIONS = {
    # name → degrees to ADD to user input to get internal FITS BPA
    # (degrees east of north, measured CCW from +y on N-up E-left).
    'fits': 0.0,    # FITS BPA standard — no shift
    'astro': 0.0,   # alias for 'fits'
    'iau': 0.0,     # alias for 'fits'
    'plot': -90.0,  # matplotlib angle (CCW from +x) → subtract 90
}


def _convert_pa_to_fits(value: float, convention: str) -> float:
    """Convert a user-supplied PA in *convention* to internal FITS BPA."""
    key = convention.lower()
    if key not in _PA_CONVENTIONS:
        raise ValueError(
            f"pa_convention must be one of {sorted(_PA_CONVENTIONS)}, "
            f"got {convention!r}")
    return float(value) + _PA_CONVENTIONS[key]


def _convert_fits_to_pa(value: float, convention: str) -> float:
    """Inverse of :func:`_convert_pa_to_fits`."""
    return float(value) - _PA_CONVENTIONS[convention.lower()]


class Beam(Ellipse):
    """Radio beam / optical PSF ellipse with style switching.

    Parameters
    ----------
    xy : (float, float) or SkyCoord
        Beam center in axes data coordinates. A scalar
        :class:`~astropy.coordinates.SkyCoord` is also accepted when ``ax=``
        with a WCS is available (it is projected to pixel coordinates), the
        same anchoring the ``Reticle`` and ``Ruler`` overlays support.
    bmaj_pix, bmin_pix : float
        Major and minor FWHM in pixel units.
    bpa_deg : float, optional
        Beam position angle in degrees, interpreted per
        ``pa_convention``. Default ``0``.
    pa_convention : {'fits', 'astro', 'iau', 'plot'}, optional
        Convention used to interpret ``bpa_deg``:

        - ``'fits'`` / ``'astro'`` / ``'iau'`` (default): degrees east
          of north — the FITS ``BPA`` standard.
        - ``'plot'``: degrees CCW from the +x axis — matplotlib's
          native ``Ellipse.angle`` convention. The class converts
          to FITS internally.

        Read the angle back in either convention via :attr:`bpa_deg`
        (FITS) or :attr:`bpa_plot` (matplotlib).
    style : str, optional
        Display style. One of ``'ellipse'`` (default), ``'crosshair'``,
        ``'crosshairgrid'``, ``'hatch'``, ``'filled'``,
        ``'filledgrid'``.
    crosshair_color : color, optional
        Color of the crosshair lines (only drawn for the two
        ``crosshair*`` styles). Defaults to the parent ellipse's
        ``edgecolor``.
    crosshair_lw : float, optional
        Linewidth of the crosshair lines. Defaults to
        ``patch.lw * crosshair_lw_scale``.
    crosshair_ls : str, optional
        Linestyle of the crosshair lines. Default ``'-'``.
    crosshair_lw_scale : float, optional
        Fallback scale for the crosshair linewidth when
        ``crosshair_lw`` is not given. Default ``0.75`` — the look
        used by the radio-quicklook gallery.
    crosshair_length_scale : float, optional
        Half-length of the crosshair along each axis, expressed as a
        fraction of the corresponding FWHM. Default ``0.5`` (each
        crosshair line spans the full FWHM diameter).
    stroke_color : color, optional
        Legibility stroke drawn behind the ellipse edge *and* the
        crosshair lines (via :class:`matplotlib.patheffects.withStroke`).
        Default ``None`` — no stroke. Pass e.g. ``'white'`` to keep the
        beam readable over a busy background.
    stroke_lw : float, optional
        Total stroke width in points (the visible stroke on each side is
        ``(stroke_lw - lw) / 2``). Default ``3.0``; applies only when
        ``stroke_color`` is set.
    grid_marker : str, optional
        Hatch marker character driving the ``*grid`` styles. Must be
        a valid matplotlib hatch character (``'+'``, ``'x'``, ``'o'``,
        ``'O'``, ``'.'``, ``'*'``, ``'/'``, ``'\\\\'``, ``'|'``,
        ``'-'``). Default ``'+'``.
    grid_density : int, optional
        Number of times the marker is repeated to form the hatch
        string — higher means a finer / denser grid. Default ``6``.
    **kwargs
        Forwarded to :class:`matplotlib.patches.Ellipse`. Standard
        patch attributes (``fc``, ``ec``, ``alpha``, ``lw``, ``ls``,
        ``zorder``, ``path_effects``, ...) all work; defaults to
        ``facecolor='none'`` for the typical unfilled beam-marker look.
        An explicit ``hatch=`` **overrides** the pattern implied by
        ``style`` (so ``style='ellipse', hatch='////'`` hatches the
        plain ellipse); without it, ``style='hatch'`` / ``'*grid'`` set
        their own pattern and the plain styles carry none.

    Attributes
    ----------
    bmaj_pix, bmin_pix : float
        Beam major / minor FWHM in pixel units. Settable.
    bpa_deg : float
        Beam PA in FITS convention (degrees east of north). Settable.
    bpa_plot : float
        Beam PA in matplotlib convention (degrees CCW from +x).
        Settable.
    style : str
        Current display style.
    crosshair_lines : list of Line2D
        The crosshair child artists (empty for non-``crosshair*``
        styles). Lets the caller fine-tune the lines beyond
        :meth:`set_crosshair` if needed.

    Examples
    --------
    >>> beam = Beam((50, 50), bmaj_pix=20, bmin_pix=10, bpa_deg=30,
    ...             style='crosshair', ec='k', lw=0.8)
    >>> beam.add_to(ax)

    >>> # From a FITS header (auto-positioned at the axes' lower-left
    >>> # corner with 8% padding):
    >>> beam = Beam.from_header(hdr, ax=ax, style='crosshairgrid',
    ...                          ec='white', lw=1.0)
    >>> beam.add_to(ax)

    >>> # Independent control over every visual component:
    >>> beam = Beam((50, 50), bmaj_pix=40, bmin_pix=20, bpa_deg=30,
    ...             style='crosshairgrid',
    ...             ec='navy', fc='lightsteelblue', alpha=0.6, lw=1.5,
    ...             crosshair_color='crimson', crosshair_lw=1.0,
    ...             crosshair_ls='--',
    ...             grid_marker='x', grid_density=4)
    """

    def __init__(self, xy: Any,
                 bmaj_pix: float, bmin_pix: float, bpa_deg: float = 0.0, *,
                 pa_convention: str = 'fits',
                 style: str = 'ellipse',
                 crosshair_color: Any = None, crosshair_lw: float | None = None,
                 crosshair_ls: str = '-', crosshair_lw_scale: float = 0.75,
                 crosshair_length_scale: float = 0.5,
                 stroke_color: Any = None, stroke_lw: float = 3.0,
                 grid_marker: str = '+', grid_density: int = 6,
                 **kwargs: Any) -> None:
        # Default to an unfilled patch — the typical beam-marker look.
        # Users can override via ``fc=`` / ``facecolor=``.
        # Skipped when the caller passed ``color=``, which sets face AND edge
        # together: setting facecolor as well makes matplotlib emit
        # "Setting the 'color' property will override the edgecolor or
        # facecolor properties" on every beam built that way.
        if 'color' not in kwargs:
            kwargs.setdefault('facecolor', 'none')
        bpa_fits = _convert_pa_to_fits(bpa_deg, pa_convention)
        super().__init__(xy, width=float(bmin_pix), height=float(bmaj_pix),
                         angle=bpa_fits, **kwargs)
        self._style = 'ellipse'
        self._crosshair_color = crosshair_color
        self._crosshair_lw = crosshair_lw
        self._crosshair_ls = crosshair_ls
        self._crosshair_lw_scale = float(crosshair_lw_scale)
        self._crosshair_length_scale = float(crosshair_length_scale)
        self._stroke_color = stroke_color
        self._stroke_lw = float(stroke_lw)
        self._grid_marker = str(grid_marker)
        self._grid_density = int(grid_density)
        # An explicit hatch= reaches the Ellipse via kwargs; remember it so
        # the style-driven hatch reset below doesn't silently clear it (a
        # user hatch= wins over the style's implied pattern).
        self._user_hatch = kwargs.get('hatch')
        self._decorations: list[Line2D] = []
        # Optional legibility stroke behind the ellipse edge; the crosshair
        # children pick up the same stroke in _build_decorations. The
        # default stroke_color=None means no stroke (unchanged look).
        _pe = _stroke_path_effects(stroke_color, stroke_lw)
        if _pe is not None:
            self.set_path_effects(_pe)
        self.set_style(style)

    # ----- Semantic accessors ------------------------------------------

    @property
    def bmaj_pix(self) -> float:
        return float(self.height)

    @bmaj_pix.setter
    def bmaj_pix(self, value: float) -> None:
        self.height = float(value)
        self._sync_decorations()

    @property
    def bmin_pix(self) -> float:
        return float(self.width)

    @bmin_pix.setter
    def bmin_pix(self, value: float) -> None:
        self.width = float(value)
        self._sync_decorations()

    @property
    def bpa_deg(self) -> float:
        """Beam PA in FITS convention (degrees east of north)."""
        return float(self.angle)

    @bpa_deg.setter
    def bpa_deg(self, value: float) -> None:
        self.angle = float(value)
        self._sync_decorations()

    @property
    def bpa_plot(self) -> float:
        """Beam PA in matplotlib convention (degrees CCW from +x)."""
        return _convert_fits_to_pa(self.bpa_deg, 'plot')

    @bpa_plot.setter
    def bpa_plot(self, value: float) -> None:
        self.bpa_deg = _convert_pa_to_fits(value, 'plot')

    @property
    def style(self) -> str:
        return self._style

    @property
    def crosshair_lines(self) -> list[Line2D]:
        """The crosshair child :class:`~matplotlib.lines.Line2D` artists.

        Empty for non-``crosshair*`` styles. The list is rebuilt
        on every geometry / style change, so don't cache the
        references across mutations; if you need persistent
        customization beyond what :meth:`set_crosshair` covers,
        re-apply your tweaks after each mutation (or override
        ``_build_decorations``).
        """
        return list(self._decorations)

    @property
    def grid_marker(self) -> str:
        return self._grid_marker

    @grid_marker.setter
    def grid_marker(self, value: str) -> None:
        self._grid_marker = str(value)
        self._apply_hatch_for_style()

    @property
    def grid_density(self) -> int:
        return self._grid_density

    @grid_density.setter
    def grid_density(self, value: int) -> None:
        self._grid_density = int(value)
        self._apply_hatch_for_style()

    # ----- Style management --------------------------------------------

    def set_style(self, style: str) -> None:
        """Switch the beam's display style and rebuild child artists.

        Parameters
        ----------
        style : str
            One of ``'ellipse'``, ``'crosshair'``, ``'crosshairgrid'``,
            ``'hatch'``, ``'filled'``, ``'filledgrid'``.
        """
        if style not in _BEAM_STYLES:
            raise ValueError(
                f"style must be one of {sorted(_BEAM_STYLES)}, "
                f"got {style!r}")
        self._style = style
        self._apply_hatch_for_style()
        # Filled styles default to fc=ec at alpha=0.8 when the caller
        # hasn't already set a non-transparent facecolor — gives the
        # expected "solid filled beam" look from radio publications
        # while still letting an explicit ``fc=`` override stand.
        if style in ('filled', 'filledgrid'):
            fc = self.get_facecolor()
            if len(fc) == 4 and fc[3] == 0:
                self.set_facecolor(self.get_edgecolor())
                if self.get_alpha() is None:
                    self.set_alpha(0.8)
        self._sync_decorations()

    def _apply_hatch_for_style(self) -> None:
        """Set the parent ellipse's hatch pattern per current style.

        An explicit ``hatch=`` passed at construction always wins — matplotlib
        lets it reach the Ellipse, and without this override the style reset
        would silently clear it. Otherwise ``hatch`` style uses a fixed
        diagonal pattern, the two ``*grid`` styles use
        ``grid_marker * grid_density``, and the plain styles carry none.
        """
        # Matplotlib's stub types set_hatch's arg as ``str``, but ``None`` is a
        # valid runtime value that clears the hatch pattern.
        if self._user_hatch is not None:
            self.set_hatch(self._user_hatch)
            return
        style = self._style
        if style == 'hatch':
            self.set_hatch('///')
        elif style in ('crosshairgrid', 'filledgrid'):
            self.set_hatch(self._grid_marker * max(1, self._grid_density))
        else:
            self.set_hatch(None)  # type: ignore[arg-type]

    def set_crosshair(self, *, color: Any = None, lw: float | None = None,
                       ls: str | None = None,
                       lw_scale: float | None = None,
                       length_scale: float | None = None) -> 'Beam':
        """Configure the crosshair lines independently from the patch.

        Any argument left ``None`` keeps its current setting.
        Returns ``self`` for chaining.

        Parameters
        ----------
        color : color, optional
            Crosshair line color. ``None`` (default) means inherit
            from the parent ellipse's ``edgecolor``.
        lw : float, optional
            Crosshair linewidth (absolute, in points). ``None``
            means ``patch.lw * lw_scale``.
        ls : str, optional
            Crosshair linestyle.
        lw_scale : float, optional
            Scale factor used when ``lw`` is ``None`` (resolved
            against the patch's current linewidth).
        length_scale : float, optional
            Half-length of each crosshair as a fraction of its
            corresponding FWHM. ``0.5`` (default) gives a full-FWHM
            diameter.
        """
        if color is not None:
            self._crosshair_color = color
        if lw is not None:
            self._crosshair_lw = float(lw)
        if ls is not None:
            self._crosshair_ls = ls
        if lw_scale is not None:
            self._crosshair_lw_scale = float(lw_scale)
        if length_scale is not None:
            self._crosshair_length_scale = float(length_scale)
        self._sync_decorations()
        return self

    def set_stroke(self, stroke_color: Any,
                   stroke_lw: float | None = None) -> 'Beam':
        """Set or clear the legibility stroke on the ellipse + crosshair.

        Pass ``stroke_color=None`` to remove the stroke. ``stroke_lw``
        keeps its current value when left ``None``. Returns ``self``
        for chaining.
        """
        self._stroke_color = stroke_color
        if stroke_lw is not None:
            self._stroke_lw = float(stroke_lw)
        pe = _stroke_path_effects(self._stroke_color, self._stroke_lw)
        # [] clears any existing effects (set_path_effects(None) is invalid).
        self.set_path_effects(pe if pe is not None else [])
        self._sync_decorations()
        return self

    def set_grid(self, *, marker: str | None = None,
                 density: int | None = None) -> 'Beam':
        """Configure the grid hatch (only visible for ``*grid`` styles).

        Any argument left ``None`` keeps its current value. Returns
        ``self`` for chaining.

        Parameters
        ----------
        marker : str, optional
            Hatch marker character — typically ``'+'`` or ``'x'``.
            (Matplotlib accepts ``'+xoO.*\\\\/|-'`` and silently
            ignores anything else.)
        density : int, optional
            Number of times the marker is repeated to form the hatch
            string. Higher = denser grid.
        """
        if marker is not None:
            self._grid_marker = str(marker)
        if density is not None:
            self._grid_density = int(density)
        self._apply_hatch_for_style()
        return self

    # ----- Decoration rebuild ------------------------------------------

    def _sync_decorations(self) -> None:
        """Replace existing crosshair lines with rebuilt ones.

        If the beam is currently attached to an axes, the old child
        lines are removed and the new ones re-added in place so the
        change is reflected immediately.
        """
        host_axes = self.axes
        for d in self._decorations:
            if getattr(d, 'axes', None) is not None:
                d.remove()
        self._decorations = self._build_decorations()
        if host_axes is not None:
            for d in self._decorations:
                host_axes.add_line(d)

    def _build_decorations(self) -> list[Line2D]:
        """Construct the child Line2D artists for the current style.

        Crosshair lines run through the beam center along the major
        and minor axes, ``length_scale × FWHM`` in each direction
        from the center.
        """
        if self._style not in ('crosshair', 'crosshairgrid'):
            return []

        # ``get_center`` returns a 2-tuple at runtime; matplotlib's stub
        # mis-infers it as a scalar ``float``, so route through ``Any``.
        center: Any = self.get_center()
        cx, cy = float(center[0]), float(center[1])
        bpa_rad = float(np.radians(self.bpa_deg))
        # Major axis direction in pixel space (Ellipse height axis,
        # rotated by ``bpa_deg`` CCW from +x at angle=0 → +y axis).
        maj_dir = (-np.sin(bpa_rad), np.cos(bpa_rad))
        # Minor axis direction (perpendicular to major).
        min_dir = (np.cos(bpa_rad), np.sin(bpa_rad))

        if self._crosshair_color is not None:
            color = self._crosshair_color
        else:
            color = self.get_edgecolor()
        if self._crosshair_lw is not None:
            lw = self._crosshair_lw
        else:
            lw = self.get_linewidth() * self._crosshair_lw_scale
        ls = self._crosshair_ls
        zorder = self.get_zorder() + 1
        half_factor = self._crosshair_length_scale
        stroke_pe = _stroke_path_effects(self._stroke_color, self._stroke_lw)

        lines: list[Line2D] = []
        for direction, fwhm in (
            (maj_dir, self.bmaj_pix),
            (min_dir, self.bmin_pix),
        ):
            half = fwhm * half_factor
            dx = direction[0] * half
            dy = direction[1] * half
            lines.append(Line2D(
                [cx - dx, cx + dx], [cy - dy, cy + dy],
                color=color, lw=lw, ls=ls, zorder=zorder,
                path_effects=stroke_pe))
        return lines

    # ----- Mutation hooks ----------------------------------------------

    def set_center(self, xy: Any) -> None:
        super().set_center(xy)
        self._sync_decorations()

    def set_size_arcsec(self, bmaj_asec: float, bmin_asec: float,
                        pixscale_asec: float) -> 'Beam':
        """Update the beam size from arcsec values given a pixel scale.

        Convenience for the arcsec workflow — mirrors
        :meth:`from_arcsec` for in-place mutation when the user has
        the pixel scale separately (rather than embedded in a header).
        Returns ``self`` for chaining.

        Parameters
        ----------
        bmaj_asec, bmin_asec : float
            New major / minor FWHM in arcseconds.
        pixscale_asec : float
            Pixel scale in arcseconds per pixel. Must be positive.
        """
        if pixscale_asec <= 0:
            raise ValueError(
                f"pixscale_asec must be positive, got {pixscale_asec!r}")
        self.bmin_pix = float(bmin_asec) / float(pixscale_asec)
        self.bmaj_pix = float(bmaj_asec) / float(pixscale_asec)
        return self

    # ----- Axes wiring -------------------------------------------------

    def add_to(self, ax: Any) -> 'Beam':
        """Add the beam patch and any style decorations to *ax* in
        data coordinates.

        Equivalent to ``ax.add_patch(beam)`` plus
        ``ax.add_line(child)`` for each crosshair line. The beam is
        placed at its ``xy`` center; if the user pans / zooms the
        axes, the beam moves with the underlying data. Returns
        ``self`` for chaining.

        For corner-anchored placement that stays in place during
        pan / zoom, see :meth:`add_anchored`.
        """
        ax.add_patch(self)
        for d in self._decorations:
            ax.add_line(d)
        return self

    def add_anchored(self, ax: Any, loc: str | int = 'lower left', *,
                     borderpad: float = 0.4, frameon: bool = False,
                     **offsetbox_kwargs: Any) -> Any:
        """Add the beam to *ax* anchored to a corner of the axes bbox.

        Wraps the beam (and any crosshair child lines) in an
        :class:`~matplotlib.offsetbox.AnchoredOffsetbox`. Unlike
        :meth:`add_to`, the beam stays in the corner during pan /
        zoom — useful for interactive sessions and for static
        figures where you want a beam marker at a fixed offset
        from the axes spines. The beam's ``xy`` center is ignored
        (the wrapping bbox sets the placement); the beam's pixel
        dimensions are interpreted in ``ax.transData`` so the
        beam still scales with the axes' data extent.

        Parameters
        ----------
        ax : matplotlib Axes
        loc : str or int, optional
            Anchor location — same vocabulary as
            :class:`~matplotlib.offsetbox.AnchoredOffsetbox`:
            ``'upper right'``, ``'upper left'``, ``'lower left'``
            (default), ``'lower right'``, ``'right'``, ``'center
            left'``, ``'center right'``, ``'lower center'``,
            ``'upper center'``, ``'center'``.
        borderpad : float, optional
            Padding between the anchor box and the axes spine.
            Default ``0.4``.
        frameon : bool, optional
            Draw a frame around the anchor box. Default ``False``.
        **offsetbox_kwargs
            Forwarded to :class:`AnchoredOffsetbox` (e.g.
            ``prop``, ``bbox_to_anchor``, ``bbox_transform``).

        Returns
        -------
        anchored : matplotlib.offsetbox.AnchoredOffsetbox
            The wrapping artist. The beam itself is one of the
            children of ``anchored.get_child()`` (an
            :class:`~matplotlib.offsetbox.AuxTransformBox`).
        """
        from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox

        aux = AuxTransformBox(ax.transData)
        aux.add_artist(self)
        for d in self._decorations:
            aux.add_artist(d)
        anchored = AnchoredOffsetbox(
            # ``loc`` accepts an int code at runtime; the stub types it ``str``.
            loc=loc, child=aux,  # type: ignore[arg-type]
            borderpad=borderpad, frameon=frameon,
            **offsetbox_kwargs)
        ax.add_artist(anchored)
        return anchored

    def remove(self) -> None:
        for d in self._decorations:
            if getattr(d, 'axes', None) is not None:
                d.remove()
        super().remove()

    # ----- Factories ---------------------------------------------------

    @classmethod
    def from_header(cls, hdr: Any, ax: Any = None,
                    xy: Any = None,
                    style: str = 'ellipse',
                    pad_frac: float = 0.08, **kwargs: Any) -> 'Beam':
        """Build a :class:`Beam` from FITS ``BMAJ`` / ``BMIN`` / ``BPA``.

        Reads beam parameters from the FITS header (where ``BMAJ`` /
        ``BMIN`` are in degrees and ``BPA`` is in degrees east of north)
        and converts to pixel units using the header's pixel scale.

        Parameters
        ----------
        hdr : astropy.io.fits.Header or astropy.wcs.WCS or dict-like
            Header (or WCS) containing ``BMAJ`` / ``BMIN`` / ``BPA``
            and ``CDELT*`` (or equivalent).
        ax : matplotlib Axes, optional
            If supplied and ``xy`` is None, the beam is positioned at
            the axes' lower-left corner with ``pad_frac`` padding (in
            data coordinates).

            **Passing** ``ax=`` **does not draw the beam** — it only resolves
            the position. Call :meth:`add_to` (or :meth:`add_anchored`) to
            attach it, or use the one-call :func:`add_beam`. Building a Beam
            and never adding it is a silent no-op.

            Prefer :meth:`add_to` over a bare ``ax.add_patch(beam)``: the
            latter adds only the ellipse, silently dropping the crosshair /
            grid child artists for the ``crosshair*`` and ``*grid`` styles.
        xy : (float, float), optional
            Explicit center position in data coordinates. Required
            when *ax* is not provided.
        style : str, optional
            Beam style. See :class:`Beam`. Default ``'ellipse'``.
        pad_frac : float, optional
            Fractional padding from the axes' lower-left corner when
            auto-positioning. Default ``0.08``.
        **kwargs
            Forwarded to :class:`Beam`.

        Returns
        -------
        beam : Beam

        Raises
        ------
        KeyError
            If ``BMAJ`` / ``BMIN`` / ``BPA`` cannot be read from *hdr*.
        ValueError
            If the header's pixel scale is non-positive or both *ax*
            and *xy* are omitted.
        """
        from astropy.wcs import WCS

        from ..core.fits_utils import beampars_asec_fromhdr, getdegperpix

        if isinstance(hdr, WCS):
            hdr = hdr.to_header()
        bmaj_asec, bmin_asec, bpa = beampars_asec_fromhdr(hdr)
        deg_per_pix = getdegperpix(hdr)
        asec_per_pix = deg_per_pix * 3600.0
        if asec_per_pix <= 0:
            raise ValueError(
                f"Non-positive pixel scale ({asec_per_pix} arcsec/pix) "
                "— cannot compute beam ellipse size.")
        xy = cls._resolve_xy(xy, ax, pad_frac)
        return cls.from_arcsec(bmaj_asec, bmin_asec, bpa_deg=bpa,
                                pixscale_asec=asec_per_pix,
                                xy=xy, style=style, **kwargs)

    @classmethod
    def from_arcsec(cls, bmaj_asec: float, bmin_asec: float,
                    bpa_deg: float = 0.0, *,
                    pixscale_asec: float,
                    xy: Any = None, ax: Any = None,
                    style: str = 'ellipse', pad_frac: float = 0.08,
                    **kwargs: Any) -> 'Beam':
        """Build a :class:`Beam` from arcsec extents given a pixel scale.

        For when ``BMAJ`` / ``BMIN`` / ``BPA`` are in hand directly
        (e.g. literature values) rather than embedded in a header.

        Parameters
        ----------
        bmaj_asec, bmin_asec : float
            Major / minor FWHM in arcseconds.
        bpa_deg : float, optional
            Position angle in degrees, interpreted per
            ``pa_convention`` (default FITS = east of north).
            Default ``0``.
        pixscale_asec : float
            Pixel scale in arcseconds per pixel.
        xy : (float, float), optional
            Beam center in axes data coordinates. Required if *ax*
            is not provided.
        ax : matplotlib Axes, optional
            If supplied and *xy* is None, the beam is positioned at
            the axes' lower-left corner with *pad_frac* padding.
        style : str, optional
            Beam style. Default ``'ellipse'``.
        pad_frac : float, optional
            Fractional corner padding when auto-positioning. Default
            ``0.08``.
        **kwargs
            Forwarded to :class:`Beam` (``ec``, ``lw``,
            ``pa_convention``, ``crosshair_color`` ...).

        Returns
        -------
        beam : Beam
        """
        if pixscale_asec <= 0:
            raise ValueError(
                f"pixscale_asec must be positive, got {pixscale_asec!r}")
        bmaj_pix = float(bmaj_asec) / float(pixscale_asec)
        bmin_pix = float(bmin_asec) / float(pixscale_asec)
        xy = cls._resolve_xy(xy, ax, pad_frac)
        return cls(xy, bmaj_pix=bmaj_pix, bmin_pix=bmin_pix,
                   bpa_deg=bpa_deg, style=style, **kwargs)

    @staticmethod
    def _resolve_xy(xy: Any, ax: Any,
                    pad_frac: float) -> tuple[float, float]:
        """Pick the beam center: explicit ``xy=`` wins, else infer from
        the axes' lower-left corner with ``pad_frac`` padding.

        ``xy`` is in data (pixel) coordinates, so a SkyCoord anchor is
        projected through the axes' WCS first — the same treatment
        :class:`~skyplothelper.Reticle` and :class:`~skyplothelper.Ruler`
        give their anchors.
        """
        if xy is not None:
            if hasattr(xy, 'transform_to'):  # SkyCoord duck-type
                if ax is None or getattr(ax, 'wcs', None) is None:
                    raise ValueError(
                        "A SkyCoord xy= needs ax= with a WCS so the position "
                        "can be projected to pixel coordinates.")
                px, py = ax.wcs.world_to_pixel(xy)
                return float(px), float(py)
            return xy
        if ax is None:
            raise ValueError(
                "Either ax= or xy= must be provided so the beam has a "
                "center position.")
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        return (xlim[0] + pad_frac * (xlim[1] - xlim[0]),
                ylim[0] + pad_frac * (ylim[1] - ylim[0]))

    @classmethod
    def from_psf_fit(cls, psf_image: npt.ArrayLike,
                     xy: Any = None, ax: Any = None, *,
                     pa_convention: str = 'plot', style: str = 'ellipse',
                     pad_frac: float = 0.08, **kwargs: Any) -> 'Beam':
        """Build a :class:`Beam` by fitting a 2D Gaussian to an image.

        Useful when you have a PSF snapshot (e.g. an optical PSF cutout
        from a bright star, a simulated dirty beam, or a per-pixel
        peak-find around an unresolved source) but no ``BMAJ`` /
        ``BMIN`` / ``BPA`` header cards. The fit returns FWHM major
        and minor axes (in pixel units) plus the position angle,
        which then construct a :class:`Beam` exactly as the
        ``from_arcsec`` / ``from_header`` factories do.

        Parameters
        ----------
        psf_image : 2D ndarray
            Image to fit. NaN values are masked out. The image is
            assumed to contain a single dominant Gaussian centered near
            the image center (no centroid-finding is performed beyond
            seeding the fit at the brightest pixel).
        xy : (float, float), optional
            Center position in axes data coords for the resulting
            :class:`Beam`. Required if *ax* is not provided.
        ax : matplotlib Axes, optional
            If supplied and *xy* is None, the beam is auto-positioned
            at the axes' lower-left corner with *pad_frac* padding.
        pa_convention : {'plot', 'fits', 'astro', 'iau'}, optional
            How to interpret / present the fitted angle. The 2D
            Gaussian fit naturally returns an angle in matplotlib's
            CCW-from-+x convention; default ``'plot'`` keeps that
            interpretation, ``'fits'`` shifts it to FITS BPA
            (degrees east of north) for an N-up E-left image.
        style : str, optional
            Beam style. Default ``'ellipse'``.
        pad_frac : float, optional
            Fractional corner padding when *xy* is None.
        **kwargs
            Forwarded to :class:`Beam`.

        Returns
        -------
        beam : Beam

        Raises
        ------
        ImportError
            If ``astropy.modeling`` is unavailable.
        RuntimeError
            If the fit fails to converge or the image is fully NaN.
        """
        from astropy.modeling import fitting, models

        psf = np.asarray(psf_image, dtype=float)
        if psf.ndim != 2:
            raise ValueError(
                f"psf_image must be 2D, got shape {psf.shape}")
        mask = np.isfinite(psf)
        if not mask.any():
            raise RuntimeError(
                "psf_image is fully non-finite; nothing to fit")

        ny, nx = psf.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        # Seed at the brightest pixel.
        flat_idx = np.argmax(np.where(mask, psf, -np.inf))
        y0_seed, x0_seed = np.unravel_index(flat_idx, psf.shape)
        amp_seed = float(psf[y0_seed, x0_seed])
        # Rough size seed — half the smaller image dimension.
        sig_seed = max(1.0, min(nx, ny) / 8.0)

        model = models.Gaussian2D(
            amplitude=amp_seed,
            x_mean=float(x0_seed), y_mean=float(y0_seed),
            x_stddev=sig_seed, y_stddev=sig_seed, theta=0.0,
        )
        fitter = fitting.LevMarLSQFitter()
        with np.errstate(invalid='ignore'):
            fit = fitter(model, xx[mask], yy[mask], psf[mask])

        # astropy returns x_stddev / y_stddev (sigma) and theta (CCW
        # from +x in radians). Convert to FWHM and pick the larger as
        # the major axis.
        FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
        s_x = abs(float(fit.x_stddev.value))
        s_y = abs(float(fit.y_stddev.value))
        theta_rad = float(fit.theta.value)
        # theta is the angle of the x_stddev axis CCW from +x.
        if s_x >= s_y:
            bmaj_pix = s_x * FWHM
            bmin_pix = s_y * FWHM
            pa_plot_deg = float(np.degrees(theta_rad))
        else:
            bmaj_pix = s_y * FWHM
            bmin_pix = s_x * FWHM
            pa_plot_deg = float(np.degrees(theta_rad)) + 90.0
        # Fold to (-90, 90] so plot conventions stay sane.
        pa_plot_deg = ((pa_plot_deg + 90.0) % 180.0) - 90.0

        xy = cls._resolve_xy(xy, ax, pad_frac)
        return cls(xy, bmaj_pix=bmaj_pix, bmin_pix=bmin_pix,
                   bpa_deg=pa_plot_deg, pa_convention=pa_convention,
                   style=style, **kwargs)

    # ----- PSF inset ---------------------------------------------------

    def add_psf_inset(self, parent_ax: Any, psf_image: npt.ArrayLike, *,
                      size: str | tuple[str, str] = '25%',
                      loc: str = 'upper right', borderpad: float = 0.4,
                      stretch: str = 'asinh',
                      vmin: float | None = None, vmax: float | None = None,
                      cmap: Any = 'viridis',
                      show_beam: bool = True,
                      beam_kwargs: dict[str, Any] | None = None,
                      border: bool = True, border_color: Any = 'k',
                      border_lw: float = 0.6,
                      title: str | None = None,
                      title_kwargs: dict[str, Any] | None = None,
                      origin: str = 'lower') -> Any:
        """Render a PSF (or dirty / clean beam) image as an inset.

        The PSF image is displayed on a small inset axes anchored to
        a corner of *parent_ax*, with an asinh-style stretch by
        default to reveal the sidelobe structure. The :class:`Beam`'s
        FWHM ellipse is overlaid at the PSF center by default — a
        direct visual sanity check that the analytical beam matches
        the actual point-spread function.

        Parameters
        ----------
        parent_ax : matplotlib Axes
            Axes on which to anchor the inset (typically the same
            axes the parent image is drawn on).
        psf_image : 2D ndarray
            PSF / beam image to display. NaN values are passed
            through to ``imshow`` and rendered as transparent.
        size : str or (str, str), optional
            Inset size, forwarded to
            :func:`mpl_toolkits.axes_grid1.inset_locator.inset_axes`.
            String like ``'25%'`` gives a square inset whose side is
            that fraction of the parent's bounding box; tuple lets
            you set width / height separately. Default ``'25%'``.
        loc : str, optional
            :class:`AnchoredOffsetbox` location string (``'upper
            right'``, ``'lower left'``, ...). Default
            ``'upper right'``.
        borderpad : float, optional
            Padding between the inset and the parent axes' spine.
        stretch : str, optional
            Image stretch passed to
            :func:`skyplothelper.images.levels.make_norm`. Default
            ``'asinh'``.
        vmin, vmax : float, optional
            Display limits in PSF data units. If both are None
            (default), an automatic percentile interval is used.
        cmap : str or Colormap, optional
            Colormap for the PSF. Default ``'viridis'``.
        show_beam : bool, optional
            Overlay a copy of this :class:`Beam` (same bmaj / bmin /
            bpa) at the PSF center. Default ``True``.
        beam_kwargs : dict, optional
            Overrides for the overlay :class:`Beam`'s style (e.g.
            ``{'style': 'crosshair', 'ec': 'white', 'lw': 0.8}``).
            Default puts a thin white outline on top of the PSF.
        border : bool, optional
            Draw a thin border around the inset axes. Default
            ``True``.
        border_color, border_lw : color / float, optional
            Border styling.
        title : str, optional
            Title above the inset.
        title_kwargs : dict, optional
            Extra kwargs forwarded to ``inset_ax.set_title``.
        origin : {'lower', 'upper'}, optional
            ``imshow`` origin. Default ``'lower'``.

        Returns
        -------
        inset_ax : matplotlib Axes
            The inset axes — returned for further customization
            (e.g. additional contours, colorbar, overplotted points).
        """
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        from ..images.levels import make_norm

        psf = np.asarray(psf_image)
        if psf.ndim != 2:
            raise ValueError(
                f"psf_image must be 2D, got shape {psf.shape}")
        ny, nx = psf.shape

        if isinstance(size, str):
            width = height = size
        elif isinstance(size, (tuple, list)) and len(size) == 2:
            width, height = size
        else:
            raise ValueError(
                f"size must be a percent-string or (width, height) "
                f"pair, got {size!r}")

        iax = inset_axes(parent_ax, width=width, height=height,
                          loc=loc, borderpad=borderpad)

        if vmin is None and vmax is None:
            norm = make_norm(stretch=stretch, data=psf)
        else:
            norm = make_norm(stretch=stretch, vmin=vmin, vmax=vmax,
                              data=psf)
        iax.imshow(psf, norm=norm, cmap=cmap, origin=origin,
                    extent=(-0.5, nx - 0.5, -0.5, ny - 0.5))
        iax.set_xticks([])
        iax.set_yticks([])
        if border:
            for spine in iax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(border_lw)
        else:
            for spine in iax.spines.values():
                spine.set_visible(False)

        if title is not None:
            tkw = dict(fontsize=8)
            if title_kwargs:
                tkw.update(title_kwargs)
            iax.set_title(title, **tkw)

        if show_beam:
            bkw: dict[str, Any] = dict(ec='white', lw=1.0, style='ellipse')
            # Inherit the parent beam's legibility stroke so the inset copy
            # matches it (unless the caller overrides via beam_kwargs).
            if self._stroke_color is not None:
                bkw['stroke_color'] = self._stroke_color
                bkw['stroke_lw'] = self._stroke_lw
            if beam_kwargs:
                bkw.update(beam_kwargs)
            cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
            Beam((cx, cy),
                 bmaj_pix=self.bmaj_pix,
                 bmin_pix=self.bmin_pix,
                 bpa_deg=self.bpa_deg,
                 **bkw).add_to(iax)

        return iax


class BeamStack:
    """Collection of co-located :class:`Beam` patches.

    Common in radio astronomy publications: ALMA 12 m + 7 m + total
    power combinations, VLA A/B/C/D configurations stacked together,
    multi-frequency observations showing the resolution at each band.
    Each member beam is a full :class:`Beam` instance — independent
    style, color, mutability — but :class:`BeamStack` adds them to /
    removes them from an axes together, and lets you move the whole
    stack with one :meth:`set_center` call.

    The stack draws members in the order they are passed; later beams
    paint on top of earlier ones. The publication convention is
    *largest-first* (so smaller, often filled, beams sit visibly on
    top of the outer outline beams).

    Examples
    --------
    >>> # ALMA 12 m + 7 m + ACA combined synthesised beam
    >>> stack = BeamStack([
    ...     Beam((50, 50), bmaj_pix=30, bmin_pix=22, bpa_deg=20,
    ...          style='ellipse', ec='C3', lw=1.2, label='7 m'),
    ...     Beam((50, 50), bmaj_pix=18, bmin_pix=12, bpa_deg=20,
    ...          style='ellipse', ec='C0', lw=1.2, label='12 m'),
    ...     Beam((50, 50), bmaj_pix=8, bmin_pix=6, bpa_deg=20,
    ...          style='filled', ec='C2', label='12 m + 7 m'),
    ... ])
    >>> stack.add_to(ax)
    >>> ax.legend()           # member ``label=`` kwargs render directly

    >>> # Build all three from a list of kwarg specs, shared xy:
    >>> stack = BeamStack.from_specs(
    ...     [dict(bmaj_pix=30, bmin_pix=22, bpa_deg=20, ec='C3',
    ...           label='7 m'),
    ...      dict(bmaj_pix=18, bmin_pix=12, bpa_deg=20, ec='C0',
    ...           label='12 m'),
    ...      dict(bmaj_pix=8, bmin_pix=6, bpa_deg=20, style='filled',
    ...           ec='C2', label='12 m + 7 m')],
    ...     xy=(50, 50))
    >>> stack.add_to(ax)
    """

    def __init__(self, beams: Iterable[Beam]) -> None:
        self._beams = list(beams)
        bad = [b for b in self._beams if not isinstance(b, Beam)]
        if bad:
            raise TypeError(
                "BeamStack expects an iterable of Beam instances; "
                f"got {[type(b).__name__ for b in bad]}")

    # ----- Container protocol -----------------------------------------

    @property
    def beams(self) -> list[Beam]:
        """Member beams, in draw order (earliest first)."""
        return list(self._beams)

    def __iter__(self) -> Iterator[Beam]:
        return iter(self._beams)

    def __len__(self) -> int:
        return len(self._beams)

    def __getitem__(self, idx: int) -> Beam:
        return self._beams[idx]

    # ----- Axes wiring ------------------------------------------------

    def add_to(self, ax: Any) -> 'BeamStack':
        """Add every member beam (and its crosshair children) to *ax*.

        Beams are added in stack order so later members paint on top.
        Returns ``self`` for chaining.
        """
        for b in self._beams:
            b.add_to(ax)
        return self

    def remove(self) -> None:
        """Remove every member beam from its axes."""
        for b in self._beams:
            if b.axes is not None:
                b.remove()

    # ----- Bulk mutation ----------------------------------------------

    def set_center(self, xy: Any) -> 'BeamStack':
        """Move every member beam to a shared center."""
        for b in self._beams:
            b.set_center(xy)
        return self

    def set_visible(self, visible: bool) -> 'BeamStack':
        """Show / hide every member beam (and its crosshair children)."""
        for b in self._beams:
            b.set_visible(visible)
            for d in b.crosshair_lines:
                d.set_visible(visible)
        return self

    # ----- Factory ----------------------------------------------------

    @classmethod
    def from_specs(cls, specs: Iterable[dict[str, Any]],
                   xy: Any = None, ax: Any = None,
                   pad_frac: float = 0.08,
                   **shared_kwargs: Any) -> 'BeamStack':
        """Build a stack from a list of :class:`Beam` kwarg dicts.

        All members share the same center (``xy`` or auto-derived
        from ``ax``). Per-member kwargs in ``specs`` win over
        ``shared_kwargs`` on key collisions, so you can set defaults
        once and override per beam.

        Parameters
        ----------
        specs : sequence of dict
            Each dict is forwarded as kwargs to :class:`Beam`. At
            minimum each must contain ``bmaj_pix`` and ``bmin_pix``
            (or use ``Beam.from_arcsec`` separately and pass the
            resulting beams to the canonical constructor).
        xy : (float, float), optional
            Shared beam center in axes data coordinates.
        ax : matplotlib Axes, optional
            If provided and *xy* is None, the lower-left corner of
            *ax* (with ``pad_frac`` padding) is used.
        pad_frac : float, optional
            Fractional padding for axes auto-positioning. Default
            ``0.08``.
        **shared_kwargs
            Kwargs applied to every member beam unless overridden in
            a per-member spec (e.g. ``style='ellipse'`` for a
            consistent look across the stack).
        """
        xy = Beam._resolve_xy(xy, ax, pad_frac)
        beams = [Beam(xy=xy, **{**shared_kwargs, **spec})
                  for spec in specs]
        return cls(beams)


def add_beam(ax: Any, hdr: Any = None, *, xy: Any = None,
             style: str = 'ellipse', pad_frac: float = 0.08,
             anchored: bool = False, loc: str | int = 'lower left',
             **kwargs: Any) -> Beam:
    """Build a :class:`Beam` and draw it in one call.

    The function form of the Beam workflow, mirroring
    :func:`~skyplothelper.add_reticle` for :class:`~skyplothelper.Reticle`:
    ``Beam.from_header(hdr, ax=ax, ...).add_to(ax)`` in a single step, so the
    beam can't be built and then silently never added.

    Parameters
    ----------
    ax : WCSAxes or Axes
        Target axes. The beam is positioned in its data coordinates.
    hdr : Header or WCS, optional
        FITS header carrying ``BMAJ`` / ``BMIN`` / ``BPA``. Omit only when
        passing a pre-built beam through ``**kwargs`` is not intended — a
        header (or WCS) is normally required.
    xy : (float, float) or SkyCoord, optional
        Explicit center in data coordinates (a SkyCoord is projected through
        the axes WCS). Defaults to the axes' lower-left corner.
    style : str, optional
        Beam style — see :class:`Beam`. Default ``'ellipse'``.
    pad_frac : float, optional
        Corner padding used when *xy* is not given. Default ``0.08``.
    anchored : bool, optional
        If True, attach via :meth:`Beam.add_anchored` so the beam stays put
        during pan / zoom, instead of moving with the data. Default False.
    loc : str or int, optional
        Corner for ``anchored=True``. Default ``'lower left'``.
    **kwargs
        Forwarded to :class:`Beam`.

    Returns
    -------
    beam : Beam
        The beam, already attached to *ax*.

    Examples
    --------
    >>> sph.add_beam(ax, hdr)                       # lower-left, unfilled
    >>> sph.add_beam(ax, hdr, style='crosshair', ec='white')
    >>> sph.add_beam(ax, hdr, anchored=True, loc='lower right')
    """
    if hdr is None:
        raise TypeError(
            "add_beam: a FITS header (or WCS) carrying BMAJ/BMIN/BPA is "
            "required. For a beam built from explicit sizes use "
            "Beam.from_arcsec(...).add_to(ax).")
    beam = Beam.from_header(hdr, ax=ax, xy=xy, style=style,
                            pad_frac=pad_frac, **kwargs)
    if anchored:
        beam.add_anchored(ax, loc=loc)
    else:
        beam.add_to(ax)
    return beam
