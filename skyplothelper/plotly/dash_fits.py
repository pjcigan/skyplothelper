"""Optional Dash convenience layer for the FITS image viewer.

Wires the stateless per-view computers (:func:`skyplothelper.plotly.fits_ticks_for_range`
and :func:`~skyplothelper.plotly.beam_shape_for_range`) to a Dash app, so the
WCS tick labels (and optionally a corner-pegged beam) recompute on every zoom /
pan via a ``relayoutData`` callback.

This is the path for a **live, Python-backed** viewer. Note that
``coords='offset'`` figures already get round, zoom-adaptive ticks from
plotly's *native* numeric axes (no callback needed) — so the main thing this
adds is **absolute-mode** RA/Dec tick labels that follow the view, plus an
optional dynamically corner-pegged beam.

``dash`` is an optional dependency: ``pip install dash`` or
``pip install skyplothelper[dash]``. Importing this module does not require
``dash`` (it is imported lazily on first use), so ``import skyplothelper``
stays light.

Example
-------
::

    from astropy.io import fits
    from astropy.wcs import WCS
    from skyplothelper.plotly import dash_fits

    hdr = fits.getheader('image.fits'); data = fits.getdata('image.fits')
    app = dash_fits.fits_viewer_app(data, WCS(hdr), coords='absolute',
                                    stretch='asinh', header=hdr)
    app.run(debug=True)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _import_dash() -> tuple[Any, Any, Any, Any, Any]:
    """Import Dash lazily with a friendly error if it isn't installed."""
    try:
        import dash
        from dash import Input, Output, dcc, html
    except ImportError as exc:
        raise ImportError(
            "skyplothelper.plotly.dash_fits requires the optional `dash` "
            "package. Install with `pip install dash` or "
            "`pip install skyplothelper[dash]`.") from exc
    return dash, dcc, html, Input, Output


def _ranges_from_relayout(
    relayout_data: dict[str, Any] | None, default_xrange: Sequence[float],
    default_yrange: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Resolve ``(xrange, yrange)`` from a plotly ``relayoutData`` dict.

    Handles explicit ``axis.range[0]`` / ``[1]`` zoom events, the
    ``axis.range`` list form, autorange / reset events (fall back to the
    supplied defaults), and partial updates (only one axis changed). Pure —
    no Dash needed — so it is directly unit-testable.
    """
    xr = list(default_xrange)
    yr = list(default_yrange)
    if not relayout_data:
        return xr, yr
    rd = relayout_data
    for axis, cur in (('xaxis', xr), ('yaxis', yr)):
        if rd.get(f'{axis}.autorange'):
            cur[:] = (default_xrange if axis == 'xaxis' else default_yrange)
        elif f'{axis}.range[0]' in rd and f'{axis}.range[1]' in rd:
            cur[:] = [float(rd[f'{axis}.range[0]']),
                      float(rd[f'{axis}.range[1]'])]
        elif f'{axis}.range' in rd:
            cur[:] = [float(v) for v in rd[f'{axis}.range']]
    return xr, yr


def register_fits_relayout(app: Any, graph_id: str, wcs: Any, *,
                           coords: str = 'absolute',
                           ref_coord: Sequence[float] | None = None,
                           offset_units: str = 'auto',
                           default_xrange: Sequence[float],
                           default_yrange: Sequence[float],
                           beam: dict[str, Any] | None = None,
                           beam_shape_index: int | None = None,
                           max_ticks: int = 6) -> Any:
    """Register a Dash callback that recomputes WCS ticks (+ optional beam)
    from the current view and patches the ``dcc.Graph`` figure.

    Parameters
    ----------
    app : dash.Dash
        The Dash app to register the callback on.
    graph_id : str
        ``id`` of the ``dcc.Graph`` holding the FITS figure.
    wcs : astropy.wcs.WCS
        Image WCS (reduced to 2-D celestial internally).
    coords : {'absolute', 'offset'}
        Tick mode. For ``'offset'`` the native numeric axes already adapt on
        zoom, so tick patching is skipped (the callback only updates the beam,
        if any). For ``'absolute'`` it recomputes RA/Dec labels per view.
    ref_coord : (ra, dec) or None
        Offset reference (deg); defaults to ``CRVAL``.
    offset_units : {'auto', 'arcsec', 'arcmin', 'mas', 'uas'}
    default_xrange, default_yrange : (lo, hi)
        The initial full-view ranges, used on autorange / reset events.
    beam : dict or None
        If given, ``dict(bmaj_arcsec=, bmin_arcsec=, bpa_deg=, corner=)`` —
        the beam is recomputed corner-pegged to the current view each event.
    beam_shape_index : int or None
        Index of the beam shape in ``figure.layout.shapes`` to replace. Defaults
        to the last shape when ``beam`` is given.
    max_ticks : int
        Target number of ticks per axis.

    Returns
    -------
    The registered callback function.
    """
    dash, dcc, html, Input, Output = _import_dash()
    from .fits import _celestial_wcs, beam_shape_for_range, fits_ticks_for_range
    wcs2d = _celestial_wcs(wcs)
    dxr, dyr = list(default_xrange), list(default_yrange)

    @app.callback(Output(graph_id, 'figure'),
                  Input(graph_id, 'relayoutData'),
                  prevent_initial_call=True)
    def _on_relayout(relayout_data: dict[str, Any] | None) -> Any:
        xr, yr = _ranges_from_relayout(relayout_data, dxr, dyr)
        patch = dash.Patch()
        if coords == 'absolute':
            ticks = fits_ticks_for_range(
                wcs2d, xr, yr, coords='absolute', ref_coord=ref_coord,
                offset_units=offset_units, max_ticks=max_ticks)
            patch['layout']['xaxis']['tickvals'] = ticks['xaxis']['tickvals']
            patch['layout']['xaxis']['ticktext'] = ticks['xaxis']['ticktext']
            patch['layout']['yaxis']['tickvals'] = ticks['yaxis']['tickvals']
            patch['layout']['yaxis']['ticktext'] = ticks['yaxis']['ticktext']
        if beam is not None:
            shape = beam_shape_for_range(wcs2d, xr, yr, **beam)
            idx = beam_shape_index if beam_shape_index is not None else -1
            patch['layout']['shapes'][idx] = shape
        return patch

    return _on_relayout


def fits_viewer_app(data: Any, wcs: Any, *, coords: str = 'absolute',
                    graph_id: str = 'sph-fits-graph',
                    ref_coord: Sequence[float] | None = None,
                    offset_units: str = 'auto',
                    beam: dict[str, Any] | None = None,
                    title: str | None = None, theme: str = 'light',
                    width: int = 720, height: int = 720,
                    **add_image_kwargs: Any) -> Any:
    """Build a ready-to-run Dash app: one ``dcc.Graph`` of the FITS image with
    the WCS-tick relayout callback wired in.

    Parameters
    ----------
    data, wcs : the image array and its astropy WCS.
    coords : {'absolute', 'offset'}
    graph_id : str
        ``dcc.Graph`` id.
    ref_coord, offset_units, beam :
        Forwarded to :func:`register_fits_relayout`. ``beam`` as a dict also
        drives the initial static beam via ``add_fits_image`` (its keys map to
        ``beam_maj`` / ``beam_min`` / ``beam_pa``).
    title, theme, width, height :
        Figure scaffold options.
    **add_image_kwargs :
        Forwarded to :func:`~skyplothelper.plotly.add_fits_image` (``stretch``,
        ``colormap``, ``colorbar``, ``header``, ``display_factor``,
        ``field_size``, ``hover``, ...).

    Returns
    -------
    dash.Dash
        The app. Call ``app.run(...)`` to serve it.
    """
    dash, dcc, html, Input, Output = _import_dash()
    from .fits import add_fits_image, make_fits_figure

    if beam is not None:
        add_image_kwargs.setdefault('beam_maj', beam.get('bmaj_arcsec'))
        add_image_kwargs.setdefault('beam_min', beam.get('bmin_arcsec'))
        add_image_kwargs.setdefault('beam_pa', beam.get('bpa_deg', 0.0))

    fig = make_fits_figure(wcs, theme=theme, width=width, height=height,
                           title=title)
    add_fits_image(fig, data, wcs, coords=coords, ref_coord=ref_coord,
                   offset_units=offset_units, **add_image_kwargs)

    xr = [float(v) for v in fig.layout.xaxis.range]
    yr = [float(v) for v in fig.layout.yaxis.range]

    app = dash.Dash(__name__)
    app.layout = html.Div([dcc.Graph(id=graph_id, figure=fig)])
    register_fits_relayout(
        app, graph_id, wcs, coords=coords, ref_coord=ref_coord,
        offset_units=offset_units, default_xrange=xr, default_yrange=yr,
        beam=beam,
        beam_shape_index=(len(fig.layout.shapes) - 1) if beam else None)
    return app
