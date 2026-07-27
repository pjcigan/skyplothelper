"""Optional Dash convenience layer for the compound-region explorer.

The genuinely-live counterpart to the static
:func:`skyplothelper.plotly.add_region_slider`. Both consume the same
``region_factory`` a notebook writes once::

    def region_factory(**params):
        return (make_compound_region(fig)
                .add_frame_band(-params["band_hw"], params["band_hw"],
                                frame="galactic")
                .add_circle(*CYG_A, 8).add_circle(*CEN_A, 8))

Feed it to :func:`add_region_slider` for a precomputed, kernel-free slider
that renders in the Sphinx docs / nbviewer, or to
:func:`region_explorer_app` here for a live Dash app whose every slider drag
re-runs ``region_factory`` in real Python — recomputing the set algebra and
the per-source containment — and repaints the region + reclassified sources.

Why a live app is the only way to get a *continuous* region slider: a pure
analytic region (``|b| < w``) could be recomputed in client-side JS, but set
algebra would mean porting the whole compound-region boundary tracer and
point-in-region engine to JavaScript. So the split is precomputed states
(:func:`add_region_slider`) for the static page, live Dash here for real-time.
This needs a running kernel — it cannot embed in the static docs or nbviewer
(both kernel-free), only local / Binder / Colab.

``dash`` is an optional dependency: ``pip install dash`` or
``pip install skyplothelper[dash]``. Importing this module does not require
``dash`` (it is imported lazily on first use), so ``import skyplothelper``
stays light.

The factory takes a ``fig`` argument so it accumulates geometry in the app's
own figure (whose projector defines the canvas coords). The app builds that
figure and passes it back on every call.

Example
-------
::

    import numpy as np
    from skyplothelper.plotly import make_compound_region
    from skyplothelper.plotly import dash_region

    lon = np.random.default_rng(0).uniform(0, 360, 2000)
    lat = np.random.default_rng(1).uniform(-90, 90, 2000)

    def region_factory(fig, band_hw):
        return make_compound_region(fig).add_frame_band(
            -band_hw, band_hw, frame="galactic")

    app = dash_region.region_explorer_app(
        (lon, lat), region_factory,
        params={"band_hw": (2.0, 40.0, 1.0)},
        projection="AIT", center=180, frame="icrs")
    app.run(debug=True)
"""

from __future__ import annotations

import inspect
from typing import Any

from .dash_fits import _import_dash  # one home for the lazy-dash import

# Distinct plotly marker symbols for ``marker_by`` categories, in assignment
# order. Enough for the handful of source classes a region explorer splits on
# (e.g. VLBI defining vs. standard); wraps if a caller has more.
_MARKER_SYMBOL_CYCLE = ['circle', 'diamond', 'square', 'cross', 'x',
                        'triangle-up', 'star', 'pentagon']


def _slider_id(graph_id: str, name: str) -> str:
    """The ``dcc.Slider`` id for parameter *name* under a given graph."""
    return f"{graph_id}-slider-{name}"


def _symbols_for(marker_by: Any) -> Any:
    """Map a per-source category array to a per-source plotly symbol array.

    ``None`` -> ``None`` (all markers the default circle). Otherwise each
    distinct category is assigned a symbol from :data:`_MARKER_SYMBOL_CYCLE`
    in first-seen order, so the two VLBI classes (say) read as two shapes.
    """
    if marker_by is None:
        return None
    order: dict[Any, str] = {}
    symbols: list[str] = []
    for c in marker_by:
        key = c
        if key not in order:
            order[key] = _MARKER_SYMBOL_CYCLE[len(order)
                                              % len(_MARKER_SYMBOL_CYCLE)]
        symbols.append(order[key])
    return symbols


def register_region_callback(
    app: Any, graph_id: str, fig: Any, region_factory: Any,
    param_names: list[str], catalog: Any = None, *,
    fill_shape_index: int, outline_shape_index: int,
    catalog_trace_index: int | None = None) -> Any:
    """Register the Dash callback that re-runs *region_factory* on every slider
    drag and patches the region shapes + reclassified markers.

    Each parameter contributes one ``dcc.Slider`` (id
    ``f"{graph_id}-slider-{name}"``); their values are the callback Inputs.
    On any change the region is rebuilt against *fig*'s projector, the set
    algebra and per-source containment are recomputed in Python, and a
    ``dash.Patch`` swaps the two ``layout.shapes`` paths and (if a catalog was
    drawn) the marker trace's 0/1 color array — the same three updates a static
    slider step makes, but live.

    Parameters
    ----------
    app : dash.Dash
        The Dash app to register the callback on.
    graph_id : str
        ``id`` of the ``dcc.Graph`` holding the region figure.
    fig : plotly.graph_objects.Figure
        The figure the region is projected against.
    region_factory : callable
        ``region_factory(**params) -> CompoundRegion``, params keyed by
        *param_names*.
    param_names : list of str
        Slider parameter names, in the order their Inputs are wired.
    catalog : SkyCoord or (lon, lat), optional
        Sources to reclassify; must match what was drawn.
    fill_shape_index, outline_shape_index : int
        Indices of the fill / outline shapes in ``figure.layout.shapes``.
    catalog_trace_index : int or None
        Index of the marker trace in ``figure.data`` (``None`` if no catalog).

    Returns
    -------
    The registered callback function.
    """
    dash, dcc, html, Input, Output = _import_dash()
    from .core import compound_region_states

    inputs = [Input(_slider_id(graph_id, name), 'value')
              for name in param_names]

    @app.callback(Output(graph_id, 'figure'), *inputs,
                  prevent_initial_call=True)
    def _on_change(*values: float) -> Any:
        params = dict(zip(param_names, values))
        state = compound_region_states(
            fig, region_factory, [params], catalog)[0]
        patch = dash.Patch()
        patch['layout']['shapes'][fill_shape_index]['path'] = state['fill_path']
        patch['layout']['shapes'][outline_shape_index]['path'] = \
            state['outline_path']
        if catalog_trace_index is not None:
            patch['data'][catalog_trace_index]['marker']['color'] = \
                state['contains_int']
        return patch

    return _on_change


def region_explorer_app(
    catalog: Any, region_factory: Any, *,
    params: dict[str, tuple[float, float, float]],
    projection: str = 'AIT', center: float = 180.0, lat_center: float = 0.0,
    frame: str | None = 'icrs', theme: str = 'dark',
    color: str = 'steelblue', fillcolor: str | None = 'rgba(70,130,180,0.25)',
    width: float = 1.0, opacity: float = 0.4,
    inside_color: str = 'crimson', outside_color: str = '0.5',
    marker_size: float = 4.0, marker_by: Any = None,
    name: str | None = None, graph_id: str = 'sph-region-graph',
    title: str | None = None, fig_width: int = 900, fig_height: int = 500,
) -> Any:
    """Build a ready-to-run Dash app: an all-sky figure with a
    :class:`~skyplothelper.CompoundRegion` that grows under one slider per
    parameter, reclassifying a source catalog inside / outside it live.

    The ``region_factory`` must accumulate geometry against **this app's**
    figure. When it declares a ``fig`` parameter it is called as
    ``region_factory(fig=<the app figure>, **params)`` (the recommended form);
    otherwise it is called as ``region_factory(**params)`` and is responsible
    for building its own region against a figure with identical projection
    metadata.

    Parameters
    ----------
    catalog : SkyCoord or (lon, lat)
        Sources to reclassify. A SkyCoord is used in its own frame; a bare
        ``(lon, lat)`` pair is read in the figure's display *frame*.
    region_factory : callable
        ``region_factory(**params) -> CompoundRegion`` (optionally taking
        ``fig=``), params keyed by *params*.
    params : dict[str, (min, max, step)]
        One continuous slider per entry.
    projection, center, lat_center, frame, theme :
        Forwarded to :func:`~skyplothelper.plotly.make_figure`.
    color, fillcolor, width, opacity, inside_color, outside_color, marker_size, name :
        Region + marker styling, as in :func:`add_region_slider`.
    marker_by : array-like, optional
        Per-source category; each distinct value gets its own marker symbol
        (e.g. VLBI defining vs. standard).
    graph_id : str
        ``dcc.Graph`` id.
    title : str, optional
        Figure title.
    fig_width, fig_height : int
        Figure size.

    Returns
    -------
    dash.Dash
        The app. Call ``app.run(...)`` to serve it.
    """
    dash, dcc, html, Input, Output = _import_dash()
    from .core import (
        _draw_region_and_catalog,
        _project_catalog,
        compound_region_states,
        make_figure,
    )

    fig = make_figure(projection=projection, center=center,
                      lat_center=lat_center, frame=frame, theme=theme,
                      width=fig_width, height=fig_height, title=title)

    param_names = list(params)

    # Bind the factory to this app's figure only when it explicitly declares a
    # ``fig`` parameter — checking the signature (rather than a try/except on
    # TypeError) avoids masking a TypeError raised from inside the factory.
    _takes_fig = 'fig' in inspect.signature(region_factory).parameters

    def _call_factory(**kw: Any) -> Any:
        return region_factory(fig=fig, **kw) if _takes_fig \
            else region_factory(**kw)

    # Initial state: start each slider at its minimum.
    init_params = {name_: rng[0] for name_, rng in params.items()}
    init_state = compound_region_states(
        fig, _call_factory, [init_params], catalog)[0]

    cx = cy = None
    if catalog is not None:
        cx, cy = _project_catalog(fig, catalog)
    fill_idx, outline_idx, cat_idx = _draw_region_and_catalog(
        fig, init_state, cx, cy, color=color, fillcolor=fillcolor, width=width,
        opacity=opacity, inside_color=inside_color, outside_color=outside_color,
        marker_size=marker_size, marker_symbol=_symbols_for(marker_by),
        name=name)

    # One labeled slider per parameter, above the graph.
    slider_rows = []
    for name_ in param_names:
        lo, hi, step = params[name_]
        slider_rows.append(html.Div([
            html.Label(name_),
            dcc.Slider(id=_slider_id(graph_id, name_), min=lo, max=hi,
                       step=step, value=lo,
                       tooltip={'placement': 'bottom',
                                'always_visible': True}),
        ], style={'margin': '0.5em 1em'}))

    app = dash.Dash(__name__)
    app.layout = html.Div(
        slider_rows + [dcc.Graph(id=graph_id, figure=fig)])
    register_region_callback(
        app, graph_id, fig, _call_factory, param_names, catalog,
        fill_shape_index=fill_idx, outline_shape_index=outline_idx,
        catalog_trace_index=cat_idx)
    return app
