"""Plotly export helpers for sky-map projections.

A thin layer that lets users build web-shareable sky plots by reusing
skyplothelper's astropy-WCS-driven projection math
(:func:`skyplothelper.project`) and emitting :mod:`plotly.graph_objects`
traces in linear coords.

This module deliberately stays small. The scope is:

* :func:`make_figure` — :class:`plotly.graph_objects.Figure` scaffold
  pre-configured for sky maps (no axis chrome, equal aspect, sky
  orientation, dark/light theme).
* :func:`project` — re-export of :func:`skyplothelper.project` so the
  same primitive is one ``import`` away in plotly workflows.
* :func:`add_scatter` — projects ``(lons, lats)`` arrays + emits a
  ``go.Scatter`` trace with sensible (RA / Dec) hover defaults.
* :func:`add_healpix` — pre-projects HEALPix tile boundary polygons as
  ``go.Scatter(fill='toself')`` traces with hover-over showing tile
  coords + value. The novel feature — no existing astropy-adjacent
  package does HEALPix + plotly cleanly.
* :func:`add_constellation_boundaries`,
  :func:`add_constellation_lines`,
  :func:`add_constellation_labels` — sky-overlay primitives,
  re-using the same data loaders as the matplotlib side.
* :func:`add_great_circle` /  :func:`add_plane_overlay` — draw
  great circles (and small-circle parallels) for galactic /
  ecliptic / supergalactic / pole-defined frames.
* :func:`add_geodesic_circle` — circles of given angular radius
  on the sphere, with optional fill.
* :func:`add_spherical_polygon` — generic spherical polygon
  with edge densification and wrap-edge splitting.

Mostly **out of scope** for this module: interactive widgets
(sliders, dropdowns, animations), drag-rotate orthographic globes,
arbitrary-cursor lon/lat readout between data points. These are
layered on top of any ``go.Figure`` by users via plotly's own APIs,
and the last would require Scattergeo (with its terrestrial-map
chrome baggage and constrained projection list). The one purpose-built
exception is :func:`add_region_slider` (+ its
:func:`compound_region_states` primitive): a slider that grows a
:class:`~skyplothelper.CompoundRegion` under a parameter cannot be
recomputed in client-side JS — it would mean porting the whole
set-algebra boundary tracer and point-in-region engine — so the
states are precomputed here and the figure stays static. Its live
counterpart lives in the optional :mod:`~skyplothelper.plotly.dash_region`
layer.

Plotly is an optional dependency: ``pip install skyplothelper[plotly]``
or ``pip install plotly``. Import errors are deferred to first use of
this module's functions (you can import ``skyplothelper`` without
plotly installed).
"""

from . import (
    dash_fits,  # optional Dash layer (dash imported lazily)
    dash_region,  # optional Dash layer (dash imported lazily)
)
from .core import (
    add_compound_region,
    add_constellation_boundaries,
    add_constellation_labels,
    add_constellation_lines,
    add_constellation_polygon,
    add_coord_labels,
    add_frame_band,
    add_frame_edge,
    add_geodesic_circle,
    add_great_circle,
    add_great_circle_band,
    add_healpix,
    add_healpix_sparse,
    add_legend,
    add_lonlat_box,
    add_plane_overlay,
    add_region_slider,
    add_reticle,
    add_ruler,
    add_scatter,
    add_sky_vectors,
    add_spherical_polygon,
    compound_region_states,
    make_compound_region,
    make_figure,
    project,
)
from .fits import (
    add_fits_image,
    add_fits_scatter,
    beam_shape_for_range,
    fits_ticks_for_range,
    make_fits_compound_region,
    make_fits_figure,
)

__all__ = [
    'make_figure',
    'project',
    'add_scatter',
    'add_healpix',
    'add_healpix_sparse',
    'add_constellation_boundaries',
    'add_constellation_lines',
    'add_constellation_labels',
    'add_constellation_polygon',
    'add_great_circle',
    'add_plane_overlay',
    'add_geodesic_circle',
    'add_spherical_polygon',
    'add_lonlat_box',
    'add_frame_band',
    'add_great_circle_band',
    'add_sky_vectors',
    'add_coord_labels',
    'add_frame_edge',
    'add_reticle',
    'add_ruler',
    'add_compound_region',
    'make_compound_region',
    'compound_region_states',
    'add_region_slider',
    'add_legend',
    # FITS image viewer
    'make_fits_figure',
    'add_fits_image',
    'add_fits_scatter',
    'make_fits_compound_region',
    'fits_ticks_for_range',
    'beam_shape_for_range',
    'dash_fits',
    'dash_region',
]
