Interactive (plotly)
====================

.. currentmodule:: skyplothelper.plotly

The ``skyplothelper.plotly`` subpackage is the interactive web-export
backend (conventionally ``import skyplothelper.plotly as sphpl``). It
mirrors the matplotlib overlay surface against plotly figures — same
names, same projection pipeline, same geometry — and adds a WCS-aware
FITS image viewer with a ready-made Dash app. Figures remember their
projection setup, so the ``add_*`` helpers don't need it repeated. The
API here is function-shaped by design: plotly's added shapes are
effectively immutable, so configuration happens at call time rather
than through mutable objects. Requires the ``plotly`` extra (``dash``
for the viewer app). Narrative: :doc:`/guide/plotly`.

Figures & projection
--------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   make_figure
   project

Data overlays
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   add_scatter
   add_healpix
   add_healpix_sparse
   add_sky_vectors
   add_legend

Constellations
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   add_constellation_boundaries
   add_constellation_lines
   add_constellation_labels
   add_constellation_polygon

Lines, circles & regions
------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   add_great_circle
   add_plane_overlay
   add_geodesic_circle
   add_spherical_polygon
   add_lonlat_box
   add_frame_band
   add_great_circle_band
   make_compound_region
   add_compound_region
   add_region_slider
   compound_region_states

Decorations
-----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   add_coord_labels
   add_frame_edge
   add_reticle
   add_ruler

FITS image viewer
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   make_fits_figure
   add_fits_image
   add_fits_scatter
   make_fits_compound_region
   fits_ticks_for_range
   beam_shape_for_range

Dash apps
---------

The complete interactive viewers are packaged as ready-to-run Dash apps: the
FITS image stack in the ``dash_fits`` submodule, and the live compound-region
explorer (the kernel-backed counterpart of :func:`add_region_slider`) in
``dash_region``.

.. autosummary::
   :toctree: generated

   dash_fits
   dash_region
