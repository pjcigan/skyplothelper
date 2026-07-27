Frames & figures
================

Frame construction — the starting point of every plot. The master
builder :func:`~skyplothelper.make_wcs_frame` handles any projection and
frame shape; the figure builders below it wrap the common cases in one
call. The projection registry enumerates what's available, and
:func:`~skyplothelper.project` is the canonical ``(lon, lat) → (x, y)``
primitive that the whole package (both backends) projects through. The
synthetic ("dummy") headers supply ready-made WCS for layout work and
testing. Narrative: :doc:`/guide/frames`; package-wide conventions
(longitude direction, centering, units): :doc:`/guide/concepts`.

.. currentmodule:: skyplothelper

Frame builders
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   make_wcs_frame
   clip_to_frame
   apply_boundary_labels
   WCS_to_offsetWCS
   offset_coord_WCS

Dummy headers
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   dummy_allsky_hdr
   dummy_ortho_hdr
   dummy_offset_hdr
   dummy_standard_hdr

Figure builders
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   allsky_figure
   offset_figure
   projection_gallery

Projection registry & projection primitive
-------------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   list_projections
   get_frame_class
   project
   project_to_canvas
   clip_to_projection_boundary

Plotting on sky axes
--------------------

Thin wrappers around the matplotlib ``Axes`` methods that accept sky
coordinates (a ``SkyCoord`` or ``(lon, lat)`` degrees, honoring ``frame=``)
instead of pixels, projecting through the same primitive and splitting lines
at the antimeridian. :func:`~skyplothelper.world_transform` is the underlying
``(lon, lat) → display`` transform for a WCSAxes.

.. autosummary::
   :toctree: generated
   :nosignatures:

   plot
   scatter
   errorbar
   step
   fill
   fill_between
   annotate
   text
   contour
   contourf
   tricontourf
   pcolormesh
   hist2d
   world_transform
