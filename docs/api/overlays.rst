Overlays
========

The annotation layer — everything drawn *onto* a frame around the data:
coordinate planes and great circles, survey footprints, IAU
constellations, beams, rulers, reticles, instrument markers, compasses
and scale bars, and second-coordinate-system grids. All of it projects
through the shared pipeline (seam-aware in every projection), and most
helpers accept ``stroke_color=``/``stroke_lw=`` for a legibility stroke on
busy backgrounds. Narrative: :doc:`/guide/overlays` (annotations) and
:doc:`/guide/ticks` (overlay grids and ticks).

.. currentmodule:: skyplothelper

Coordinate planes & great circles
---------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   add_plane_overlay
   add_great_circle

Survey footprints
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   add_survey_footprint
   list_surveys
   survey_keys

Constellations
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   add_constellation_boundaries
   add_constellation_lines
   add_constellation_labels
   add_constellation_polygon
   list_constellations

Beams, rulers & reticles
------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   Beam
   add_beam
   BeamStack
   Ruler
   Reticle
   add_reticle

Instrument markers
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   add_antenna_marker
   add_telescope_marker
   add_dome_marker
   aim_angles
   MarkerAnchors

Annotations & scale bars
------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   add_sizebar
   add_sizebar_asec
   add_compass
   add_axis_inlay
   add_bandlabels
   add_colorbar
   add_contour_overlay

Coordinate overlays & grids
---------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   CoordinateOverlay
   add_coord_overlay
   add_graticule_overlay
   add_overlay_ticks
   add_second_grid
   style_grid
   highlight_gridline
   highlight_gridlines
