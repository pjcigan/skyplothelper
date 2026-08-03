Globes & planets
================

Orthographic globe views of the celestial sphere and of solid bodies —
sky hemispheres, Earth and planet maps, and physically tilted
orientations. The groups below follow the workflow: build and orient a
globe frame, plot on it hemisphere-aware (far-side handling), draw
Earth/planet surfaces and the day/night nightshade, decorate, and
connect zoom insets. The spherical-geodesy helpers (great-circle
distances, bearings, waypoints) are general-purpose beyond globes.
Narrative: :doc:`/guide/globe`.

.. currentmodule:: skyplothelper

Globe frames
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   make_globe_frame
   make_planet_frame
   make_globe_angles
   TiltedEarthFrame
   euler_to_fits_ortho
   quaternion_to_fits_ortho
   plot_ortho_grid
   highlight_great_circle
   highlight_meridian_tracer

Orthographic geometry
---------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   orthographic_forward
   orthographic_inverse
   orthographic_visibility
   lonlat_to_xyz
   xyz_to_lonlat

Spherical geodesy
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   great_circle_distance
   great_circle_arc
   small_circle
   midpoint
   initial_bearing
   destination_point
   split_segments

Globe plotters
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   plot_scatter_globe
   plot_line_globe
   plot_pcolormesh_globe
   plot_contour_globe
   imscatter
   imscatter_rotated
   imscatter_globe
   plot_baselines

Decorations
-----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   add_checkered_border
   add_compass_rose
   add_surface_compass
   add_pole_rod
   add_scale_bar
   add_scale_bar_cylindrical
   add_scale_bar_curved_parallel

Earth features & nightshade
---------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   load_boundary_data
   fetch_boundary_data
   prepare_earth_data
   plot_coastlines
   plot_land
   plot_lakes
   plot_rivers
   plot_tectonic_plates
   plot_time_zones
   plot_boundaries_globe
   plot_boundaries_ortho
   clip_to_land
   clip_to_ocean
   pseudofits_from_image
   make_nightshade_blend

Inset axes
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   reproject_inset_axes
   mark_inset_axes
   connect_inset_axes

Saving animations
-----------------

General-purpose writers for saving a matplotlib animation — not globe-specific,
but they live here because rotation sequences are the flagship use (the
:doc:`Animations tutorial </tutorials/animations>` covers the workflow).
:func:`save_animation` selects the writer from the output extension, handles a
transparent background, and derives the frame rate from the animation's
interval. :class:`WebPWriter` is the animated-WebP writer behind ``.webp``
output — 8-bit alpha and a seamless loop, and usable directly as a matplotlib
``Animation.save(writer=...)`` writer.

.. autosummary::
   :toctree: generated
   :nosignatures:

   save_animation
   WebPWriter
