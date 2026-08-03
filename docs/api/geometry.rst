Geometry & regions
==================

Spherical regions in three layers: *constructors* return raw boundary
vertices, *renderers* draw a single projection-aware region onto an axes
(sharing the ``clip=`` / ``backend=`` / ``resolution=`` /
``complement=`` keyword surface), and
:class:`~skyplothelper.CompoundRegion` combines shapes with set algebra
into one renderable, queryable object. Requires the ``geometry`` extra
(shapely). Narrative, including the clip-mode table and geodesic-edge
guidance: :doc:`/guide/regions`.

.. currentmodule:: skyplothelper

Region constructors
-------------------

Pure-geometry constructors that return vertex arrays.

.. autosummary::
   :toctree: generated
   :nosignatures:

   geodesic_circle
   rectangle
   ellipse

Region renderers
----------------

Helpers that draw a region onto an axes. They share a common
``clip=`` / ``backend=`` / ``resolution=`` / ``complement=`` surface (see
:doc:`/guide/regions`); ``geodesic=`` additionally tunes edge interpolation
on :func:`~skyplothelper.add_spherical_polygon`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   add_geodesic_circle
   add_spherical_polygon
   add_rectangle
   add_square
   add_ellipse
   add_annulus
   add_latitude_band
   add_longitude_band
   add_great_circle_band
   add_frame_band
   add_lonlat_box
   tissot
   choropleth

Compound regions
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   CompoundRegion
