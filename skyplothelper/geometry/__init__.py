"""Spherical region toolkit: shapes, bands, Tissot, CompoundRegion.

All public functions accept either ``(lon, lat)`` floats (degrees) or a
``SkyCoord`` for the center, and either floats or astropy ``Quantity``
values for angular sizes. SkyCoord inputs are auto-transformed to the
WCS native frame before projection.
"""

from .bands import (
    add_frame_band,
    add_great_circle_band,
    add_latitude_band,
    add_longitude_band,
    add_lonlat_box,
)
from .compound import CompoundRegion
from .shapes import (
    add_annulus,
    add_ellipse,
    add_geodesic_circle,
    add_rectangle,
    add_spherical_polygon,
    add_square,
    ellipse,
    geodesic_circle,
    rectangle,
)
from .tissot import tissot

__all__ = [
    # Vertex generators
    "geodesic_circle", "rectangle", "ellipse",
    # Shape renderers
    "add_geodesic_circle", "add_spherical_polygon",
    "add_rectangle", "add_square",
    "add_ellipse", "add_annulus",
    # Band renderers
    "add_latitude_band", "add_longitude_band",
    "add_great_circle_band", "add_frame_band",
    "add_lonlat_box",
    # Tissot
    "tissot",
    # Compound region
    "CompoundRegion",
]
