"""Tilted globe (orthographic) frames, decorations, and Earth overlays.

The public API includes the frame builder and Euler-angle helpers,
hemisphere-aware plotters, decorations (compass / scale bar / grid /
border), nightshade blending, inset axes, and boundary loaders.
``animate_globe`` and ``animate_blended_globe`` live in
``globe.animation`` but are NOT re-exported here — they're considered
demo / examples helpers.
"""

from .animation import WebPWriter, save_animation
from .baselines import plot_baselines
from .boundaries import (
    fetch_boundary_data,
    load_boundary_data,
    plot_boundaries_globe,
    plot_boundaries_ortho,
    plot_coastlines,
    plot_lakes,
    plot_land,
    plot_rivers,
    plot_tectonic_plates,
    plot_time_zones,
    prepare_earth_data,
    split_segments,
)
from .decorations import (
    add_checkered_border,
    add_compass_rose,
    add_pole_rod,
    add_scale_bar,
    add_scale_bar_curved_parallel,
    add_scale_bar_cylindrical,
    add_surface_compass,
    highlight_great_circle,
    highlight_meridian_tracer,
    plot_ortho_grid,
)
from .frame import (
    TiltedEarthFrame,
    euler_to_fits_ortho,
    make_globe_angles,
    make_globe_frame,
    make_planet_frame,
    quaternion_to_fits_ortho,
)
from .insets import (
    connect_inset_axes,
    mark_inset_axes,
    reproject_inset_axes,
)
from .nightshade import (
    make_nightshade_blend,
    pseudofits_from_image,
)
from .plotting import (
    imscatter,
    imscatter_globe,
    imscatter_rotated,
    plot_contour_globe,
    plot_line_globe,
    plot_pcolormesh_globe,
    plot_scatter_globe,
)
from .spherical import (
    destination_point,
    great_circle_arc,
    great_circle_distance,
    initial_bearing,
    lonlat_to_xyz,
    midpoint,
    orthographic_forward,
    orthographic_inverse,
    orthographic_visibility,
    small_circle,
    xyz_to_lonlat,
)

__all__ = [
    # Frame
    "TiltedEarthFrame", "euler_to_fits_ortho", "quaternion_to_fits_ortho", "make_globe_angles",
    "make_globe_frame", "make_planet_frame",
    # Spherical
    "lonlat_to_xyz", "xyz_to_lonlat",
    "great_circle_distance", "great_circle_arc",
    "midpoint", "initial_bearing", "destination_point", "small_circle",
    "orthographic_visibility", "orthographic_forward", "orthographic_inverse",
    # Decorations
    "highlight_great_circle",
    "highlight_meridian_tracer",
    "plot_ortho_grid",
    "add_checkered_border", "add_compass_rose", "add_pole_rod", "add_surface_compass",
    "add_scale_bar", "add_scale_bar_cylindrical",
    "add_scale_bar_curved_parallel",
    # Plotting
    "plot_scatter_globe", "plot_line_globe",
    "plot_pcolormesh_globe", "plot_contour_globe",
    "imscatter", "imscatter_rotated", "imscatter_globe",
    # Boundaries
    "load_boundary_data", "fetch_boundary_data", "prepare_earth_data",
    "split_segments",
    "plot_coastlines", "plot_lakes", "plot_land", "plot_rivers",
    "plot_tectonic_plates", "plot_time_zones",
    "plot_boundaries_globe", "plot_boundaries_ortho",
    # Nightshade
    "pseudofits_from_image", "make_nightshade_blend",
    # Insets
    "reproject_inset_axes", "mark_inset_axes", "connect_inset_axes",
    # Baselines
    "plot_baselines",
    # Animation save helpers (the animators themselves — animate_globe /
    # animate_blended_globe — are deliberately NOT exported; these generic
    # save helpers are public).
    "WebPWriter",
    "save_animation",
]
