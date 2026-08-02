"""skyplothelper — Astronomy visualization toolkit.

WCS frames, spherical geometry, tilted globes, cone (z-RA) frames, spectral
cubes, HEALPix, and catalog plotting, for both matplotlib and plotly.

**New here (human or AI agent)? Call ``sph.overview()``** for the frame-first
model, the coordinate conventions, and a task→function index, or
``sph.recipes('<keyword>')`` (e.g. ``sph.recipes('cube')``) for copy-paste
recipes — including the "how do I adjust it" ones (grid spacing, tick format,
strokes).

The package is FRAME-FIRST: create a sky frame with ``make_wcs_frame`` (flat
all-sky / field), ``make_globe_frame`` / ``make_planet_frame`` (globes), or
``make_cone_frame`` (redshift wedges), then draw data and decorations onto that
frame's axes. The full public API is re-exported here from the submodules below.
"""

from . import plotly  # plotly extras live behind a lazy import internally
from ._overview import overview, recipes
from ._timeinput import to_time
from ._version import __version__
from .autosize import auto_size_ticklabels

# Public API re-exports. Imports are sorted alphabetically by ruff's I001.
from .cartopy_backend import (
    cartopy_figure,
    list_cartopy_projections,
    make_cartopy_frame,
)
from .catalog import (
    cone_search,
    crossmatch,
    region_search,
)
from .colormaps import (
    get_colormap,
    list_colormaps,
    show_colormaps,
)
from .cone import (
    add_minor_rticks,
    cone_hexbin,
    cone_pcolormesh,
    cone_plot,
    cone_scatter,
    cone_scatter_z,
    flip_label,
    get_label_pad,
    log_r,
    make_bowtie_frame,
    make_cone_frame,
    make_twinr,
    redshift_to_r,
    set_label_pad,
)
from .constants import (
    FACILITY_RESOLUTION,
    FILTER_BANDS,
    RADIO_BANDS,
    REGION_PALETTE,
    REGION_PALETTE_NAMED,
    SEPARATORS,
    SKY_POSITIONS,
    obliquities,
    planet_radii,
    rot_periods,
)
from .coord_overlay import (
    CoordinateOverlay,
    add_coord_overlay,
    add_graticule_overlay,
    add_overlay_ticks,
)
from .core.coords import (
    RAcosDEC_err,
    angulardistance,
    convert_frame,
    dec2sex,
    deg2dms,
    deg2hour,
    dms2deg,
    ecliptic_to_galactic,
    ecliptic_to_icrs,
    galactic_to_ecliptic,
    galactic_to_icrs,
    hour2deg,
    icrs_to_ecliptic,
    icrs_to_galactic,
    icrs_to_supergalactic,
    sex2dec,
    supergalactic_to_icrs,
)
from .core.fits_utils import (
    beampars_asec_fromhdr,
    convpix2sky,
    convsky2pix,
    force_hdr_floats,
    force_hdr_to_2D,
    force_hdr_to_3D,
    getasecperpix,
    getcdelts,
    getcdmatrix,
    getdegperpix,
    getsteradperpix,
    header_coord_grids,
    makesimpleheader,
    pixperbeam_from_hdr,
    pixperbeam_from_pars,
    squeeze_image,
)
from .core.math_utils import (
    map_to_newrange,
    rescale_data_range,
    wrap_24hr,
    wrap_360,
    wrap_center_pmrange,
    wrap_pm90,
    wrap_pm180,
    wrap_pmPI,
    wrap_range,
)
from .data_plots import (
    CatalogPlot,
    SkyVectorResult,
    plot_catalog,
    plot_displacement,
    plot_sky_vectors,
    sky_quiverkey,
)
from .diagnostics import (
    describe_wcs,
    saved_plot_size_reducer,
)
from .figures import (
    allsky_figure,
    offset_figure,
    projection_gallery,
)
from .geometry import (
    CompoundRegion,
    add_annulus,
    add_ellipse,
    add_frame_band,
    add_geodesic_circle,
    add_great_circle_band,
    add_latitude_band,
    add_longitude_band,
    add_lonlat_box,
    add_rectangle,
    add_spherical_polygon,
    add_square,
    ellipse,
    geodesic_circle,
    rectangle,
    tissot,
)
from .globe import (
    TiltedEarthFrame,
    WebPWriter,
    add_checkered_border,
    add_compass_rose,
    add_pole_rod,
    add_scale_bar,
    add_scale_bar_curved_parallel,
    add_scale_bar_cylindrical,
    add_surface_compass,
    connect_inset_axes,
    destination_point,
    euler_to_fits_ortho,
    fetch_boundary_data,
    great_circle_arc,
    great_circle_distance,
    highlight_great_circle,
    highlight_meridian_tracer,
    imscatter,
    imscatter_globe,
    imscatter_rotated,
    initial_bearing,
    load_boundary_data,
    lonlat_to_xyz,
    make_globe_angles,
    make_globe_frame,
    make_nightshade_blend,
    make_planet_frame,
    mark_inset_axes,
    midpoint,
    orthographic_forward,
    orthographic_inverse,
    orthographic_visibility,
    plot_baselines,
    plot_boundaries_globe,
    plot_boundaries_ortho,
    plot_coastlines,
    plot_contour_globe,
    plot_lakes,
    plot_land,
    plot_line_globe,
    plot_ortho_grid,
    plot_pcolormesh_globe,
    plot_rivers,
    plot_scatter_globe,
    plot_tectonic_plates,
    plot_time_zones,
    prepare_earth_data,
    pseudofits_from_image,
    quaternion_to_fits_ortho,
    reproject_inset_axes,
    save_animation,
    small_circle,
    split_segments,
    xyz_to_lonlat,
)
from .grid import (
    add_second_grid,
    highlight_gridline,
    highlight_gridlines,
    style_grid,
)
from .healpix import (
    HealpixBins,
    HealpixResult,
    auto_nside,
    bin_data_as_healpix,
    bin_data_sparse,
    healpix_allsky_figure,
    healpix_circle_query,
    healpix_combine,
    healpix_downgrade,
    healpix_pixel_corners,
    healpix_polygon_query,
    healpix_smooth,
    healpix_to_celestial,
    healpix_upgrade,
    image_to_healpix,
    mask_seam_crossing_quads,
    nside_from_array,
    plot_healpix_allsky,
    plot_healpix_map,
    plot_healpix_sparse,
    sources_to_healpix_bins,
    sources_to_healpix_plot,
)
from .images.channels import (
    ChannelMapResult,
    channel_map,
)
from .images.cube import (
    DataCube,
    MomentMap,
)
from .images.levels import (
    adjust_gamma,
    auto_interval,
    auto_stretch,
    clip_percentile,
    clip_sigma,
    clip_zscale,
    describe_image,
    list_stretches,
    make_norm,
    rescale_image,
    rescale_percentile,
)
from .images.quicklook import (
    QuicklookResult,
    SimpleImageResult,
    quicklook_figure,
    quicklook_fits,
    quicklook_plot,
    simpleimage_figure,
    simpleimageplot,
)
from .images.reprojection import (
    load_sky_image,
    reproject_background,
    reproject_rgb_map,
)
from .legend import (
    AlphaBlock,
    ColorbarBlock,
    ColorBlock,
    EdgeBlock,
    FillBlock,
    GlyphBlock,
    LegendBlock,
    LineBlock,
    MultiLegend,
    OrientBlock,
    RegionBlock,
    ShapeBlock,
    SizeBlock,
    TextBlock,
    list_glyphs,
    register_glyph,
)
from .overlays.annotations import (
    add_axis_inlay,
    add_bandlabels,
    add_colorbar,
    add_compass,
    add_contour_overlay,
    add_sizebar,
    add_sizebar_asec,
)
from .overlays.beam import Beam, BeamStack, add_beam
from .overlays.constellations import (
    add_constellation_boundaries,
    add_constellation_labels,
    add_constellation_lines,
    add_constellation_polygon,
    list_constellations,
)
from .overlays.instruments import (
    MarkerAnchors,
    add_antenna_marker,
    add_dome_marker,
    add_telescope_marker,
    aim_angles,
)
from .overlays.planes import (
    add_great_circle,
    add_plane_overlay,
)
from .overlays.reticle import Reticle, add_reticle
from .overlays.ruler import Ruler
from .overlays.surveys import (
    add_survey_footprint,
    list_surveys,
    survey_keys,
)
from .plotting import (
    annotate,
    contour,
    contourf,
    errorbar,
    fill,
    fill_between,
    hist2d,
    pcolormesh,
    plot,
    scatter,
    step,
    text,
    to_lonlat,
    tricontourf,
    world_transform,
)
from .projections.canvas import healpix_to_canvas, project_to_canvas
from .projections.project import project
from .projections.registry import get_frame_class, list_projections
from .queries import (
    download_hips,
    download_skyview,
    list_skyview_surveys,
    overlay_cutout,
    query_ned,
    query_simbad,
    resolve_name,
    resolve_names,
    search_vizier,
)
from .star_colors import (
    bp_rp_to_rgb,
    bv_to_rgb,
    color_index_to_rgb,
    teff_to_rgb,
)
from .style import (
    ANNOTATION_PALETTES,
    BASE_PRESETS,
    CYCLE_PALETTES,
    FONT_PRESETS,
    MONO_STACK,
    apply_frame_stroke,
    set_base_style,
    set_font,
    set_palette,
    set_style,
    set_theme,
    style_annotation,
    style_context,
    style_wcs_axes,
)
from .ticks import (
    AnchoredOffsetFormatter,
    OffsetFormatter,
    RAlabelformatter,
    RAlabellist,
    add_curved_lon_ticks,
    apply_anchored_offset,
    apply_offset_ticks,
    format_mpl_ticklabels,
    format_ticklabels,
    format_WCS_ticklabels,
)
from .visibility import (
    covisibility_circles,
    covisibility_duration_band,
    covisibility_region,
)
from .vsh import VSH_PARAM_NAMES, vsh_field, vsh_shift_frame, vsh_shift_sources
from .wcs_frame import (
    WCS_to_offsetWCS,
    apply_boundary_labels,
    clip_to_frame,
    clip_to_projection_boundary,
    dummy_allsky_hdr,
    dummy_offset_hdr,
    dummy_ortho_hdr,
    dummy_standard_hdr,
    make_wcs_frame,
    offset_coord_WCS,
)

__all__ = [
    "__version__",
    "overview", "recipes",
    "auto_size_ticklabels",
    # constants
    "obliquities", "rot_periods", "planet_radii",
    "SKY_POSITIONS", "RADIO_BANDS", "FILTER_BANDS",
    "FACILITY_RESOLUTION", "SEPARATORS",
    "REGION_PALETTE", "REGION_PALETTE_NAMED",
    # math_utils
    "wrap_360", "wrap_pm180", "wrap_pm90", "wrap_pmPI", "wrap_24hr",
    "wrap_range", "wrap_center_pmrange",
    "map_to_newrange", "rescale_data_range",
    # coords
    "deg2dms", "dms2deg", "deg2hour", "hour2deg",
    "dec2sex", "sex2dec",
    "angulardistance", "RAcosDEC_err",
    "convert_frame",
    "icrs_to_galactic", "galactic_to_icrs",
    "icrs_to_ecliptic", "ecliptic_to_icrs",
    "galactic_to_ecliptic", "ecliptic_to_galactic",
    "icrs_to_supergalactic", "supergalactic_to_icrs",
    # fits_utils
    "getcdelts", "getdegperpix", "getasecperpix", "getsteradperpix",
    "getcdmatrix",
    "beampars_asec_fromhdr", "pixperbeam_from_hdr", "pixperbeam_from_pars",
    "makesimpleheader", "force_hdr_to_2D", "force_hdr_to_3D", "force_hdr_floats",
    "convsky2pix", "convpix2sky", "header_coord_grids",
    "squeeze_image",
    # projections
    "list_projections", "get_frame_class",
    "project_to_canvas", "healpix_to_canvas", "project",
    # vector spherical harmonics
    "vsh_field", "vsh_shift_sources", "vsh_shift_frame", "VSH_PARAM_NAMES",
    # mutual sky visibility (co-visibility)
    "covisibility_circles", "covisibility_region", "covisibility_duration_band",
    # ticks
    "format_ticklabels", "format_WCS_ticklabels", "format_mpl_ticklabels",
    "RAlabelformatter", "RAlabellist",
    "OffsetFormatter", "AnchoredOffsetFormatter",
    "apply_offset_ticks", "apply_anchored_offset",
    "add_curved_lon_ticks",
    # wcs_frame
    "make_wcs_frame", "clip_to_projection_boundary",
    "dummy_allsky_hdr", "dummy_ortho_hdr", "dummy_offset_hdr", "dummy_standard_hdr",
    "clip_to_frame", "apply_boundary_labels",
    "WCS_to_offsetWCS", "offset_coord_WCS",
    # === overlays ===
    "add_plane_overlay", "add_great_circle",
    "add_survey_footprint", "list_surveys", "survey_keys",
    "add_constellation_boundaries", "add_constellation_labels",
    "add_constellation_lines", "add_constellation_polygon",
    "list_constellations",
    "Beam", "BeamStack",
    "Ruler",
    "Reticle", "add_reticle",
    "add_antenna_marker", "add_telescope_marker", "add_dome_marker",
    "MarkerAnchors",
    "aim_angles",
    "add_sizebar", "add_sizebar_asec",
    "add_compass", "add_axis_inlay", "add_bandlabels",
    "add_colorbar", "add_contour_overlay",
    # images
    "clip_percentile", "clip_sigma", "clip_zscale", "auto_interval",
    "rescale_image", "rescale_percentile",
    "make_norm", "adjust_gamma",
    "auto_stretch", "describe_image", "list_stretches",
    "quicklook_plot", "quicklook_figure", "quicklook_fits", "QuicklookResult",
    "simpleimageplot", "simpleimage_figure", "SimpleImageResult",
    "channel_map", "ChannelMapResult", "DataCube", "MomentMap",
    "load_sky_image", "reproject_background", "reproject_rgb_map",
    # healpix
    "bin_data_as_healpix", "bin_data_sparse", "image_to_healpix",
    "nside_from_array", "HealpixBins",
    "plot_healpix_allsky", "healpix_allsky_figure", "HealpixResult",
    "plot_healpix_sparse", "plot_healpix_map",
    "healpix_circle_query", "healpix_polygon_query",
    "auto_nside", "healpix_pixel_corners",
    "healpix_smooth", "healpix_upgrade", "healpix_downgrade", "healpix_combine",
    "sources_to_healpix_bins", "sources_to_healpix_plot",
    "healpix_to_celestial", "mask_seam_crossing_quads",
    # style/grid/figures/data_plots/cartopy/queries/diagnostics
    "set_theme", "set_base_style", "style_context", "set_style", "set_palette",
    "set_font", "style_wcs_axes", "apply_frame_stroke", "style_annotation",
    "CYCLE_PALETTES",
    "ANNOTATION_PALETTES", "BASE_PRESETS", "FONT_PRESETS", "MONO_STACK",
    "add_second_grid", "style_grid", "highlight_gridline", "highlight_gridlines",
    "CoordinateOverlay", "add_coord_overlay", "add_graticule_overlay",
    "add_overlay_ticks",
    "allsky_figure", "offset_figure", "projection_gallery",
    "plot_sky_vectors", "SkyVectorResult", "sky_quiverkey",
    "plot_displacement", "plot_catalog", "CatalogPlot", "add_beam",
    # convenience plotting wrappers (sky transform + SkyCoord + seam split)
    "scatter", "plot", "text", "annotate", "errorbar", "fill",
    "fill_between", "step", "contour", "contourf", "pcolormesh",
    "tricontourf", "hist2d", "world_transform", "to_lonlat", "to_time",
    "MultiLegend", "LegendBlock", "ColorBlock", "ShapeBlock", "LineBlock",
    "SizeBlock", "EdgeBlock", "FillBlock", "AlphaBlock", "OrientBlock",
    "RegionBlock", "TextBlock", "ColorbarBlock", "GlyphBlock",
    "register_glyph", "list_glyphs",
    "make_cartopy_frame", "cartopy_figure", "list_cartopy_projections",
    "resolve_name", "resolve_names",
    "query_simbad", "query_ned",
    "download_skyview", "download_hips", "list_skyview_surveys",
    "search_vizier", "overlay_cutout",
    "cone_search", "region_search", "crossmatch",
    "list_colormaps", "get_colormap", "show_colormaps",
    "teff_to_rgb", "bv_to_rgb", "bp_rp_to_rgb", "color_index_to_rgb",
    "describe_wcs", "saved_plot_size_reducer",
    # === cone ===
    "make_cone_frame", "make_bowtie_frame",
    "make_twinr",
    "cone_scatter", "cone_plot", "cone_scatter_z",
    "cone_hexbin", "cone_pcolormesh",
    "add_minor_rticks", "log_r",
    "flip_label", "set_label_pad", "get_label_pad",
    "redshift_to_r",
    # === globe ===
    "TiltedEarthFrame", "euler_to_fits_ortho", "quaternion_to_fits_ortho", "make_globe_angles", "make_globe_frame",
    "make_planet_frame",
    "lonlat_to_xyz", "xyz_to_lonlat",
    "great_circle_distance", "great_circle_arc",
    "midpoint", "initial_bearing", "destination_point", "small_circle",
    "orthographic_visibility", "orthographic_forward", "orthographic_inverse",
    "highlight_great_circle",
    "highlight_meridian_tracer",
    "plot_ortho_grid",
    "add_checkered_border", "add_compass_rose", "add_pole_rod", "add_surface_compass",
    "add_scale_bar", "add_scale_bar_cylindrical",
    "add_scale_bar_curved_parallel",
    "plot_scatter_globe", "plot_line_globe",
    "plot_pcolormesh_globe", "plot_contour_globe",
    "imscatter", "imscatter_rotated", "imscatter_globe",
    "load_boundary_data", "fetch_boundary_data", "prepare_earth_data",
    "split_segments",
    "plot_coastlines", "plot_lakes", "plot_land", "plot_rivers",
    "plot_tectonic_plates", "plot_time_zones",
    "plot_boundaries_globe", "plot_boundaries_ortho",
    "pseudofits_from_image", "make_nightshade_blend",
    "reproject_inset_axes", "mark_inset_axes", "connect_inset_axes",
    "plot_baselines",
    "save_animation", "WebPWriter",
    # === geometry ===
    "geodesic_circle", "rectangle", "ellipse",
    "add_geodesic_circle", "add_spherical_polygon",
    "add_rectangle", "add_square", "add_ellipse", "add_annulus",
    "add_latitude_band", "add_longitude_band",
    "add_great_circle_band", "add_frame_band",
    "add_lonlat_box",
    "tissot",
    "CompoundRegion",
    # === Submodules ===
    "plotly",
]
