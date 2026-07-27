"""Agent- and human-facing capability map for skyplothelper.

This is the single source of truth for the package's task→function recipes.
:func:`overview` and :func:`find` read it at runtime (so an agent driving a
REPL gets oriented on its first exploratory call), and the docs generator
renders it into ``llms.txt`` / ``llms-full.txt`` (so those never drift from the
code). Every ``code`` snippet is verified runnable by ``tests/test_overview.py``.

The most important thing to convey — to a human or an AI agent — is the
**frame-first model** (:data:`FRAME_FIRST`) and the coordinate
:data:`CONVENTIONS`; those are the top first-attempt mistakes.
"""

from __future__ import annotations

from typing import Any, NamedTuple

FRAME_FIRST = """\
skyplothelper is FRAME-FIRST. You almost never call matplotlib directly:
you create a *sky frame* (a WCSAxes with the right projection + coordinate
grid), then draw data and decorations ONTO that frame's axes.

    import skyplothelper as sph
    ax = sph.make_wcs_frame(111, 'AIT', frame='ICRS', center=0)   # 1. a frame
    sph.plot_catalog(ax, catalog, ra_col='ra', dec_col='dec')     # 2. data onto it
    ax.legend(); plt.savefig('sky.png')                           # 3. decorate/save

`make_wcs_frame` is the one entry point for flat all-sky and field frames;
`make_globe_frame` (sky globes) and `make_planet_frame` (Earth/planets) build
3D-looking orthographic frames; `make_cone_frame` builds redshift/velocity
(z vs RA) wedges. Data-plotting and overlay helpers all take that `ax` first."""

CONVENTIONS = [
    "Coordinates are astronomical/sky by default: longitude increases to the "
    "LEFT (east-left). Only make_planet_frame (and cartopy frames) default to "
    "geographic (east-right). Do NOT flip a mirrored-looking Earth by changing "
    "defaults — use make_planet_frame.",
    "Projections use FITS codes ('AIT','MOL','SIN','TAN','CAR', ...) or "
    "human names ('aitoff','mollweide','orthographic'). 'center' sets the "
    "central longitude (deg).",
    "frame= takes an astropy frame name: 'ICRS' (equatorial, the default), "
    "'galactic', or 'ecliptic' (among others). It sets the grid labels AND the "
    "coordinate system your data lon/lat are interpreted in.",
    "plot_catalog's 2nd arg is a CATALOG (astropy Table / pandas DataFrame / "
    "dict of columns), and ra_col/dec_col/colorby/sizeby name COLUMNS in it — "
    "not coordinate arrays. Pass raw arrays to ax.scatter(..., "
    "transform=ax.get_transform('world')) instead.",
    "Prefer the bundled 'sph.*' colormaps (e.g. cmap='sph.deepsky') over "
    "matplotlib built-ins for map/image renders; sph.show_colormaps() lists them.",
    "Optional dependencies gate some features: reproject (raster draping), "
    "healpy (HEALPix), cartopy (Earth coastlines), astroquery (catalog queries). "
    "Core plotting needs only numpy/matplotlib/astropy.",
    "Longitudes/latitudes passed to plotting helpers are in the frame's own "
    "coordinate system, in degrees (e.g. galactic l,b for a galactic frame).",
]

# Tutorial slug -> title. The tutorials are where full compositions live (a
# recipe names the functions and shows the core call; the tutorial shows how
# they combine into a real scene). Recipes cross-link to one via ``see=`` /
# the category default below, so an agent can jump from "these are the tools"
# to "here is the worked example".
_TUTORIALS = {
    "getting_started": "Getting Started with skyplothelper",
    "projections": "A Tour of Projections",
    "catalogs": "Catalogs — Querying, Plotting and Searching",
    "fits_images": "FITS Images & Quicklook",
    "globe_plots": "Globe and Planet Plotting",
    "healpix_workflows": "HEALPix Workflows",
    "cone_bowtie": "Cone & Bowtie Plots",
    "markers": "Markers — Rotatable and Image Stamps",
    "insets_and_zoom": "Insets and Zoom Axes",
    "decorating_frames": "Decorating Frames",
    "styling": "Themes, Palettes & Fonts",
    "overlay_grids": "Overlay Coordinate Grids",
    "regions": "Regions & Spherical Polygons",
    "constellations": "Constellations and Asterisms",
    "annotations": "Annotations & Overlays",
    "interactive_plotly": "Interactive Plotting with Plotly",
    "vector_fields": "Vector Fields & Sky Kinematics",
}

# Default tutorial per recipe category (a recipe's own ``see=`` overrides).
_CATEGORY_TUTORIAL = {
    "all-sky frames": "projections",
    "catalogs": "catalogs",
    "images": "fits_images",
    "globes": "globe_plots",
    "healpix": "healpix_workflows",
    "cubes": "fits_images",
    "cone (z-RA) frames": "cone_bowtie",
    "markers": "markers",
    "insets": "insets_and_zoom",
    "adjusting & legibility": "decorating_frames",
    "interactive (plotly)": "interactive_plotly",
    "overlays": "annotations",
}


class Recipe(NamedTuple):
    """One task→code recipe. ``code`` is a runnable snippet with the user's own
    data as named placeholders (``ra``, ``dec``, ``image``, ...)."""
    task:      str            # the user-facing goal
    category:  str            # grouping for the task index
    functions: tuple[str, ...]  # the key sph names it uses
    code:      str            # idiomatic snippet
    notes:     str = ""       # optional gotcha / adjustment hint
    see:       str = ""       # tutorial slug for the worked composition
                              # (overrides the category default)


RECIPES: tuple[Recipe, ...] = (
    Recipe(
        "Empty all-sky map with a coordinate grid",
        "all-sky frames", ("make_wcs_frame",),
        "import skyplothelper as sph\n"
        "ax = sph.make_wcs_frame(111, 'AIT', frame='ICRS', center=0)\n"
        "# ax is a WCSAxes with an Aitoff all-sky grid; draw onto it, then save.",
        "projection='MOL' (Mollweide), 'CAR' (rectangular), etc; "
        "frame='galactic'/'ecliptic' relabels the grid.",
    ),
    Recipe(
        "Scatter a catalog on an all-sky map",
        "catalogs", ("make_wcs_frame", "plot_catalog"),
        "import skyplothelper as sph\n"
        "ax = sph.make_wcs_frame(111, 'AIT', frame='ICRS', center=0)\n"
        "sph.plot_catalog(ax, catalog, ra_col='ra', dec_col='dec',\n"
        "                 colorby='mag', sizeby='flux', cbar=True)\n"
        "# catalog: an astropy Table / pandas DataFrame / dict of columns;\n"
        "# colorby/sizeby name COLUMNS (deg for ra/dec).",
        "for a non-ICRS catalog use lon_col=/lat_col= + frame=; "
        "size_scale=('sqrt'|'log'|callable) shapes the sizeby mapping.",
    ),
    Recipe(
        "Overlay a second coordinate system's grid (e.g. galactic on ICRS)",
        "overlays", ("make_wcs_frame", "CoordinateOverlay"),
        "import skyplothelper as sph\n"
        "ax = sph.make_wcs_frame(111, 'AIT', frame='ICRS', center=0)\n"
        "sph.CoordinateOverlay(ax, frame='galactic').plot(color='orange')",
        notes="",
        see="overlay_grids",
    ),
    Recipe(
        "Quick-look a FITS image (contours/pixel map, beam, stats)",
        "images", ("quicklook_plot",),
        "import skyplothelper as sph\n"
        "res = sph.quicklook_plot('image.fits', image=True, contours=False,\n"
        "                         colorbar=True, colormap='sph.deepsky')\n"
        "# res.fig / res.ax / res.image; auto beam + title from the header.",
        "image_data may be a path or a 2-D array (+ header=/wcs=). "
        "contours=True overlays sigma-based contours.",
    ),
    Recipe(
        "Show a FITS image on a WCS frame with a colorbar",
        "images", ("simpleimage_figure", "add_colorbar"),
        "import skyplothelper as sph\n"
        "res = sph.simpleimage_figure(image, header, cmap='sph.deepsky',\n"
        "                             colorbar=True)\n"
        "# image is a 2-D ndarray, header its FITS WCS header.",
    ),
    Recipe(
        "Orthographic sky globe centered on a target",
        "globes", ("make_globe_frame",),
        "import skyplothelper as sph\n"
        "ax = sph.make_globe_frame(111, center_LONdeg=266.4, center_LATdeg=-29.0)\n"
        "# a tilted-globe view centered on (RA, Dec); draw sky data onto ax.",
    ),
    Recipe(
        "Globe of the Earth (or a planet)",
        "globes", ("make_planet_frame",),
        "import skyplothelper as sph\n"
        "ax = sph.make_planet_frame(111, body='earth', center_LONdeg=0,\n"
        "                           center_LATdeg=20)\n"
        "# geographic (east-right) orientation; drape a texture onto it with the\n"
        "# next recipe, then add surface markers / features.",
    ),
    Recipe(
        "Drape an Earth/planet texture onto a globe",
        "globes",
        ("pseudofits_from_image", "reproject_rgb_map", "make_planet_frame"),
        "import numpy as np\n"
        "import skyplothelper as sph\n"
        "ax = sph.make_planet_frame(111, body='earth', center_LONdeg=0,\n"
        "                           center_LATdeg=20)\n"
        "# 1. wrap an equirectangular RGB raster (Blue Marble, planet map) in a\n"
        "#    synthetic GEOGRAPHIC WCS (geo=True — lands it the right way round):\n"
        "hdu = sph.pseudofits_from_image('earth_texture.jpg', geo=True)\n"
        "# 2. resample onto the globe frame's pixel grid, drape below the graticule:\n"
        "out_hdr = ax.wcs.to_header()\n"
        "nx = round(ax.get_xlim()[1] - ax.get_xlim()[0])\n"
        "ny = round(ax.get_ylim()[1] - ax.get_ylim()[0])\n"
        "out_hdr['NAXIS1'], out_hdr['NAXIS2'] = nx, ny\n"
        "bg = sph.reproject_rgb_map(hdu, out_hdr, shape_out=(ny, nx))\n"
        "ax.imshow(np.nan_to_num(bg), zorder=-10)",
        "geo=True is essential — it gives the raster an east-right geographic WCS. "
        "reproject_rgb_map handles the RGB resample (needs the reproject extra). "
        "The same drape works on a make_globe_frame SKY globe with a celestial "
        "raster (geo=False).",
    ),
    Recipe(
        "Plot a sparse HEALPix map",
        "healpix", ("plot_healpix_sparse",),
        "import skyplothelper as sph\n"
        "ax = sph.make_wcs_frame(111, 'MOL', frame='ICRS', center=0)\n"
        "sph.plot_healpix_sparse(pixel_indices, values, nside, ax=ax,\n"
        "                        nest=False, cmap='sph.deepsky')",
        "needs healpy. Omit values (None) to just outline pixels.",
    ),
    Recipe(
        "Channel maps from a spectral-line cube",
        "cubes", ("channel_map",),
        "import skyplothelper as sph\n"
        "res = sph.channel_map('cube.fits', channels=9, cmap='sph.dusk')\n"
        "# a compact grid of velocity-labeled channel panels + one colorbar.",
        "moment0=True adds an integrated-intensity panel; beam=True, "
        "scalebar=<arcsec> add furniture.",
    ),
    Recipe(
        "Moment maps (integrated intensity / velocity field / dispersion)",
        "cubes", ("DataCube", "MomentMap"),
        "import skyplothelper as sph\n"
        "cube = sph.DataCube.from_fits('cube.fits')\n"
        "cube.moment0().plot()                       # integrated intensity\n"
        "cube.moment1(threshold=3*rms).plot()        # velocity field\n"
        "# moments 1/2 NEED a threshold (a few x RMS) or they are noise.",
        "MomentMap.plot() renders on an sph frame (diverging velocity field) and "
        "returns a result with .fig/.ax; the header beam draws automatically "
        "(beam=False to disable). MomentMap.from_fits(path, order=) wraps your "
        "own map.",
    ),
    Recipe(
        "Draw a great circle / plane on a sky frame",
        "overlays", ("make_wcs_frame", "add_great_circle", "add_plane_overlay"),
        "import skyplothelper as sph\n"
        "ax = sph.make_wcs_frame(111, 'AIT', frame='ICRS', center=0)\n"
        "sph.add_plane_overlay(ax, plane='galactic')   # or 'ecliptic'\n"
        "sph.add_great_circle(ax, pole_lon=0, pole_lat=90)",
    ),
    Recipe(
        "Drape an RGB sky panorama onto a projection",
        "images", ("load_sky_image", "reproject_background", "make_wcs_frame"),
        "import skyplothelper as sph\n"
        "img, hdr = sph.load_sky_image('milkyway.jpg', frame='galactic', center=0)\n"
        "ax = sph.make_wcs_frame(111, 'AIT', frame='galactic', center=0)\n"
        "ax.imshow(sph.reproject_background(img, hdr, ax))",
        "needs reproject. downscale=2 gives a faster draft render.",
    ),
    Recipe(
        "Interactive (plotly) all-sky figure",
        "interactive (plotly)", ("plotly.make_figure", "plotly.add_scatter"),
        "import skyplothelper.plotly as sphpl\n"
        "fig = sphpl.make_figure(projection='aitoff')\n"
        "sphpl.add_scatter(fig, lon, lat)\n"
        "fig.show()",
        "the plotly subpackage mirrors the matplotlib API for hoverable, "
        "zoomable HTML figures.",
    ),
    Recipe(
        "Catalog scatter with color + size legends",
        "catalogs", ("make_wcs_frame", "plot_catalog"),
        "import skyplothelper as sph\n"
        "ax = sph.make_wcs_frame(111, 'AIT', frame='ICRS', center=0)\n"
        "sph.plot_catalog(ax, catalog, ra_col='ra', dec_col='dec',\n"
        "                 colorby='mag', sizeby='flux',\n"
        "                 cbar=True, size_legend=True)",
        "cbar_label=/size_legend_num= customize; sizeby='flux' + "
        "size_scale='sqrt' is common for fluxes.",
    ),
    Recipe(
        "Color stars by temperature (perceived color)",
        "catalogs", ("make_wcs_frame", "teff_to_rgb"),
        "import skyplothelper as sph\n"
        "ax = sph.make_wcs_frame(111, 'AIT', frame='ICRS', center=0)\n"
        "colors = sph.teff_to_rgb(teff)      # (N,3) perceived RGB per star\n"
        "ax.scatter(ra, dec, c=colors, transform=ax.get_transform('world'))",
        "From a color index instead: sph.color_index_to_rgb(value, index='BP-RP') "
        "(Gaia BP-RP, SDSS g-r, 2MASS J-K, or Johnson B-V). sph.bp_rp_to_rgb / "
        "sph.bv_to_rgb are shortcuts. Hot=blue-white, cool=orange; the Sun is "
        "white, not green. Don't feed BP-RP to bv_to_rgb -- it over-reddens.",
    ),
    Recipe(
        "Redshift / velocity wedge (cone frame)",
        "cone (z-RA) frames", ("make_cone_frame", "cone_scatter"),
        "import skyplothelper as sph\n"
        "ax = sph.make_cone_frame(111, angle_center=180, angle_half_width=60,\n"
        "                         r_min=0, r_max=0.1)\n"
        "sph.cone_scatter(ax, ra, redshift, s=8)   # angle=RA (deg), r=z",
        "make_cone_frame draws a redshift/velocity vs RA wedge; cone_hexbin / "
        "cone_pcolormesh for density.",
    ),
    Recipe(
        "Mark a telescope/facility on a globe",
        "markers", ("make_planet_frame", "add_telescope_marker"),
        "import skyplothelper as sph\n"
        "ax = sph.make_planet_frame(111, body='earth', center_LONdeg=-70,\n"
        "                           center_LATdeg=-24)\n"
        "sph.add_telescope_marker(ax, (-70.4, -24.6))   # (lon, lat) deg",
        "also add_antenna_marker / add_dome_marker; aim_at= points the dish.",
    ),
    Recipe(
        "Reticle / crosshair on a target",
        "overlays", ("make_wcs_frame", "add_reticle"),
        "import skyplothelper as sph\n"
        "ax = sph.make_wcs_frame(111, 'TAN', frame='ICRS', center=180)\n"
        "sph.add_reticle(ax, (180.0, 0.0))          # (ra, dec) deg",
    ),
    Recipe(
        "Field-of-view / geodesic circle",
        "overlays", ("make_wcs_frame", "add_geodesic_circle"),
        "import skyplothelper as sph\n"
        "ax = sph.make_wcs_frame(111, 'AIT', frame='ICRS', center=0)\n"
        "sph.add_geodesic_circle(ax, 180.0, 0.0, radius_deg=15)",
        "a true small circle on the sphere (e.g. a survey footprint radius).",
        see="regions",
    ),
    Recipe(
        "Outline a region (spherical polygon)",
        "overlays", ("make_wcs_frame", "add_spherical_polygon"),
        "import skyplothelper as sph\n"
        "ax = sph.make_wcs_frame(111, 'AIT', frame='ICRS', center=0)\n"
        "sph.add_spherical_polygon(ax, [150, 210, 210, 150],\n"
        "                          [-20, -20, 20, 20], facecolor='C0', alpha=0.2)",
        "edges follow great circles (geodesic='auto'); add_great_circle_band for "
        "a zone between two parallels of a great circle.",
        see="regions",
    ),
    Recipe(
        "Constellation stick figures",
        "overlays", ("make_wcs_frame", "add_constellation_lines"),
        "import skyplothelper as sph\n"
        "ax = sph.make_wcs_frame(111, 'AIT', frame='ICRS', center=0)\n"
        "sph.add_constellation_lines(ax)            # bundled line data\n"
        "sph.add_constellation_labels(ax)",
        "add_constellation_boundaries for IAU borders; constellations=['Ori',...] "
        "to select.",
        see="constellations",
    ),
    Recipe(
        "Zoom-in inset panel on a map/globe",
        "insets", ("make_wcs_frame", "reproject_inset_axes"),
        "import skyplothelper as sph\n"
        "ax = sph.make_wcs_frame(111, 'AIT', frame='ICRS', center=0)\n"
        "inset = sph.reproject_inset_axes(ax, [0.05, 0.05, 0.3, 0.3],\n"
        "                                 projection='TAN', center=(180, 0),\n"
        "                                 size=20)   # 20 deg field",
        "draws a magnified WCS panel; drape data/rasters onto `inset` as usual.",
    ),
    Recipe(
        "Scale bar on an image",
        "images", ("simpleimage_figure", "add_sizebar_asec"),
        "import skyplothelper as sph\n"
        "res = sph.simpleimage_figure(image, header, cmap='sph.deepsky')\n"
        "sph.add_sizebar_asec(res.ax, header, 30, '30\\\"')   # 30 arcsec bar",
    ),
    Recipe(
        "Rotation-axis rod on an orthographic globe",
        "globes", ("make_globe_frame", "add_pole_rod"),
        "import skyplothelper as sph\n"
        "ax = sph.make_globe_frame(111, center_LONdeg=30, center_LATdeg=15)\n"
        "sph.add_pole_rod(ax)",
        "SIN/globe frames only; length= sets how far the rod tips reach.",
    ),
    Recipe(
        "Set the coordinate grid spacing & style",
        "adjusting & legibility", ("make_wcs_frame", "style_grid"),
        "import skyplothelper as sph\n"
        "ax = sph.make_wcs_frame(111, 'AIT', frame='ICRS', center=0,\n"
        "                        lon_spacing=30, lat_spacing=15,\n"
        "                        gridcolor='0.4', gridalpha=0.6)\n"
        "sph.style_grid(ax, color='0.4', alpha=0.6)     # or restyle post-hoc",
        "lon_spacing/lat_spacing in degrees (or 'auto'); style_grid also takes "
        "stroke_color=/stroke_lw= for legibility over busy backgrounds.",
    ),
    Recipe(
        "Format the coordinate tick labels",
        "adjusting & legibility", ("make_wcs_frame", "format_ticklabels"),
        "import skyplothelper as sph\n"
        "ax = sph.make_wcs_frame(111, 'AIT', frame='ICRS', center=0)\n"
        "sph.format_ticklabels(ax, style='publication', lon_fmt='hh:mm')",
        "lon_fmt/lat_fmt control sexagesimal vs decimal; which='lon'/'lat'/'both' "
        "targets one axis; lon_sep sets the separators.",
    ),
    Recipe(
        "Add strokes (outlines) for legibility",
        "adjusting & legibility", ("apply_frame_stroke", "style_grid"),
        "import skyplothelper as sph\n"
        "ax = sph.make_wcs_frame(111, 'AIT', frame='ICRS', center=0)\n"
        "# most sph text / marker / overlay helpers take stroke_color=/stroke_lw=:\n"
        "sph.add_reticle(ax, (180, 0), stroke_color='white', stroke_lw=2)\n"
        "sph.apply_frame_stroke(ax, stroke_color='white')      # stroke the frame\n"
        "sph.style_grid(ax, stroke_color='white', stroke_lw=1.5)  # stroke the grid",
        "a contrasting stroke keeps labels/lines readable over busy images; "
        "~39 helpers accept the stroke_color=/stroke_lw= pair.",
    ),
    Recipe(
        "Add or customize a colorbar",
        "adjusting & legibility", ("simpleimage_figure", "add_colorbar"),
        "import skyplothelper as sph\n"
        "res = sph.simpleimage_figure(image, header, cmap='sph.deepsky')\n"
        "sph.add_colorbar(res.image, ax=res.ax, label='Jy/beam', location='right')",
        "location='left'/'top'/'bottom'; mode='inset' for ImageGrid/channel_map "
        "panels (auto-falls back to inset when a divider can't be added).",
    ),
    Recipe(
        "Apply a publication theme / global style",
        "adjusting & legibility", ("set_theme", "set_base_style"),
        "import skyplothelper as sph\n"
        "sph.set_theme('publication')      # or a dict of rcParam overrides\n"
        "# then build frames/plots as usual; set_base_style('standard'|...) too.",
        see="styling",
    ),
)


# ---------------------------------------------------------------------------
# Rendering (shared by the printed overview and the docs generator)
# ---------------------------------------------------------------------------

def _categories() -> list[str]:
    seen: list[str] = []
    for r in RECIPES:
        if r.category not in seen:
            seen.append(r.category)
    return seen


def _render_overview() -> str:
    """The compact orientation text (scope + frame-first + conventions + index)."""
    lines = [
        "skyplothelper — astronomy sky-plotting (WCS frames, all-sky maps, "
        "globes, spectral cubes, HEALPix, catalogs; matplotlib + plotly).",
        "",
        FRAME_FIRST,
        "",
        "CONVENTIONS (read these — they are the common first-attempt mistakes):",
    ]
    lines += [f"  - {c}" for c in CONVENTIONS]
    lines += ["", "TASK INDEX (call sph.recipes('<keyword>') for runnable code):"]
    for cat in _categories():
        lines.append(f"  {cat}:")
        for r in RECIPES:
            if r.category == cat:
                fns = ", ".join(r.functions)
                lines.append(f"    - {r.task}  [{fns}]")
    lines += ["",
              f"{len(RECIPES)} recipes. sph.recipes('cube'), "
              "sph.recipes('catalog'), ... print copy-paste code; "
              "sph.recipes() lists them all. sph.overview(as_dict=True) "
              "returns the structured catalog."]
    lines += ["",
              "EXAMPLE ASSETS (repo, not shipped in the wheel): the demo FITS / "
              "catalogs / Earth+planet textures / marker icons used by the "
              "tutorials live in `examples/data/`, indexed with sources, credits, "
              "and the marker-icon rest angles (radio dish 130 deg, optical "
              "telescope 65 deg, space telescope 194 deg) in "
              "`examples/data/README.md`."]
    return "\n".join(lines)


def _recipe_tutorial(r: Recipe) -> str:
    """The tutorial slug for a recipe's worked composition: its own ``see=``
    if set, else the category default. ``''`` if neither maps."""
    return r.see or _CATEGORY_TUTORIAL.get(r.category, "")


def _render_recipe(r: Recipe) -> str:
    out = [f"### {r.task}", f"  ({r.category}; uses: {', '.join(r.functions)})",
           "", r.code]
    if r.notes:
        out += ["", f"# Adjust: {r.notes}"]
    slug = _recipe_tutorial(r)
    if slug:
        out += ["", f"# See the {_TUTORIALS.get(slug, slug)} tutorial for the "
                f"full worked example ({_DOCS_URL}/tutorials/{slug}.html)."]
    return "\n".join(out)


def _match(r: Recipe, q: str) -> bool:
    hay = " ".join((r.task, r.category, " ".join(r.functions), r.notes)).lower()
    return all(tok in hay for tok in q.lower().split())


# ---------------------------------------------------------------------------
# llms.txt / llms-full.txt rendering (the docs generator + the CI sync test
# both call render_llms; keeping it here means those files are rendered from
# the SAME catalog the runtime helpers read, so they can never drift).
# ---------------------------------------------------------------------------

_DOCS_URL = "https://skyplothelper.readthedocs.io/en/latest"
_REPO_URL = "https://github.com/pjcigan/skyplothelper"


def _llms_header() -> str:
    from ._version import __version__
    return (
        f"# skyplothelper\n\n"
        f"> Astronomy sky-plotting for Python (v{__version__}): WCS frames, "
        f"all-sky maps, tilted globes, redshift (z-RA) wedges, spectral cubes + "
        f"moment maps, HEALPix, and catalog plotting — matplotlib and plotly.\n\n"
        f"skyplothelper is **frame-first**: create a sky frame, then draw data "
        f"and decorations onto its axes. In a Python session, "
        f"`import skyplothelper as sph; sph.overview()` prints the orientation "
        f"below and `sph.recipes('<keyword>')` prints these recipes.\n")


def _llms_model_conventions() -> str:
    out = ["## The frame-first model\n", FRAME_FIRST, "",
           "## Conventions (the common first-attempt mistakes)\n"]
    out += [f"- {c}" for c in CONVENTIONS]
    return "\n".join(out) + "\n"


def _llms_recipes() -> str:
    out = ["## Recipes (runnable; your data as named placeholders)\n"]
    for cat in _categories():
        out.append(f"### {cat}\n")
        for r in RECIPES:
            if r.category == cat:
                out += [f"**{r.task}**\n", "```python", r.code, "```"]
                if r.notes:
                    out.append(f"*Adjust:* {r.notes}")
                slug = _recipe_tutorial(r)
                if slug:
                    out.append(
                        f"*Worked example:* "
                        f"[{_TUTORIALS.get(slug, slug)}]"
                        f"({_DOCS_URL}/tutorials/{slug}.html)")
                out.append("")
    return "\n".join(out)


def _llms_signatures() -> str:
    """A best-effort signature reference for every function the recipes use."""
    import inspect
    import re

    import skyplothelper as sph
    names: list[str] = []
    for r in RECIPES:
        for fn in r.functions:
            if fn not in names:
                names.append(fn)
    out = ["## Function signatures\n"]
    for name in names:
        obj: Any = sph
        for part in name.split("."):          # resolve e.g. 'plotly.add_scatter'
            obj = getattr(obj, part, None)
            if obj is None:
                break
        try:
            # Python 3.14's inspect wraps unresolved string annotations as
            # ``ForwardRef('X')``; strip that back to ``X`` so the rendered
            # signatures are identical across interpreter versions (keeps the
            # committed llms*.txt in sync on every CI Python).
            sig = re.sub(r"ForwardRef\('([^']*)'\)", r"\1",
                         str(inspect.signature(obj)))
            out.append(f"- `{name}{sig}`")
        except (TypeError, ValueError):
            if obj is not None:
                out.append(f"- `{name}` ({type(obj).__name__})")
    return "\n".join(out) + "\n"


def _llms_links() -> str:
    return (f"## Docs\n\n"
            f"- [Quickstart]({_DOCS_URL}/quickstart.html)\n"
            f"- [User guide]({_DOCS_URL}/guide/index.html)\n"
            f"- [API reference]({_DOCS_URL}/api/index.html)\n"
            f"- [Tutorials]({_DOCS_URL}/tutorials/index.html)\n"
            f"- [Example data & marker icons]"
            f"({_REPO_URL}/blob/main/examples/data/README.md) — the demo FITS, "
            f"catalogs, Earth/planet textures, and imscatter marker icons the "
            f"tutorials use (with sources, credits, and the icon rest angles). "
            f"In the repo under `examples/data/`; not shipped in the wheel.\n")


def render_llms(full: bool = False) -> str:
    """Render the ``llms.txt`` (``full=False``) or ``llms-full.txt``
    (``full=True``) content from the capability catalog.

    Both carry the frame-first model, conventions, and the **runnable recipe
    code** (a name-only index left agents guessing arguments); ``full=True``
    additionally appends a function-signature reference.
    """
    parts = [_llms_header(), _llms_model_conventions(), _llms_recipes()]
    if full:
        parts.append(_llms_signatures())
    parts.append(_llms_links())
    return "\n".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def overview(query: str | None = None, *, as_dict: bool = False) -> Any:
    """Print an orientation to skyplothelper — scope, the frame-first model,
    coordinate conventions, and a task→function index.

    Parameters
    ----------
    query : str, optional
        If given, defer to :func:`recipes` (print matching runnable recipes).
    as_dict : bool
        Return the structured catalog (for tools / programmatic use) instead of
        printing: ``{'conventions': [...], 'recipes': [{'task', 'category',
        'functions', 'code', 'notes'}, ...]}``.

    Notes
    -----
    Designed as the first call an agent (or newcomer) makes:
    ``import skyplothelper as sph; sph.overview()``.
    """
    if as_dict:
        return {
            "frame_first": FRAME_FIRST,
            "conventions": list(CONVENTIONS),
            "recipes": [r._asdict() for r in RECIPES],
        }
    if query is not None:
        return recipes(query)
    print(_render_overview())
    return None


def recipes(query: str | None = None) -> None:
    """Print runnable task→code recipes.

    With no argument, list every recipe (task + category). With a *query*
    keyword (a task / function / topic), print the matching recipes' copy-paste
    code — including the cross-cutting "how do I adjust it" ones (grid spacing,
    tick-label format, strokes for legibility, colorbars).

    Examples
    --------
    >>> import skyplothelper as sph
    >>> sph.recipes()              # the full menu
    >>> sph.recipes('cube')        # channel maps + moment maps
    >>> sph.recipes('stroke')      # legibility outlines
    """
    if query is None:
        lines = [f"{len(RECIPES)} recipes "
                 "(call sph.recipes('<keyword>') for the code):", ""]
        for cat in _categories():
            lines.append(f"  {cat}:")
            lines += [f"    - {r.task}" for r in RECIPES if r.category == cat]
        print("\n".join(lines))
        return
    hits = [r for r in RECIPES if _match(r, query)]
    if not hits:
        cats = ", ".join(_categories())
        print(f"No recipe matched {query!r}. Try a topic: {cats}.\n"
              f"Or run sph.recipes() for the full menu.")
        return
    print("\n\n".join(_render_recipe(r) for r in hits))
