Utilities
=========

The supporting cast: the three-layer styling system (base / theme /
palette, each accepting bundled presets or your own definitions),
catalog and vector plotting onto frames, the cartopy backend for
terrestrial maps, the astroquery-backed lookup and download wrappers,
and the bundled reference constants (solar-system properties, band
definitions, sky positions, color palettes). Narrative:
:doc:`/guide/styling` (styling), :doc:`/guide/queries` (queries), and
:doc:`/guide/vectors` (catalog & vector plotting).

.. currentmodule:: skyplothelper

Getting oriented
----------------

With 300+ public names, the fastest way in — for a newcomer or an AI agent — is
the built-in orientation layer. :func:`overview` prints the scope, the
frame-first model, the coordinate conventions, and a task-to-function index
(``overview(as_dict=True)`` returns it as a structured catalog).
:func:`recipes` prints copy-paste code for a task keyword (``recipes('cube')``),
or the full menu with no argument. The same catalog is published as a
`map <../llms.txt>`__ and a `recipe corpus <../llms-full.txt>`__ at the site
root for agent ingestion (the ``llms.txt`` convention).

.. autosummary::
   :toctree: generated
   :nosignatures:

   overview
   recipes

Styling
-------

Global setters mutate matplotlib's rcParams (``set_*``); the axes appliers
retrofit the look onto a single existing axes (``style_*``). The four
layers compose in order: base (structure) → theme → palette (colors) →
font → explicit overrides.

:func:`set_base_style` selects the structural layer from eight named
presets in :data:`BASE_PRESETS` (``standard``, ``structural``, ``journal``,
``press``, ``poster``, ``tufte``, ``screen``, ``minimalist``), a custom RC
dict, or a reset directive (``'default'`` / ``'reset'`` / ``'mpl'``). Fonts
are applied as stacks for graceful degradation; serif presets pair
``mathtext.fontset='cm'`` and sans presets ``'stixsans'``. ``minimalist``
strips the axis scaffolding — splash / title images only, not data plots.

.. autosummary::
   :toctree: generated
   :nosignatures:

   set_base_style
   set_theme
   set_palette
   set_font
   set_style
   style_context
   style_wcs_axes
   apply_frame_stroke
   style_annotation

Presets, palettes, and font stacks:

- :data:`BASE_PRESETS` — the eight structural presets backing
  :func:`set_base_style`.
- :data:`CYCLE_PALETTES` — data-color cycles for :func:`set_palette`,
  CVD- and grayscale-tested (the colors are lightness-separated so they
  stay distinguishable for color-vision-deficient viewers and in print).
- :data:`ANNOTATION_PALETTES` — figure-scaffolding palettes for
  :func:`style_annotation`: ``parchment``, ``publication``, ``dark``,
  ``night``, ``denim``, each defining 12 roles (frame, label, compass,
  accent, …).
- :data:`FONT_PRESETS` — font-family stacks for :func:`set_font`
  (``journal``, ``talk``, ``tufte``, ``web``, ``classical``, ``sketch``,
  ``mono``); a guaranteed always-available fallback face is enforced when the
  stack is applied. ``classical`` / ``sketch`` need their faces registered
  (``register=``).
- :data:`MONO_STACK` — a per-artist monospace font stack (not a preset)
  for fixed-width readouts, e.g. ``ax.text(..., family=sph.MONO_STACK)``.

.. currentmodule:: skyplothelper.style

.. autosummary::
   :toctree: generated
   :nosignatures:

   BASE_PRESETS
   CYCLE_PALETTES
   ANNOTATION_PALETTES
   FONT_PRESETS
   MONO_STACK

.. currentmodule:: skyplothelper

Colormaps
---------

Curated image colormaps registered under the ``sph.`` prefix on import
(each with an auto ``sph.<name>_r`` reverse) — twelve luminance-smoothed
linear maps and six diverging ``diff_*`` maps. Reach a map by its registry
string (``cmap="sph.deepsky"``), via :func:`get_colormap`, or as an
attribute for readers who prefer objects (``sph.colormaps.deepsky``,
``_r`` included). See :doc:`/guide/styling` for the swatch gallery.

.. autosummary::
   :toctree: generated
   :nosignatures:

   list_colormaps
   get_colormap
   show_colormaps

Star colors
-----------

Convert a stellar effective temperature (K) or a photometric color index to
the RGB color the eye *perceives* — hot stars blue-white, cool stars
orange-red — for coloring a catalog by spectral type. :func:`teff_to_rgb`
takes a temperature; :func:`color_index_to_rgb` takes a named index — Johnson
``B-V``, Gaia ``BP-RP``, SDSS/PS1 ``g-r``, or 2MASS ``J-K`` — and resolves it
to a temperature (the survey indices against the Pecaut & Mamajek (2013) dwarf
sequence, ``B-V`` via the Ballesteros relation), so a star reads the *same*
perceived color whichever index you arrive with. :func:`bv_to_rgb` and
:func:`bp_rp_to_rgb` are thin shortcuts for the two most common indices —
prefer :func:`bp_rp_to_rgb` for Gaia rather than feeding ``BP-RP`` to
``bv_to_rgb`` (which over-reddens, since ``BP-RP`` spans a wider range).

All are a **tristimulus** (Planckian-locus / CIE color-matching) integral, not
a Wien-peak shortcut, so a Sun-temperature star comes out **white, not
green**. Brightness is deliberately not encoded (every color sits at a common
maximum channel); map magnitude to marker size or alpha separately. They are
vectorized, so they drop straight into a scatter call:
``ax.scatter(ra, dec, c=sph.bp_rp_to_rgb(cat["bp_rp"]))``. Missing photometry
(a non-finite input color) yields a masked, non-finite RGB row, so
``np.isfinite(colors).all(axis=1)`` flags the stars to drop or mark.

.. autosummary::
   :toctree: generated
   :nosignatures:

   teff_to_rgb
   color_index_to_rgb
   bp_rp_to_rgb
   bv_to_rgb

Catalog & vector plots
----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   plot_sky_vectors
   SkyVectorResult
   sky_quiverkey
   plot_displacement
   plot_catalog
   CatalogPlot

cartopy backend
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   make_cartopy_frame
   cartopy_figure
   list_cartopy_projections

Catalog queries
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   resolve_name
   resolve_names
   query_simbad
   query_ned
   download_skyview
   download_hips
   list_skyview_surveys
   search_vizier
   overlay_cutout

Catalog search
--------------

Offline spatial filters on a catalog *you already hold* (vs. the remote
fetchers above). Type-preserving — a Table in gives a Table out, a
DataFrame a DataFrame, and so on. :func:`region_search` accepts any object
with ``contains_points`` — including a :class:`~skyplothelper.CompoundRegion`
(see :doc:`/api/geometry`).

.. autosummary::
   :toctree: generated
   :nosignatures:

   cone_search
   region_search
   crossmatch

Constants
---------

.. currentmodule:: skyplothelper.constants

.. autosummary::
   :toctree: generated
   :nosignatures:

   obliquities
   rot_periods
   planet_radii
   SKY_POSITIONS
   RADIO_BANDS
   FILTER_BANDS
   FACILITY_RESOLUTION
   SEPARATORS
   REGION_PALETTE
   REGION_PALETTE_NAMED
