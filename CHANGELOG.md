# Changelog

All notable changes to skyplothelper are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0]

Initial release.

skyplothelper is an astronomy visualization toolkit built on matplotlib and
astropy WCSAxes. A single `import skyplothelper as sph` provides:

- WCS frame builders for all-sky, globe, and field plots (`make_wcs_frame`,
  `allsky_figure`, `offset_figure`, `make_globe_frame`, `make_planet_frame`),
  plus a registry of FITS and non-FITS projections.
- Sky-coordinate tick formatting, coordinate-system conversions, and FITS
  header utilities.
- Overlays and annotations: coordinate planes, survey footprints, IAU
  constellations, beams, rulers, reticles, scale bars, and compasses.
- Spherical-region geometry — geodesic circles, rectangles, ellipses, bands,
  polygons, and set-algebraic `CompoundRegion`.
- Tilted-globe (orthographic) frames with Earth-feature overlays and a
  day/night nightshade blend.
- Cone (z-RA wedge) frames for cosmology diagrams.
- HEALPix binning, plotting, and queries; FITS image quicklook and
  reprojection; catalog queries (SIMBAD / NED / SkyView / VizieR).
- An interactive plotly export backend with a Dash-based FITS image viewer.
- Vector spherical harmonics and mutual sky-visibility (co-visibility) regions.

Optional features are gated behind per-feature install extras and fail
gracefully with informative messages when a dependency is absent.

[1.0.0]: https://github.com/pjcigan/skyplothelper/releases/tag/v1.0.0
