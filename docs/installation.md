# Installation

skyplothelper requires **Python 3.10+** and **astropy 6.0+**. The core install
pulls in numpy, matplotlib, astropy, shapely, and healpy:

```bash
pip install skyplothelper
```

HEALPix binning/plotting and the spherical-region tools (`CompoundRegion`,
frame bands) are therefore available out of the box — no extra needed. The one
caveat: `healpy` has no Windows wheel, so on Windows the HEALPix features raise
an informative error while everything else works normally (use WSL or conda for
HEALPix there).

## Optional features

To get everything in one go:

```bash
pip install skyplothelper[all]
```

That's all most users need. The rest of this section is only relevant if you'd
rather install dependencies selectively.

**You never need to pick and choose to get the full package.** skyplothelper
itself always ships *all* of its code — the names in brackets are not separate
feature packages, they just tell `pip` which *third-party libraries* to pull in
so a given feature has what it needs at runtime. Optional features fail
gracefully: if a dependency is missing, calling that feature raises an
informative `ImportError` naming the extra to install, and everything else keeps
working.

The optional third-party dependencies are:

| Library | Enables |
|---------|---------|
| `cartopy` | cartopy backend, nightshade |
| `reproject` | image reprojection |
| `astroquery` | SIMBAD / NED / SkyView / VizieR queries |
| `scipy` | extra cone tools |
| `pysymlog` | symmetric-log image stretch |
| `plotly` (≥5.0) | interactive plotly export backend |
| `dash` (≥2.9) | interactive Dash FITS viewer |

If you manage your environment with conda, you can install any of these
yourself (e.g. `conda install -c conda-forge healpy cartopy`) — skyplothelper
will pick them up automatically.

To let `pip` install just the dependencies for one feature, use the matching
extra (combine several in a single bracket list as needed):

```bash
pip install skyplothelper[cartopy]      # cartopy backend, nightshade
pip install skyplothelper[reproject]    # image reprojection
pip install skyplothelper[query]        # SIMBAD / NED / SkyView / VizieR (astroquery)
pip install skyplothelper[cone]         # extra cone tools (scipy)
pip install skyplothelper[pysymlog]     # symmetric-log image stretch
pip install skyplothelper[plotly]       # interactive plotly export backend
pip install skyplothelper[dash]         # interactive Dash FITS viewer
```

## Development install

```bash
git clone https://github.com/pjcigan/skyplothelper
cd skyplothelper
pip install -e .[all,dev]
```

The `[dev]` extra adds `pytest`, `pytest-mpl`, `pytest-cov`, `ruff`, and `mypy`.

## Earth boundary data (optional)

Globe Earth-feature overlays (`plot_coastlines`, `plot_tectonic_plates`,
`plot_time_zones`) rely on a few `.npz` data files that are not bundled in the
wheel. Download them once:

```python
import skyplothelper as sph
sph.fetch_boundary_data()
```

This uses the standard-library `urllib` (no extra dependency) to fetch the
canonical files from the GitHub repo, after which the boundary plotters work on
any globe axes.

## Example data (optional)

The example notebooks and the visual gallery use a small set of reference
datasets — a sample VLBA FITS image, an all-sky photograph and equirectangular
Earth maps for projection backgrounds, planetary-body textures, and marker
icons (Sun, Moon, planets). These are **not installed with the package** (the
wheel ships only `skyplothelper/`, to keep installs lean); they live in the
GitHub repository under
[`examples/data/`](https://github.com/pjcigan/skyplothelper/tree/main/examples/data).

You only need them if you want to run the examples with the same inputs. Grab
the whole directory by cloning the repo, or download individual files from the
GitHub web interface. See
[`examples/data/README.md`](https://github.com/pjcigan/skyplothelper/blob/main/examples/data/README.md)
for the full file list, provenance, and licensing / attribution of each dataset.
