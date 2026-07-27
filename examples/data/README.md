# Example data

Reference datasets used by the example notebooks and the visual gallery.
**Not installed with the package** — these files live in the git repository
only (the wheel packages just `skyplothelper/`, and the sdist `include` list
omits `/examples/`).

## `0316+413.u.stacked.icd.fits`

VLBA 15 GHz (U-band) stacked total-intensity image of the radio source
**0316+413 (3C 84 / NGC 1275 / Perseus A)**, from the MOJAVE program.

- **Source page:** https://www.cv.nrao.edu/MOJAVE/sourcepages/0316+413.shtml
- **Program:** MOJAVE — Monitoring Of Jets in Active galactic nuclei with VLBA
  Experiments (https://www.cv.nrao.edu/MOJAVE/).
- **Please cite** the MOJAVE program when using these data, e.g. Lister et al.
  (2009, AJ 137, 3718) and Lister et al. (2018, ApJS 234, 12), and acknowledge:
  *"This research has made use of data from the MOJAVE database that is
  maintained by the MOJAVE team (Lister et al., 2018, ApJS, 234, 12)."*

The National Radio Astronomy Observatory is a facility of the National Science
Foundation operated under cooperative agreement by Associated Universities, Inc.

## `1502+106.u.stacked.icd.fits`

VLBA 15 GHz stacked image of the blazar **1502+106**, from the MOJAVE program — a
one-sided core-plus-jet, a useful morphological contrast to the 3C 84 image above.
Same program, credit, and acknowledgment as `0316+413.u.stacked.icd.fits`
(please cite Lister et al. 2009, 2018, and the MOJAVE database acknowledgment).

## `sn1987a_hst_F502N.fits`, `…_F625W.fits`, `…_F656N.fits`, `…_F658N.fits`

HST WFC3/UVIS imaging of **SN 1987A** in four filters (2014), drizzled to a common
grid and cropped to a ~9″ field that frames the famous **triple-ring system** (the
bright inner ring plus the two outer rings). The optical/line channels of a
multiwavelength view of the remnant.

- **Source:** Hubble Space Telescope archive (MAST), Proposal 13405 (2014).
- **Credit:** NASA/ESA Hubble. Public archival data.
- **Note:** these are on the native HST (WFC3, ~0.04″/pix) grid; the ALMA images
  below are on a separate 600² grid — reproject one onto the other if you need them
  pixel-aligned.

## `sn1987a_alma_315GHz.fits`, `sn1987a_alma_679GHz.fits`

ALMA **Band 7 (315 GHz)** and **Band 9 (679 GHz)** continuum images of SN 1987A
(the warm dust ring), both on a common 600² grid (co-registered with **each other**
for a two-band beam comparison) and recompressed to 32-bit. Each retains its
synthesized-beam header (`BMAJ`/`BMIN`/`BPA`): Band 7 ≈ 0.19″×0.14″,
Band 9 ≈ 0.081″×0.063″.

- **Please cite:** Cigan, P., et al. 2019, ApJ, 886, 51 (*High Angular Resolution
  ALMA Images of Dust and Molecules in the SN 1987A Ejecta*).
- **Credit:** ALMA (ESO/NAOJ/NRAO). *This paper makes use of ALMA data.* The
  Atacama Large Millimeter/submillimeter Array is a partnership of ESO, NSF (USA),
  and NINS (Japan), together with NRC (Canada), NSC and ASIAA (Taiwan), and KASI
  (Republic of Korea), in cooperation with the Republic of Chile.

## `ngc602_IR.fits`, `ngc602_R.fits`, `ngc602_B.fits`

Multiwavelength imaging of the star-forming region **NGC 602** in the Small
Magellanic Cloud — IR (Spitzer) plus optical R and B bands — used to build a
three-color composite. Downsampled 5× (to 720×720) from the originals.

- **Source:** Chandra "OpenFITS" multiwavelength dataset —
  http://chandra.harvard.edu/photo/openFITS/multiwavelength_data.html
- **Credit:** NASA/CXC/SAO and the originating observatories. Provided for
  educational and public use.

## `m51_optical.fits`

Optical image of the **Whirlpool galaxy (M51)** and its companion NGC 5195 — a
B+V luminance from ground-based Cousins-filter exposures, background-subtracted and
downsampled 2× (to 512×512). The **native, un-stretched** pixel values make it a
genuine high-dynamic-range target for image-stretch demonstrations.

- **Source:** courtesy of P. Cigan, from an educational observing night at the
  Observatoire de Haute-Provence (OHP), 2018.

## `crab_hst_F502N.fits`, `…_F547M.fits`, `…_F631N.fits`, `…_F673N.fits`

HST WFPC2/WFC mosaic of the **Crab Nebula (M1)** in four filters, used to build a
four-color composite. Background-subtracted, cropped, and downsampled 2× (to
1000×1000).

- **Source:** Hubble Space Telescope archive (MAST), Proposal 8222
  (filters F502N/F547M/F631N/F673N).
- **Credit:** NASA/ESA Hubble. Public archival data.

## `ddo70_hi_subcube.fits`

VLA **H I (21 cm) spectral cube** of the dwarf irregular galaxy **DDO 70
(Sextans B)** — a worked example of a 3-D data cube (43 velocity channels around
the line, 2× spatially binned to 225×225, cropped from the full survey cube).

- **Source:** LITTLE THINGS survey — VLA H I imaging, hosted by NRAO
  (https://science.nrao.edu/science/surveys/littlethings).
- **Please cite:** Hunter, D. A., et al. 2012, AJ, 144, 134 (*LITTLE THINGS*).

## `Allsky_noirlab2430b_1280x640.jpg`

All-sky photograph of the night sky (downsampled to 1280×640 from the
NOIRLab original — one of the largest open-source all-sky images, compiled
from dark-site imagery by astrophotographer Eckhard Slawik).

- **Source:** NOIRLab — https://noirlab.edu/public/images/noirlab2430b/
- **Credit:** NOIRLab/NSF/AURA/E. Slawik
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
  (per the NOIRLab image-usage policy, https://noirlab.edu/public/copyright/).
  Downsampled from the original.

## Earth maps

Global equirectangular Earth maps (used e.g. by the nightshade demo as the
night-lights and day topography/bathymetry layers). Both resized from NASA
originals; public domain (NASA).

- **`BlackMarble_2016_01deg.jpg`** — NASA "Black Marble" 2016 night lights
  (Suomi NPP VIIRS). Source: NASA Earth Observatory / SVS,
  https://svs.gsfc.nasa.gov/30876/ (also
  https://earthobservatory.nasa.gov/features/NightLights). Credit: NASA
  Earth Observatory. Public domain (NASA).
- **`world.topo.bathy.200412.3x5400x2700.jpg`** — NASA Blue Marble Next
  Generation, base topography + bathymetry. Source:
  https://science.nasa.gov/earth/earth-observatory/blue-marble-next-generation/base-topography-bathymetry/.
  Credit: NASA Earth Observatory (Reto Stöckli). Public domain (NASA).

## Source catalogs

Small CSV catalogs used by the tutorial notebooks to demonstrate plotting a
real source catalog across projections.

- **`icrf3_defining.csv`** — the 303 *defining* sources of the **Third
  Realization of the International Celestial Reference Frame (ICRF3)**,
  S/X-band catalog. Columns: `icrf_name`, `iers_name`, `ra_deg`, `dec_deg`,
  `n_sess` (number of VLBI sessions), `pos_err_mas` (combined formal
  positional uncertainty, mas). Derived from the published ICRF3 S/X catalog
  (RA/Dec converted to degrees; the positional error combines the RA and Dec
  formal uncertainties).
  - **Source:** IERS ICRF3 — https://www.iers.org/IERS/EN/DataProducts/ICRF/icrf.html
  - **Reference / please cite:** Charlot, P., et al. 2020, A&A, 644, A159
    (*The third realization of the International Celestial Reference Frame
    by very long baseline interferometry*).
- **`usno2025a_vlbi.csv`** — 5804 radio sources from the **USNO 2025a** quarterly
  VLBI global solution, cross-matched against the ICRF3 defining list. Used by the
  **Catalogs** tutorial (§2) as a real multi-dimensional catalog for the
  `MultiLegend` all-sky example (marker size = observations, color = accuracy,
  shape = ICRF3 defining). Columns: `iers_name`, `ra_deg`, `dec_deg`, `n_delays`
  (number of delay observations), `err_mas` (combined formal position error, mas),
  `defining` (1 if an ICRF3 defining source). Derived from the USNO gsnoop source
  solution (RA/Dec converted to degrees; the error combines the RA and Dec formal
  uncertainties).
  - **Source:** USNO quarterly VLBI global solutions —
    https://crf.usno.navy.mil/quarterly-vlbi-solution ; ICRF3 defining list from
    the IERS ICRF Product Center — https://hpiers.obspm.fr/icrs-pc/icrf/index.php
- **`messier.csv`** — the 110 **Messier** objects. Columns: `name`,
  `ra_deg`, `dec_deg`, `otype` (SIMBAD object type), `vmag` (V magnitude
  where available). Positions, types, and magnitudes resolved from SIMBAD.
  - **Credit:** *This research has made use of the SIMBAD database, operated
    at CDS, Strasbourg, France* (Wenger et al. 2000, A&AS, 143, 9).
- **`hipparcos_bright_pm.csv`** — the 4992 naked-eye stars (V < 6) of the
  **Hipparcos** main catalog with complete astrometry, used by the **Vector
  Fields & Sky Kinematics** tutorial as a real proper-motion field (the
  binned median field shows the solar-apex reflex dipole) and by the
  **Constellations** tutorial as a star field. Columns: `HIP`,
  `RAICRS`/`DEICRS` (deg, ICRS), `Vmag`, `Plx` (mas), `pmRA` (μ_α cos δ,
  mas/yr), `pmDE` (mas/yr), `BV` (Johnson B−V color index, mag).
  - **Provenance:** retrieved 2026-07-02 via `astroquery.vizier` from VizieR
    catalog **I/239/hip_main** with the filter `Vmag < 6.0` (4995 rows); the
    3 rows with incomplete astrometry (any masked value among the columns
    above) were dropped. Values are otherwise unmodified from the catalog.
    The `BV` column was added 2026-07 from the same catalog (matched on `HIP`);
    it is blank for the 2 stars (HIP 26220, 32609) with no catalogued B−V.
    Regenerate with `python fetch_hipparcos_bright.py` (needs `astroquery`) —
    the script reproduces every existing value byte-for-byte and only adds
    `BV`, so figures that predate the column are unaffected.
  - **Source:** VizieR — https://vizier.cds.unistra.fr/ (catalog I/239)
  - **Reference / please cite:** Perryman, M. A. C., et al. 1997, A&A, 323,
    L49 (*The Hipparcos Catalogue*); ESA 1997, ESA SP-1200.
- **`sstar_orbits.csv`** — orbital elements of the 16 best-measured bound
  **S-stars** orbiting **Sgr A\***, the Milky Way's central black hole, for the
  Keplerian-orbit animation in the **Animations** tutorial. Columns: `star`
  (MPE/Genzel name), `a_arcsec` (semi-major axis, angular), `ecc`, `incl_deg`,
  `node_deg` (P.A. of ascending node Ω), `periapsis_deg` (argument of periapsis
  ω), `t_peri_yr` (epoch of pericenter), `period_yr`, `Kmag`, `spectral`
  (e=early/l=late), `simbad`. Positions at any epoch follow from solving
  Kepler's equation and rotating onto the sky with the Thiele–Innes constants;
  the elements are osculating Keplerian, and pair with M = 4.28×10⁶ M☉ at
  R₀ = 8.32 kpc (reference epoch 2009.0).
  - **Provenance:** derived via `astroquery.vizier` from VizieR catalog
    **J/ApJ/837/30** (table 3), filtered to bound orbits (`e < 1`) with
    period < 100 yr — the tight central cluster. Values otherwise unmodified.
  - **Source:** VizieR — https://vizier.cds.unistra.fr/ (catalog J/ApJ/837/30)
  - **Reference / please cite:** Gillessen, S., et al. 2017, ApJ, 837, 30
    (*An Update on Monitoring Stellar Orbits in the Galactic Center*,
    DOI 10.3847/1538-4357/aa5c41). Please also acknowledge the VizieR service
    (Ochsenbein et al. 2000, A&AS, 143, 23; DOI 10.26093/cds/vizier).

## Redshift-survey slices

Small galaxy redshift subsamples (RA, redshift) used by the **Cone & Bowtie**
tutorial to draw the classic "slice of the universe" wedge diagrams. Each is a
seeded random subsample of a public survey, trimmed to a thin declination band
so the large-scale structure (filaments, voids, the Great Wall) reads clearly in
a wedge.

- **`sdss_slice.csv`** — 8000 galaxies from the **SDSS** main spectroscopic
  sample, in a thin equatorial band (Dec ≈ ±1.5°, R.A. 120°–255°, z 0.005–0.20)
  that runs through the **Sloan Great Wall**. Columns: `ra` (deg), `dec` (deg),
  `z` (redshift). Queried from the SDSS SkyServer (`SpecObj`, `class=GALAXY`,
  `zWarning=0`).
  - **Source:** SDSS SkyServer — https://skyserver.sdss.org/
  - **Credit:** *Funding for the Sloan Digital Sky Survey has been provided by
    the Alfred P. Sloan Foundation and the participating institutions.* See
    https://www.sdss.org/collaboration/citing-sdss/ for the acknowledgment.
- **`2dfgrs_slice.csv`** — 10000 galaxies (5000 from each of the North and South
  Galactic Pole strips) from the **2dF Galaxy Redshift Survey (2dFGRS)** final
  release, reliable redshifts (`q_z ≥ 3`, z < 0.20). Columns: `ra` (deg),
  `dec` (deg), `z` (redshift), `cap` (`NGP`/`SGP`). The two caps drive the
  double-sided "bowtie" diagram. Retrieved from VizieR catalog **VII/250**.
  - **Source:** VizieR — https://vizier.cds.unistra.fr/ (catalog VII/250)
  - **Reference / please cite:** Colless, M., et al. 2001, MNRAS, 328, 1039;
    Colless, M., et al. 2003, arXiv:astro-ph/0306581 (*The 2dF Galaxy Redshift
    Survey: final data release*).

## `query_cache/`

Cached live-query results used by the **Catalogs** tutorial's
try-live-else-cached pattern: the notebook runs every query live and falls
back to these small committed copies when offline (or when a service is
down), so it always executes and renders. To refresh one, delete the file and
re-execute the notebook with network access.

- **`simbad_m1.ecsv`** — `query_simbad("M1")` object lookup (the Crab Nebula's
  SIMBAD entry).
- **`simbad_m45_region.ecsv`** — SIMBAD region query within 40′ of the
  Pleiades (`MAIN_ID`, `RA`, `DEC`).
  - **Credit (both):** *This research has made use of the SIMBAD database,
    operated at CDS, Strasbourg, France* (Wenger et al. 2000, A&AS, 143, 9).
- **`gaia_m45.ecsv`, `gaia_m44.ecsv`, `gaia_m67.ecsv`** — **Gaia DR3**
  (VizieR **I/355**) cone searches around the Pleiades, the Beehive, and
  M67, trimmed to G < 16. Columns: `RA_ICRS`, `DE_ICRS`, `Gmag`, `BP-RP`.
  - **Reference / please cite:** Gaia Collaboration 2016, A&A, 595, A1 and
    Gaia Collaboration 2023, A&A, 674, A1 (*Gaia DR3*). *This work has made
    use of data from the ESA mission Gaia, processed by the Gaia DPAC.*
- **`vizier_ngc2000.ecsv`** — the complete **NGC 2000.0** catalog (VizieR
  **VII/118**; 13,226 NGC/IC entries) with VizieR's computed decimal
  positions (`_RAJ2000`, `_DEJ2000`) plus `Name`, `Type`, `mag`.
  - **Reference / please cite:** Sinnott, R. W. 1988, *NGC 2000.0* (Sky
    Publishing / Cambridge University Press).
- **`xsc_virgo.ecsv`** — the **2MASS Extended Source Catalog** (VizieR
  **VII/233**) within 4.5° of the Virgo Cluster (`RAJ2000`, `DEJ2000`,
  `K.ext`).
  - **Reference / please cite:** Jarrett, T. H., et al. 2000, AJ, 119, 2498;
    Skrutskie, M. F., et al. 2006, AJ, 131, 1163. *This publication makes use
    of data products from the Two Micron All Sky Survey.*
- **`skyview_dss2red_m51.fits`, `skyview_dss2red_virgo.fits`** — **DSS2 Red**
  cutouts from NASA **SkyView** (M51 at 0.35°; a 10.5° Virgo Cluster mosaic),
  stored as float32.
  - **Credit:** The Digitized Sky Survey (STScI/AURA, © 1993-5 by the
    Anglo-Australian Telescope Board and AURA); SkyView (McGlynn, T., et al.
    1998, in *Astrophysics and Algorithms*).
- **`hips_allwise_m51.fits`** — **AllWISE W1** cutout of the same M51 field
  from the CDS **hips2fits** service.
  - **Credit:** WISE (Wright, E. L., et al. 2010, AJ, 140, 1868); the CDS
    hips2fits service (https://alasky.cds.unistra.fr/hips-image-services/hips2fits).
- **`m74_sdss_g.fits`, `…_r.fits`, `…_i.fits`, `m74_sdss_color.png`** — 400²
  ~10′ cutouts of **M74 (NGC 628)** in SDSS g/r/i, plus the ready-made SDSS
  color image, from the CDS **hips2fits** service (`CDS/P/SDSS9/{g,r,i,color}`).
  Used by the **Markers** tutorial to build a spiral-galaxy icon from real
  survey data.
  - **Credit:** Sloan Digital Sky Survey. *Funding for the SDSS has been
    provided by the Alfred P. Sloan Foundation and the participating
    institutions* (see https://www.sdss.org/collaboration/citing-sdss/);
    served via the CDS hips2fits service.
- **`m74_wise_w1.fits`, `…_w2.fits`, `…_w3.fits`** — the same M74 field in
  **AllWISE** W1/W2/W3 (`CDS/P/allWISE/W{1,2,3}`), for the mid-IR composite
  in the same tutorial.
  - **Credit:** WISE (Wright, E. L., et al. 2010, AJ, 140, 1868); served via
    the CDS hips2fits service.

All VizieR-derived tables: *This research has made use of the VizieR catalogue
access tool, CDS, Strasbourg, France* (Ochsenbein et al. 2000, A&AS, 143, 23).

## `icons/`

Small marker images (Sun, Moon, planets, Earth, three instruments, a black hole)
for use with `imscatter` / globe annotations in the example notebooks. The
photographic icons are downsampled / cropped from the linked originals — mostly
public-domain space-agency imagery; one (`Mars`) is CC BY 2.0 and requires
credit plus a note that it was modified. The instrument and black-hole icons are
AI-generated illustrations created for this package (no third-party source
imagery).

The three instrument icons have a built-in pointing direction. Their **rest
angle** — the direction the business end faces as drawn, in degrees CCW from
screen-right — is what you subtract from a target bearing to aim them (see the
**Markers** tutorial): radio dish **125°**, optical telescope **65°**, space
telescope **194°**.

| File | Source | Credit | License |
|------|--------|--------|---------|
| `Earth_Western_Hemisphere_120pix.png` | NASA Visible Earth [#57723](https://visibleearth.nasa.gov/images/57723) ([Commons](https://commons.wikimedia.org/wiki/File:Earth_Western_Hemisphere.jpg)) | NASA / Reto Stöckli, Robert Simmon | Public domain (NASA) |
| `FullMoon_240x240.png` | NASA/JPL Galileo, PIA00405 ([Commons](https://commons.wikimedia.org/wiki/File:Full_moon.jpeg)) | NASA / JPL / USGS | Public domain (NASA) |
| `Jupiter_120pix.png` | NASA/ESA Hubble, 2014 ([Commons](https://commons.wikimedia.org/wiki/File:Jupiter_and_its_shrunken_Great_Red_Spot.jpg)) | NASA, ESA, A. Simon (GSFC) | Public domain (NASA/ESA Hubble) |
| `Mars_120pix.png` | NASA Hubble, 1999 ([Commons](https://commons.wikimedia.org/wiki/File:Mars_at_54_Million_Miles_from_Earth.jpg)) | NASA / Hubble (Steve Lee, Jim Bell, Mike Wolff) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) — *resized from original* |
| `sun1_120pix.png` | NOAA GOES-16 SUVI, 2022-10-06 ([Commons](https://commons.wikimedia.org/wiki/File:Sun_2022-10-06_2002Z.png)) | NOAA / GOES-16 | Public domain (NOAA) |
| `sun2_120pix.png` | NASA SDO/AIA, 2010-08-19 ([Commons](https://commons.wikimedia.org/wiki/File:The_Sun_by_the_Atmospheric_Imaging_Assembly_of_NASA%27s_Solar_Dynamics_Observatory_-_20100819.jpg)) | NASA / SDO (AIA 304 Å) | Public domain (NASA) |
| `RadioDish_250pix.png` | AI-generated (ChatGPT image generation, 2026), commissioned for skyplothelper | — | No known copyright (AI-generated; not derived from third-party imagery) |
| `OpticalTelescope_250pix.png` | AI-generated (ChatGPT image generation, 2026), commissioned for skyplothelper | — | No known copyright (AI-generated; not derived from third-party imagery) |
| `SpaceTelescope_250pix.png` | AI-generated (ChatGPT image generation, 2026), commissioned for skyplothelper | — | No known copyright (AI-generated; not derived from third-party imagery) |
| `SMBH_250pix.png` | AI-generated (ChatGPT image generation, 2026), commissioned for skyplothelper | — | No known copyright (AI-generated; not derived from third-party imagery) |

`Mars_120pix.png` is the only non-public-domain icon: under CC BY 2.0 it must
carry the credit above, a link to the license, and an indication that it was
modified (it has been downsampled to 120 px).

## `planet_maps/`

Equirectangular (plate carrée) body textures for globe rendering
(`make_planet_frame`, `imscatter`-style backgrounds). Resized / re-encoded
from the linked originals.

| File | Source | Credit | License |
|------|--------|--------|---------|
| `2k_*.jpg` (sun, mercury, venus_surface, moon, mars, jupiter, saturn, uranus, neptune, stars) | Solar System Scope — [solarsystemscope.com/textures](https://www.solarsystemscope.com/textures/) (based on NASA imagery / elevation data) | Solar System Scope | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `Io_full.jpg` | USGS Astrogeology — [Io Galileo/Voyager global mosaic](https://astrogeology.usgs.gov/search/map/Io/Voyager-Galileo/Io_GalileoSSI-Voyager_Global_Mosaic_ClrMerge_1km) | USGS / NASA (Galileo, Voyager) | Public domain (USGS/NASA) |
| `Ganymede.jpg` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Map_of_Ganymede_by_Bj%C3%B6rn_J%C3%B3nsson.jpg) — built from NASA Voyager/Galileo imagery | Björn Jónsson | Attribution license (credit "Björn Jónsson") |

The `2k_*` textures are CC BY 4.0 (credit Solar System Scope + link to the
license); everything else here is public domain or carries the attribution
shown.
