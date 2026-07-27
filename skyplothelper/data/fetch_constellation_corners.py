"""Regenerate ``constellation_corners.npz`` from Vizier VI/49.

Run from the repo root::

    python skyplothelper/data/fetch_constellation_corners.py

Replaces the bundled ``constellation_corners.npz`` next to this
script with a fresh fetch via astroquery. ~1 s on a typical machine
plus the network round-trip to Vizier.

Source catalog: Davenhall A.C. & Leggett S.K. (1989), *Outlines of
the IAU Constellations*, Vizier VI/49 (table ``bound_18``). The
underlying boundary definition is the IAU 1930 (Delporte) post-
revision, surfaced via Roman 1987 / Vizier VI/42.

The corner list contains 1,565 rows × 89 polygons (87 single-polygon
constellations + Serpens split into Caput and Cauda) in polygon-
traversal order. Consecutive corner pairs within the same polygon
share either RA or Dec at the B1875 epoch and are joined by a
parallel (constant-Dec) or meridian (constant-RA) edge — the IAU
convention.

Output schema::

    .npz file
      cst        : (N,) string-array — 3-char constellation code
      ra_b1875   : (N,) float32     — RA at the B1875 epoch (deg)
      dec_b1875  : (N,) float32     — Dec at the B1875 epoch (deg)
      ra_icrs    : (N,) float32     — RA at J2000/ICRS (deg, precessed)
      dec_icrs   : (N,) float32     — Dec at J2000/ICRS (deg, precessed)
      polygon_id : (N,) int16       — which polygon each row belongs to
      epoch      : str — 'B1875'
      source     : str — Vizier catalog ID

B1875 → ICRS precession is done locally via astropy.coordinates
(FK4 at equinox B1875 → ICRS) rather than the astroquery pre-
precessed columns, which sometimes mask prime-meridian rows as NaN.

This corner-list approach supersedes an earlier grid-scan regeneration
method and is the canonical source for the bundled boundary data.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from astroquery.vizier import Vizier


def fetch_corners(out_path: str | None = None) -> str:
    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "constellation_corners.npz",
        )

    print("Fetching VI/49/bound_18 from Vizier...")
    Vizier.ROW_LIMIT = -1
    Vizier.TIMEOUT = 60
    t0 = time.time()
    tab = Vizier.get_catalogs("VI/49")[0]
    print(f"  fetched {len(tab)} rows in {time.time() - t0:.1f}s")
    print(f"  columns: {tab.colnames}")

    # Walk the rows. Each constellation forms one or more closed
    # polygons; rows are listed in traversal order. Detect polygon
    # breaks by:
    #   a constellation change (new 'cst')
    #   OR a non-shared (RA, Dec) jump within the same constellation
    #     (i.e., consecutive corners share neither RA nor Dec — a sign
    #     of a polygon split, e.g., Serpens which has two parts).
    csts = [str(r["cst"]) for r in tab]
    ra_b = np.asarray([float(r["RAB1875"]) for r in tab], dtype=np.float64)
    dec_b = np.asarray([float(r["DEB1875"]) for r in tab], dtype=np.float64)
    cst_arr = np.asarray(csts, dtype="<U3")

    # Precess B1875 → J2000/ICRS ourselves. The astroquery-served
    # _RA.icrs / _DE.icrs columns sometimes mask rows at the prime
    # meridian as NaN; redoing the precession via astropy avoids that
    # and gives us tighter control over the epoch convention.
    print("Precessing B1875 → J2000 (ICRS) via astropy...")
    import astropy.units as u
    from astropy.coordinates import FK4, SkyCoord
    from astropy.time import Time
    coords_b1875 = SkyCoord(
        ra=ra_b * u.deg, dec=dec_b * u.deg,
        frame=FK4(equinox=Time("B1875.0", format="byear_str")),
    )
    coords_icrs = coords_b1875.transform_to("icrs")
    ra_i = coords_icrs.ra.to(u.deg).value.astype(np.float32)
    dec_i = coords_icrs.dec.to(u.deg).value.astype(np.float32)
    n_nan = int(np.isnan(ra_i).sum() + np.isnan(dec_i).sum())
    print(f"  NaN count after astropy precession: {n_nan}")
    ra_b = ra_b.astype(np.float32)
    dec_b = dec_b.astype(np.float32)

    # Assign polygon_id: increment whenever the constellation changes,
    # OR whenever consecutive corners (within the same constellation)
    # share neither RA nor Dec at the B1875 epoch (which would mean
    # the polygon edge isn't a parallel or meridian — a sign that
    # we've moved to a new sub-polygon).
    polygon_id = np.zeros(len(tab), dtype=np.int16)
    pid = 0
    eps = 1e-3
    for i in range(1, len(tab)):
        same_cst = csts[i] == csts[i - 1]
        share_ra = abs(ra_b[i] - ra_b[i - 1]) < eps
        share_dec = abs(dec_b[i] - dec_b[i - 1]) < eps
        if not same_cst:
            pid += 1
        elif not (share_ra or share_dec):
            pid += 1
        polygon_id[i] = pid

    n_polygons = polygon_id.max() + 1
    print(f"  detected {n_polygons} polygons across "
          f"{len(set(csts))} unique constellations")

    np.savez_compressed(
        out_path,
        cst=cst_arr,
        ra_b1875=ra_b,
        dec_b1875=dec_b,
        ra_icrs=ra_i,
        dec_icrs=dec_i,
        polygon_id=polygon_id,
        epoch="B1875",
        source="Vizier VI/49 — Davenhall & Leggett 1989",
    )
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nSaved {out_path} ({size_kb:.1f} KB, {len(tab)} corners)")
    return out_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    fetch_corners(out_path=out)
