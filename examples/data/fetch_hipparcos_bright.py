#!/usr/bin/env python
"""Regenerate ``hipparcos_bright_pm.csv`` — the naked-eye Hipparcos catalog.

The 4,992 stars brighter than V = 6 with complete astrometry, from the
Hipparcos main catalogue (VizieR ``I/239/hip_main``, ESA 1997). Columns:

    HIP      Hipparcos identifier
    RAICRS   right ascension  (deg, ICRS)
    DEICRS   declination      (deg, ICRS)
    Vmag     Johnson V magnitude (mag)
    Plx      trigonometric parallax (mas)
    pmRA     proper motion in RA, mu_alpha* = mu_alpha cos(delta) (mas/yr)
    pmDE     proper motion in Dec, mu_delta (mas/yr)
    BV       Johnson B-V color index (mag)   <-- added 2026-07; see note below

Used by the tutorial notebooks (Vector Fields, Constellations, Animations,
Interactive Plotting) and the paper figures as a real, recognizable stellar
sample. The ``BV`` column drives the perceived-star-color rendering
(``sph.bv_to_rgb``) in the Constellations opener; it is 3-decimal Johnson
B-V and is blank for the two stars (HIP 26220, 32609) that lack a catalogued
color. Every other column and every row is byte-for-byte identical to the
original file — the selection is ``Vmag < 6`` AND non-null RA/Dec/Plx/pmRA/
pmDE, sorted by RA — so downstream figures are unaffected by the addition.

Run (needs ``astroquery``)::

    python fetch_hipparcos_bright.py            # writes hipparcos_bright_pm.csv

Re-running is deterministic: VizieR returns the fixed ESA 1997 catalogue.
"""
from __future__ import annotations

import os

import pandas as pd
from astroquery.vizier import Vizier

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "hipparcos_bright_pm.csv")
_COLS = ["HIP", "RAICRS", "DEICRS", "Vmag", "Plx", "pmRA", "pmDE", "BV"]


def fetch() -> pd.DataFrame:
    """Return the assembled catalog as a DataFrame with columns ``_COLS``."""
    v = Vizier(catalog="I/239/hip_main",
               columns=["HIP", "RAICRS", "DEICRS", "Vmag", "Plx",
                        "pmRA", "pmDE", "B-V"])
    v.ROW_LIMIT = -1
    df = v.query_constraints(Vmag="<6")[0].to_pandas().rename(
        columns={"B-V": "BV"})
    # Naked-eye + complete astrometry (drops 3 V<6 stars with no RA/Dec/Plx/pm).
    df = df.dropna(subset=["RAICRS", "DEICRS", "Plx", "pmRA", "pmDE"])
    df = df.sort_values("RAICRS", kind="stable").reset_index(drop=True)
    return df[_COLS]


def write(df: pd.DataFrame, path: str = OUT) -> None:
    """Write with the catalog's fixed precision (coords 8 dp, rest 2 dp,
    B-V 3 dp; blank where B-V is absent)."""
    out = pd.DataFrame({
        "HIP": df["HIP"].astype(int),
        "RAICRS": df["RAICRS"].map("{:.8f}".format),
        "DEICRS": df["DEICRS"].map("{:.8f}".format),
        "Vmag": df["Vmag"].map("{:.2f}".format),
        "Plx": df["Plx"].map("{:.2f}".format),
        "pmRA": df["pmRA"].map("{:.2f}".format),
        "pmDE": df["pmDE"].map("{:.2f}".format),
        "BV": df["BV"].map(lambda x: "" if pd.isna(x) else f"{x:.3f}"),
    })
    out.to_csv(path, index=False)


if __name__ == "__main__":
    write(fetch())
    print(f"wrote {OUT}")
