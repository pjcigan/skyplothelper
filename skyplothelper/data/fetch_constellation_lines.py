"""Regenerate ``constellation_lines.npz`` from the d3-celestial dataset.

Run from the repo root::

    python skyplothelper/data/fetch_constellation_lines.py

Replaces the bundled ``constellation_lines.npz`` next to this
script with a fresh download from the d3-celestial GitHub raw URL.

Source: https://github.com/ofrohn/d3-celestial — ``data/
constellations.lines.json`` (BSD-licensed). The underlying line
geometry comes from the IAU Constellation page with name-position
and minor line modifications by Olaf Frohn. The IAU itself doesn't
publish a single canonical asterism (connect-the-dots) dataset
distinct from the constellation boundaries; this collection is the
de-facto IAU-derived public-domain-compatible reference.

The source JSON is GeoJSON ``FeatureCollection`` with one
``Feature`` per constellation:

    {
      "type": "Feature",
      "id": "Ori",
      "properties": {"rank": 1},
      "geometry": {
        "type": "MultiLineString",
        "coordinates": [
          [[lon0, lat0], [lon1, lat1], ...],   # one polyline
          [[lon0, lat0], [lon1, lat1], ...],   # another polyline
          ...
        ]
      }
    }

Coordinates are equatorial (RA, Dec) in **decimal degrees**, with
RA on the [-180, +180] branch around lon=0. We re-wrap to the
[0, 360) convention to match the rest of the package.

Output schema::

    .npz file
      cst         : (N,) string-array — 3-char constellation code (uppercase)
      rank        : (N,) int8         — line prominence rank (1 = most prominent)
      ra          : (V,) float32      — RA at J2000/ICRS (deg, [0, 360))
      dec         : (V,) float32      — Dec at J2000/ICRS (deg)
      seg_offsets : (N+1,) int32      — index into ra/dec of each segment's
                                          first/last+1 vertex (so a single
                                          polyline is ``ra[seg_offsets[i]:
                                          seg_offsets[i+1]]`` etc.)
      cst_seg_ids : (N,) int32        — which constellation each segment
                                          belongs to (parallel-indexed to cst)
      source      : str — d3-celestial constellations.lines.json (BSD;
                          derived from IAU Constellation page +
                          Olaf Frohn modifications)
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

import numpy as np

DEFAULT_URL = (
    "https://raw.githubusercontent.com/ofrohn/d3-celestial/master/"
    "data/constellations.lines.json"
)


def fetch_lines(out_path: str | None = None,
                source_url: str = DEFAULT_URL) -> str:
    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "constellation_lines.npz",
        )

    print(f"Fetching {source_url} ...")
    t0 = time.time()
    with urllib.request.urlopen(source_url, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
    payload = json.loads(raw)
    print(f"  fetched {len(raw) / 1024:.1f} KB in {time.time() - t0:.1f}s")

    features = payload["features"]
    print(f"  {len(features)} features (one per IAU constellation)")

    # Per-segment unique constellation code + rank, plus flat arrays of
    # RA / Dec vertices and segment offsets into them.
    cst_codes: list[str] = []   # one entry per constellation, in walk order
    rank_per_cst = []       # parallel to cst_codes
    cst_seg_ids = []        # one entry per *segment* — pointer back to cst
    seg_offsets = [0]
    all_ra = []
    all_dec = []

    for feat in features:
        abbr = str(feat["id"]).upper()
        rank = int(feat.get("properties", {}).get("rank", 0))
        geom = feat.get("geometry", {})
        if geom.get("type") != "MultiLineString":
            continue
        segments = geom.get("coordinates", [])
        if not segments:
            continue
        cst_idx = len(cst_codes)
        cst_codes.append(abbr)
        rank_per_cst.append(rank)
        for seg in segments:
            for lon, lat in seg:
                # Wrap [-180, 180] → [0, 360)
                lon = float(lon) % 360.0
                all_ra.append(lon)
                all_dec.append(float(lat))
            seg_offsets.append(len(all_ra))
            cst_seg_ids.append(cst_idx)

    cst_arr = np.asarray(cst_codes, dtype="<U3")
    rank_arr = np.asarray(rank_per_cst, dtype=np.int8)
    ra_arr = np.asarray(all_ra, dtype=np.float32)
    dec_arr = np.asarray(all_dec, dtype=np.float32)
    seg_offsets_arr = np.asarray(seg_offsets, dtype=np.int32)
    cst_seg_ids_arr = np.asarray(cst_seg_ids, dtype=np.int32)

    n_segments = len(seg_offsets_arr) - 1
    n_vertices = len(ra_arr)
    print(f"  parsed {len(cst_arr)} constellations / {n_segments} segments / "
          f"{n_vertices} vertices")

    np.savez_compressed(
        out_path,
        cst=cst_arr,
        rank=rank_arr,
        ra=ra_arr,
        dec=dec_arr,
        seg_offsets=seg_offsets_arr,
        cst_seg_ids=cst_seg_ids_arr,
        source=("d3-celestial constellations.lines.json (BSD; derived from "
                "IAU Constellation page + Olaf Frohn modifications)"),
    )
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nSaved {out_path} ({size_kb:.1f} KB)")
    return out_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    fetch_lines(out_path=out)
