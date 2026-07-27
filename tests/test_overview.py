"""Tests for skyplothelper.overview / find and the recipe catalog.

The important assertion is that EVERY recipe snippet in the capability catalog
actually runs — an agent-facing recipe that's wrong is worse than none.
"""

import matplotlib
import numpy as np
import pytest
from astropy.io import fits

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import skyplothelper as sph  # noqa: E402
from skyplothelper._overview import RECIPES  # noqa: E402


def _bindings(tmp_path):
    """Synthetic user-data placeholders + temp FITS the recipes reference."""
    rng = np.random.RandomState(0)
    n = 200
    img = rng.random((40, 60)).astype(float)
    hdr = fits.Header({
        "NAXIS": 2, "NAXIS1": 60, "NAXIS2": 40,
        "CTYPE1": "RA---SIN", "CRVAL1": 150., "CRPIX1": 30., "CDELT1": -0.01,
        "CUNIT1": "deg",
        "CTYPE2": "DEC--SIN", "CRVAL2": 2., "CRPIX2": 20., "CDELT2": 0.01,
        "CUNIT2": "deg",
        "BUNIT": "Jy/beam", "BMAJ": 5 / 3600, "BMIN": 3 / 3600, "BPA": 30.,
        "OBJECT": "Demo"})
    fits.writeto(tmp_path / "image.fits", img, hdr, overwrite=True)
    cube = rng.random((10, 40, 60)).astype(float)
    chdr = hdr.copy()
    chdr["NAXIS"], chdr["NAXIS3"] = 3, 10
    chdr["CTYPE3"], chdr["CRVAL3"], chdr["CRPIX3"] = "VRAD", 0., 1.
    chdr["CDELT3"], chdr["CUNIT3"] = 1000., "m/s"
    fits.writeto(tmp_path / "cube.fits", cube, chdr, overwrite=True)
    plt.imsave(str(tmp_path / "milkyway.jpg"), rng.random((90, 180, 3)))
    # Equirectangular RGB texture for the globe-drape recipe (2:1, geo=True).
    plt.imsave(str(tmp_path / "earth_texture.jpg"), rng.random((90, 180, 3)))

    ns = {"np": np, "plt": plt,
          "ra": rng.uniform(0, 360, n), "dec": rng.uniform(-80, 80, n),
          "lon": rng.uniform(0, 360, n), "lat": rng.uniform(-80, 80, n),
          "flux": rng.uniform(1, 10, n), "mag": rng.uniform(5, 15, n),
          "teff": rng.uniform(3000, 10000, n),
          "redshift": rng.uniform(0.0, 0.1, n),
          # small rms so 3*rms doesn't mask the whole synthetic [0,1) cube
          # (an all-NaN moment-1 would trip a benign nanvar warning).
          "rms": 0.05, "nside": 16,
          "image": img, "header": hdr,
          "catalog": {"ra": rng.uniform(0, 360, n),
                      "dec": rng.uniform(-80, 80, n),
                      "mag": rng.uniform(5, 15, n),
                      "flux": rng.uniform(1, 10, n)}}
    try:
        import healpy as hp
        ns["pixel_indices"] = np.arange(hp.nside2npix(16))[::7]
        ns["values"] = rng.random(len(ns["pixel_indices"]))
    except ImportError:
        pass
    return ns


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda r: r.task[:40])
def test_recipe_runs(recipe, tmp_path, monkeypatch):
    """Each catalog recipe executes cleanly (skipped only if an OPTIONAL
    dependency it needs isn't installed)."""
    monkeypatch.chdir(tmp_path)
    ns = _bindings(tmp_path)
    if "pixel_indices" not in ns and "healpix" in recipe.category:
        pytest.skip("healpy not installed")
    # strip interactive .show() calls (headless)
    code = "\n".join(ln for ln in recipe.code.splitlines()
                     if not ln.strip().endswith(".show()"))
    try:
        exec(code, dict(ns))
    except ImportError as exc:
        pytest.skip(f"optional dependency missing: {exc}")
    finally:
        plt.close("all")


def test_overview_prints(capsys):
    sph.overview()
    out = capsys.readouterr().out
    assert "FRAME-FIRST" in out and "TASK INDEX" in out


def test_overview_as_dict():
    d = sph.overview(as_dict=True)
    assert set(d) == {"frame_first", "conventions", "recipes"}
    assert len(d["recipes"]) == len(RECIPES)
    assert all({"task", "category", "functions", "code"} <= set(r)
               for r in d["recipes"])


def test_recipes_matches(capsys):
    sph.recipes("cube")
    out = capsys.readouterr().out
    assert "channel_map" in out.lower() or "moment" in out.lower()


def test_recipes_menu_lists_all(capsys):
    sph.recipes()
    out = capsys.readouterr().out
    assert "adjusting & legibility" in out and "recipes" in out


def test_recipes_covers_adjustment_queries(capsys):
    """The 'how do I adjust it' knobs are discoverable, not just show-recipes."""
    for kw, expect in [("grid", "spacing"), ("stroke", "stroke"),
                       ("tick", "format_ticklabels"), ("colorbar", "colorbar")]:
        sph.recipes(kw)
        assert expect in capsys.readouterr().out.lower()


def test_recipes_no_match(capsys):
    sph.recipes("zzzznope")
    assert "No recipe matched" in capsys.readouterr().out


def test_overview_query_delegates_to_recipes(capsys):
    sph.overview("healpix")
    assert "healpix" in capsys.readouterr().out.lower()


@pytest.mark.parametrize("name,full", [("llms.txt", False),
                                       ("llms-full.txt", True)])
def test_llms_txt_in_sync(name, full):
    """The committed llms.txt / llms-full.txt must match a fresh render of the
    catalog — otherwise run `python scripts/make_llms_txt.py`."""
    from pathlib import Path

    from skyplothelper._overview import render_llms

    root = Path(__file__).resolve().parent.parent
    committed = (root / name).read_text()
    assert committed == render_llms(full=full), (
        f"{name} is stale — regenerate with `python scripts/make_llms_txt.py`")


def test_llms_txt_carries_recipe_code():
    """llms.txt must contain runnable CODE, not just a function index (an
    agent reading only the concise file otherwise guesses arguments)."""
    from skyplothelper._overview import render_llms
    text = render_llms(full=False)
    assert "```python" in text
    assert "ra_col=" in text and "frame='galactic'" in text
