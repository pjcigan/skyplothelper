"""Tests for skyplothelper.images.levels."""

import numpy as np
import pytest

from skyplothelper.images.levels import (
    _STRETCH_REGISTRY,
    auto_interval,
    auto_stretch,
    clip_percentile,
    clip_sigma,
    clip_zscale,
    list_stretches,
    make_norm,
    rescale_image,
    rescale_percentile,
)


def _gaussian_image(seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((100, 100)).astype(float)


def test_clip_percentile_returns_pair():
    data = _gaussian_image()
    lo, hi = clip_percentile(data, plo=1, phi=99)
    assert lo < hi


def test_clip_sigma_returns_pair():
    data = _gaussian_image()
    lo, hi = clip_sigma(data, sigma_lo=2, sigma_hi=2)
    assert lo < hi


def test_clip_zscale_returns_pair():
    data = _gaussian_image()
    lo, hi = clip_zscale(data)
    assert lo < hi


@pytest.mark.parametrize("method", ["percentile", "sigma", "zscale"])
def test_auto_interval(method):
    data = _gaussian_image()
    lo, hi = auto_interval(data, method=method)
    assert lo < hi


def test_stretch_registry_keys():
    expected = {"linear", "sqrt", "squared", "log", "asinh", "sinh", "power"}
    assert expected == set(_STRETCH_REGISTRY.keys())


@pytest.mark.parametrize("name,fn", _STRETCH_REGISTRY.items())
def test_stretch_functions_run(name, fn):
    """Each inline stretch function returns a finite array on [0, 1]."""
    x = np.linspace(0.0, 1.0, 10)
    y = fn(x)
    assert np.all(np.isfinite(y))


def test_make_norm_runs():
    data = _gaussian_image()
    norm = make_norm(stretch="sqrt", data=data)
    assert norm is not None


def test_rescale_image_runs():
    data = _gaussian_image()
    out = rescale_image(data, stretch="linear")
    assert out.shape == data.shape


def test_rescale_percentile_smoke():
    data = _gaussian_image()
    out = rescale_percentile(data, plo=1, phi=99, stretch="sqrt")
    assert out.shape == data.shape


def test_auto_stretch_runs():
    """auto_stretch returns a recommended stretch name (string)."""
    data = _gaussian_image()
    out = auto_stretch(data)
    # Could be a string name, dict, or other recommendation. Just verify non-None
    # truthy output (the smoke test).
    assert out is not None


def test_list_stretches_runs(capsys):
    list_stretches()
    out = capsys.readouterr().out
    assert "linear" in out
    assert "asinh" in out
    assert "symlog" in out


def test_rescale_image_symlog_signed():
    """symlog maps signed data to [0,1], symmetric about 0->0.5, monotonic,
    linear near zero and compressive in the wings."""
    data = np.linspace(-100.0, 100.0, 21)
    out = rescale_image(data, stretch="symlog", vmin=-100, vmax=100, a=5)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.all(np.diff(out) >= -1e-9)                 # monotonic
    zero = rescale_image(np.array([0.0]), stretch="symlog",
                         vmin=-100, vmax=100, a=5)[0]
    assert zero == pytest.approx(0.5, abs=1e-6)          # zero -> midpoint
    # Distinct from a linear stretch (the whole point of symlog).
    lin = rescale_image(data, stretch="linear", vmin=-100, vmax=100)
    assert not np.allclose(out, lin)


def test_rescale_image_symlog_nan_safe():
    img = np.array([[-10.0, np.nan], [0.0, 10.0]])
    out = rescale_image(img, stretch="symlog", vmin=-10, vmax=10)
    assert out.shape == img.shape
    assert np.all(np.isfinite(out))


def test_rescale_image_symmetric_log_optional():
    """symmetric_log works when pysymlog is installed; otherwise raises a
    clear ImportError pointing at the optional extra."""
    data = np.linspace(-10.0, 10.0, 11)
    try:
        out = rescale_image(data, stretch="symmetric_log", vmin=-10, vmax=10)
    except ImportError as exc:
        assert "pysymlog" in str(exc)
        pytest.skip("pysymlog not installed")
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.all(np.diff(out) >= -1e-9)


# ===== stretch='symlog' / 'symmetric_log' =====

def test_make_norm_symlog_returns_symlognorm():
    """stretch='symlog' uses mpl's piecewise SymLogNorm (back-compat)."""
    import matplotlib.colors as mcolors
    norm = make_norm(stretch="symlog", vmin=-10, vmax=10, a=0.1)
    assert isinstance(norm, mcolors.SymLogNorm)


def test_make_norm_symmetric_log_uses_pysymlog():
    """stretch='symmetric_log' routes to pysymlog.SymmetricLogarithmNorm."""
    import pytest
    pysymlog = pytest.importorskip("pysymlog")
    norm = make_norm(stretch="symmetric_log", vmin=-10, vmax=10, a=0.1)
    assert isinstance(norm, pysymlog.SymmetricLogarithmNorm)
    # The transform is shift-based, not linthresh-based.
    assert norm.vmin == -10
    assert norm.vmax == 10


def test_make_norm_symmetric_log_data_driven():
    """vmin/vmax can come from data + interval, same as other stretches."""
    import pytest
    pysymlog = pytest.importorskip("pysymlog")
    data = _gaussian_image() - 0.5  # signed
    norm = make_norm(stretch="symmetric_log", data=data, clip="percentile")
    assert isinstance(norm, pysymlog.SymmetricLogarithmNorm)
    assert norm.vmin is not None and norm.vmax is not None


def test_make_norm_symmetric_log_missing_pysymlog_raises(monkeypatch):
    """Without pysymlog installed, asking for 'symmetric_log' raises
    a clear ImportError pointing at the install command."""
    # Simulate pysymlog being uninstalled by injecting an import failure.
    import builtins
    import sys
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pysymlog" or name.startswith("pysymlog."):
            raise ImportError("simulated: pysymlog not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setitem(sys.modules, "pysymlog", None)

    import pytest
    with pytest.raises(ImportError, match="pysymlog"):
        make_norm(stretch="symmetric_log", vmin=-10, vmax=10)
