"""Tests for the animation save helpers: WebPWriter + save_animation.

These are the public parts of ``skyplothelper.globe.animation`` (the globe
animators themselves are demo helpers and aren't exported). They cover the
animated-WebP writer, extension-based writer selection (incl. the splitext
regression), fps derivation, and transparency handling — no ffmpeg needed
(WebP/GIF go through Pillow).
"""

import matplotlib

matplotlib.use("Agg")

import os  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402

import skyplothelper as sph  # noqa: E402
from skyplothelper.globe.animation import (  # noqa: E402
    WebPWriter,
    _fps_from_animation,
    _tqdm_disable,
    save_animation,
)

Image = pytest.importorskip("PIL.Image")


def _make_ani(nframes=4, interval=100):
    """A tiny FuncAnimation: one marker moving across a transparent axes."""
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    (dot,) = ax.plot([], [], "o", ms=20)

    def update(i):
        dot.set_data([i / float(nframes)], [0.5])
        return (dot,)

    return FuncAnimation(fig, update, frames=nframes, interval=interval,
                         blit=True), fig


def test_public_exports_present():
    assert sph.WebPWriter is WebPWriter
    assert sph.save_animation is save_animation


def test_save_animation_writes_animated_webp_with_alpha(tmp_path):
    ani, fig = _make_ani(nframes=4)
    out = os.path.join(tmp_path, "spin.webp")
    save_animation(ani, fig, out, bgcolor="transparent")
    im = Image.open(out)
    assert im.format == "WEBP"
    assert getattr(im, "is_animated", False)
    assert im.n_frames == 4
    # Transparent frames -> 8-bit alpha (RGBA), the whole point vs GIF's 1-bit.
    assert im.mode == "RGBA" or "transparency" in im.info


def test_save_animation_splitext_not_substring(tmp_path):
    """'clip.gif.webp' must select the WebP writer, not misfire to the GIF
    writer as the old `'.gif' in savepath` substring test would."""
    ani, fig = _make_ani()
    out = os.path.join(tmp_path, "clip.gif.webp")
    save_animation(ani, fig, out, bgcolor="transparent")
    assert Image.open(out).format == "WEBP"


def test_save_animation_gif_routes_to_pillow(tmp_path):
    ani, fig = _make_ani()
    out = os.path.join(tmp_path, "clip.gif")
    save_animation(ani, fig, out, bgcolor="transparent")
    assert Image.open(out).format == "GIF"


def test_webpwriter_lossless_and_lossy_both_write(tmp_path):
    for name, lossless in (("lossy.webp", False), ("lossless.webp", True)):
        ani, fig = _make_ani()
        out = os.path.join(tmp_path, name)
        ani.save(out, writer=WebPWriter(fps=10, lossless=lossless))
        plt.close("all")
        assert Image.open(out).format == "WEBP"


def test_webpwriter_preserves_distinct_frames(tmp_path):
    """The moving marker means consecutive frames differ — the writer must
    keep them all, not collapse to a still."""
    ani, fig = _make_ani(nframes=5)
    out = os.path.join(tmp_path, "multi.webp")
    save_animation(ani, fig, out, bgcolor="transparent")
    im = Image.open(out)
    im.seek(0)
    first = im.convert("RGBA").tobytes()
    im.seek(im.n_frames - 1)
    last = im.convert("RGBA").tobytes()
    assert first != last


def test_tqdm_disable_mapping():
    """progress=None -> tqdm auto (disable=None, hides on non-TTY like
    nbconvert/CI); True/False force the bar on/off."""
    assert _tqdm_disable(None) is None
    assert _tqdm_disable(True) is False
    assert _tqdm_disable(False) is True


def test_fps_from_animation():
    class _Ani:
        _interval = 50

    class _NoInterval:
        pass

    assert _fps_from_animation(_Ani(), None) == 20.0        # 1000 / 50
    assert _fps_from_animation(_Ani(), 12) == 12.0          # explicit wins
    assert _fps_from_animation(_NoInterval(), None) == 10.0  # fallback


def test_save_animation_opaque_bgcolor(tmp_path):
    """A non-transparent bgcolor fills the background (no alpha demanded)."""
    ani, fig = _make_ani()
    out = os.path.join(tmp_path, "opaque.webp")
    save_animation(ani, fig, out, bgcolor="white")
    assert Image.open(out).format == "WEBP"


def test_save_animation_force_writer_overrides_extension(tmp_path):
    """force_writer bypasses extension selection — a .webp path forced to
    pillow writes a (Pillow-native) GIF-style animation instead."""
    ani, fig = _make_ani()
    out = os.path.join(tmp_path, "forced.gif")
    save_animation(ani, fig, out, bgcolor="transparent", force_writer="pillow")
    assert Image.open(out).format == "GIF"
