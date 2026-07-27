"""Render the images helpers (quicklook_plot / simpleimageplot / make_norm /
rescale_image) for visual eyeballing.

Produces:
  - images_01_quicklook_default.png       — quicklook_plot, contours only
  - images_02_quicklook_image_contours.png — quicklook_plot with image+contours
  - images_03_simpleimageplot.png         — simpleimageplot output
  - images_04_stretches_grid.png          — same image rendered with all 7
                                              registered stretches
  - images_05_clip_methods.png            — same image with each clip method
  - images_06_norm_demo.png               — make_norm with several stretches

Usage
-----
    python render_images.py            # save PNGs to output/
    python render_images.py --show     # display interactively
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from _common import banner, save_or_show

from skyplothelper.images.levels import make_norm, rescale_image
from skyplothelper.images.quicklook import quicklook_figure, simpleimage_figure
from skyplothelper.wcs_frame import dummy_standard_hdr

PANELS = {}


def _panel(name):
    def deco(fn):
        PANELS[name] = fn
        return fn
    return deco


def _make_radio_image(nx=128, ny=128, seed=42, with_negative=True):
    """Synthesize a radio-style image: a few sources + Gaussian noise.
    Includes a small negative residual to demonstrate negative contours."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    img = (
        2.5 * np.exp(-((xx - nx // 2) ** 2 + (yy - ny // 2) ** 2) /
                     (2 * 7.0 ** 2)) +
        0.6 * np.exp(-((xx - nx // 2 - 22) ** 2 + (yy - ny // 2 + 18) ** 2) /
                     (2 * 5.0 ** 2)) +
        0.18 * np.exp(-((xx - nx // 2 + 30) ** 2 + (yy - ny // 2 - 30) ** 2) /
                      (2 * 9.0 ** 2)) +
        rng.normal(0, 0.05, (ny, nx))
    )
    if with_negative:
        img -= 0.25 * np.exp(-((xx - nx // 2 - 14) ** 2 +
                               (yy - ny // 2 - 14) ** 2) / (2 * 4 ** 2))
    return img


def _make_header(nx=128, ny=128):
    return dummy_standard_hdr(
        centercoords_deg=(180.0, 30.0),
        cdelts=(-1.0 / 3600, 1.0 / 3600),
        cunit="deg", projection="TAN",
        naxis_xy=(nx, ny),
    )


@_panel("images_01_quicklook_default")
def render_quicklook_default():
    img = _make_radio_image()
    hdr = _make_header()
    hdr["BUNIT"] = "Jy/beam"
    hdr["BMAJ"] = 5.0 / 3600
    hdr["BMIN"] = 4.0 / 3600
    hdr["BPA"] = 30.0
    hdr["OBJECT"] = "Demo Source"
    result = quicklook_figure(img, header=hdr,
                               figsize=(7, 7.5),
                               contours=True, image=False,
                               beam_style="crosshair",
                               color="C0", contour_lw="scaled",
                               show_info=True)
    return result.fig


@_panel("images_02_quicklook_image_contours")
def render_quicklook_image_contours():
    img = _make_radio_image(seed=11)
    hdr = _make_header()
    hdr["BUNIT"] = "Jy/beam"
    hdr["BMAJ"] = 5.0 / 3600
    hdr["BMIN"] = 4.0 / 3600
    hdr["BPA"] = 30.0
    result = quicklook_figure(img, header=hdr,
                               figsize=(7, 7.5),
                               contours=True, image=True,
                               colormap="inferno", stretch="asinh",
                               colorbar=True, color="w", info_color="k",
                               show_info=True)
    return result.fig


@_panel("images_03_simpleimageplot")
def render_simpleimageplot():
    """simpleimage_figure — minimal-styling figure builder."""
    img = _make_radio_image(seed=7)
    hdr = _make_header()
    result = simpleimage_figure(img, hdr,
                                 axtitle="simpleimage_figure demo",
                                 cmap="gist_yarg")
    return result.fig


@_panel("images_04_stretches_grid")
def render_stretches_grid():
    """Same image rendered with each of 7 registered stretches."""
    img = _make_radio_image(seed=2)
    stretches = ["linear", "sqrt", "squared", "log", "asinh", "sinh", "power"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for ax, stretch in zip(axes.flat, stretches):
        out = rescale_image(img, stretch=stretch, clip="percentile")
        ax.imshow(out, origin="lower", cmap="inferno")
        ax.set_title(stretch, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    axes.flat[-1].imshow(img, origin="lower", cmap="inferno")
    axes.flat[-1].set_title("(original, no stretch)", fontsize=10)
    axes.flat[-1].set_xticks([])
    axes.flat[-1].set_yticks([])
    fig.suptitle("rescale_image — same image with each registered stretch",
                 fontsize=13)
    fig.subplots_adjust(top=0.92, hspace=0.3, wspace=0.1)
    return fig


@_panel("images_05_clip_methods")
def render_clip_methods():
    """Same image with three different intensity-clip methods."""
    img = _make_radio_image(seed=3)
    methods = [
        ("percentile", {"plo": 1, "phi": 99}),
        ("sigma",      {"sigma": 3}),
        ("zscale",     {"contrast": 0.25}),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, (clip_method, kw) in zip(axes, methods):
        out = rescale_image(img, stretch="asinh", clip=clip_method, **kw)
        ax.imshow(out, origin="lower", cmap="inferno")
        ax.set_title(f"clip={clip_method!r}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("rescale_image — three intensity-clip methods (asinh stretch)",
                 fontsize=13)
    fig.subplots_adjust(top=0.88, wspace=0.1)
    return fig


@_panel("images_06_norm_demo")
def render_norm_demo():
    """make_norm with several stretches via imshow's `norm=` parameter."""
    img = _make_radio_image(seed=4)
    stretches = ["linear", "sqrt", "log", "asinh", "power"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for ax, stretch in zip(axes, stretches):
        norm = make_norm(stretch=stretch, data=img)
        ax.imshow(img, origin="lower", cmap="viridis", norm=norm)
        ax.set_title(f"make_norm(stretch={stretch!r})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("make_norm — produces a Normalize for imshow norm= "
                 "(same image, 5 stretches)", fontsize=12)
    fig.subplots_adjust(top=0.85, wspace=0.1)
    return fig


def main():
    banner("images — merge-verification visual gallery")
    for name, builder in PANELS.items():
        save_or_show(builder(), name)
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
