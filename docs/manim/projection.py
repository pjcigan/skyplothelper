"""Flagship manim explainer for tutorial #2 — *what a projection actually is*.

Told the way the light actually travels. **You are at the center** of the
celestial sphere. Orion's stars fly in at their *real* Hipparcos distances;
since all we ever recover is their *direction*, they collapse onto the sphere.
Then their light streams **inward** to us, and where each ray crosses a flat
"window" pane held up toward the patch is its place on a map — a gnomonic
(``TAN``) projection, which is exactly a pinhole/perspective view from the eye.
The camera flies to the center and looks **out** through the pane, dissolving
into the **real** skyplothelper ``TAN`` chart; a coda flattens the whole
surrounding sphere into a real all-sky Aitoff map.

This is the manim side only. It draws *no* science map: every "real map" beat is
a genuine sph render imported as an ``ImageMobject`` (``assets/tan_orion.png``,
``assets/allsky_aitoff.png``), and the star directions, colors, and distances
come from ``assets/orion_stars.json`` — all produced by
``make_projection_assets.py`` in the astropy env. See
``.claude/MANIM_DEMO_BRIEFS.md`` §A / §4 and ``README.md``.

Build (in the optional ManimCE env, from the repo root)::

    manim -qm --fps 30 --media_dir docs/manim/media docs/manim/projection.py Projection

then re-encode the mp4 for the web and grab a poster still (see ``README.md`` for
the exact ffmpeg lines and the `<video>` embed). ManimCE 0.19.x. Renders on the
shared navy sky canvas so the sph renders blend in.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
from manim import (
    DEGREES,
    DR,
    ORIGIN,
    OUT,
    UP,
    Create,
    Dot3D,
    FadeIn,
    FadeOut,
    ImageMobject,
    LaggedStart,
    Line,
    ParametricFunction,
    Square,
    Text,
    ThreeDScene,
    VGroup,
)

SKY = "#16203A"
GRID = "#4E6188"
EYE = "#E2C275"             # observer / projection-point gold (that point is us)
LABEL = "#D7DEE8"           # light label text on navy
HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
# A committed branding asset (light wordmark + Aitoff mark on transparency), so a
# downloaded/reused clip carries attribution. Read straight from _static/logo.
LOGO = os.path.join(HERE, "..", "_static", "logo",
                    "logo_6_wordmark-dark_mark.png")

R_SPHERE = 2.6              # celestial sphere (where directions are plotted)
R_WIN = 1.5                # the flat "window" pane, held between us and the sky
NEAR, FAR = 1.7, 3.8       # scene radii the real near/far distances map onto

# Flip the east axis if Orion comes out mirrored (sky convention = east-left).
FLIP_E = -1.0


def _load_stars():
    """Read the sph-emitted stars and build, for each, three scene positions:
    where it starts (its *real* distance, log-compressed into the scene), where
    it lands once collapsed onto the celestial sphere, and where its inward ray
    crosses the window pane (the gnomonic map point). Geometry is done in a local
    frame whose +Z is the tangent point, so the pane is the flat plane z=R_WIN."""
    with open(os.path.join(ASSETS, "orion_stars.json")) as fh:
        d = json.load(fh)
    ra0, dec0 = np.radians(d["center_radec"])
    t_hat = np.array([np.cos(dec0) * np.cos(ra0),
                      np.cos(dec0) * np.sin(ra0), np.sin(dec0)])
    e_east = np.array([-np.sin(ra0), np.cos(ra0), 0.0])
    e_north = np.array([-np.sin(dec0) * np.cos(ra0),
                        -np.sin(dec0) * np.sin(ra0), np.cos(dec0)])

    raw = []
    for s in d["stars"]:
        u = np.array(s["u"])
        d_hat = np.array([FLIP_E * (u @ e_east), u @ e_north, u @ t_hat])
        if d_hat[2] <= 0.05:                 # behind the tangent hemisphere
            continue
        raw.append((d_hat, s))

    dists = [s["dist_pc"] for _, s in raw if s["dist_pc"]]
    lo, hi = math.log(min(dists)), math.log(max(dists))
    stars = []
    for d_hat, s in raw:
        d_pc = s["dist_pc"]
        frac = (math.log(d_pc) - lo) / (hi - lo) if d_pc else 0.5
        r0 = NEAR + (FAR - NEAR) * frac                 # near..far by real dist
        lz = d_hat[2]
        stars.append({
            "start": r0 * d_hat,
            "sphere": R_SPHERE * d_hat,
            "plane": np.array([R_WIN * d_hat[0] / lz,
                               R_WIN * d_hat[1] / lz, R_WIN]),
            "color": s["color"], "size": s["size"],
        })
    return stars


def _graticule(radius):
    """A faint lon/lat wireframe on the celestial sphere (pole = tangent point)."""
    curves = VGroup()
    for phi in np.linspace(0, 2 * np.pi, 12, endpoint=False):        # meridians
        curves.add(ParametricFunction(
            lambda t, phi=phi: radius * np.array(
                [np.sin(t) * np.cos(phi), np.sin(t) * np.sin(phi), np.cos(t)]),
            t_range=(0.001, np.pi - 0.001), stroke_width=1.1,
            stroke_opacity=0.30).set_color(GRID))
    for theta in np.linspace(np.pi / 6, 5 * np.pi / 6, 5):           # parallels
        curves.add(ParametricFunction(
            lambda t, theta=theta: radius * np.array(
                [np.sin(theta) * np.cos(t), np.sin(theta) * np.sin(t),
                 np.cos(theta)]),
            t_range=(0, 2 * np.pi), stroke_width=1.1,
            stroke_opacity=0.30).set_color(GRID))
    return curves


class Projection(ThreeDScene):
    def construct(self):
        self.camera.background_color = SKY
        stars = _load_stars()
        smax = max(s["size"] for s in stars)

        def star_radius(size):
            return 0.05 + 0.10 * (size / smax) ** 0.5

        # ---- geometry ---------------------------------------------------------
        eye = Dot3D(ORIGIN, radius=0.11, color=EYE)      # the observer = us
        graticule = _graticule(R_SPHERE)
        pane = Square(side_length=1.7, fill_color="#9FB3D0", fill_opacity=0.12,
                      stroke_color=GRID, stroke_width=1.4).move_to(R_WIN * OUT)

        star_mobs = [Dot3D(s["start"], radius=star_radius(s["size"]),
                           color=s["color"]) for s in stars]
        rays, landed = [], []
        for s in stars:
            rays.append(Line(s["sphere"], ORIGIN, stroke_width=1.4,
                             stroke_opacity=0.40).set_color(s["color"]))
            landed.append(Dot3D(s["plane"], radius=star_radius(s["size"]),
                                color=s["color"]))
        order = np.argsort([-s["size"] for s in stars])   # brightest first

        # ---- captions + real-map PNGs, fixed to the frame ---------------------
        self._cap = None

        def caption(txt, run_time=0.7):
            # Sequential (fade old fully out, then new in) so two captions never
            # linger superimposed — a clean swap rather than a crossfade.
            old = self._cap
            if old is not None:
                self.play(FadeOut(old), run_time=run_time * 0.4)
                self.remove(old)
            new = Text(txt, font="sans-serif", color=LABEL).scale(0.5)
            new.to_edge(UP, buff=0.3).set_opacity(0.0)
            self.add_fixed_in_frame_mobjects(new)
            self.play(new.animate.set_opacity(1.0), run_time=run_time * 0.6)
            self._cap = new

        tan_png = ImageMobject(os.path.join(ASSETS, "tan_orion.png"))
        tan_png.height = 7.0
        # The two coda frames are the SAME Orion-centered all-sky Aitoff — a bare
        # grid, then the sky image draped on it — so Orion is pixel-aligned and
        # never moves between them.
        grid_png = ImageMobject(os.path.join(ASSETS, "allsky_orion_grid.png"))
        grid_png.width = 12.8
        raster_png = ImageMobject(os.path.join(ASSETS, "allsky_orion_raster.jpg"))
        raster_png.width = 12.8

        # Persistent branding watermark, fixed in the bottom-right corner — which
        # stays navy in every beat (the Aitoff oval and TAN chart never reach the
        # corners), so the light wordmark reads throughout and a downloaded clip
        # carries attribution. Faded in with the opener and out at the loop end.
        logo = ImageMobject(LOGO)
        logo.width = 2.9
        logo.to_corner(DR, buff=0.3).set_opacity(0.0)
        # Keep the watermark on the top layer so the full-frame map images (added
        # to the fixed-frame layer later) don't clip its upper edge.
        logo.set_z_index(10)
        self.add_fixed_in_frame_mobjects(logo)

        self.set_camera_orientation(phi=68 * DEGREES, theta=-90 * DEGREES,
                                    zoom=0.85)
        # A slow orbit through the opening so the real distances read as *depth*:
        # near stars swing across the view more than far ones (parallax). It
        # stops once they are on the sphere and only direction is left.
        self.begin_ambient_camera_rotation(rate=0.12)

        # Beat 1 — you are at the center; stars lie around us at real distances.
        self.play(FadeIn(eye, scale=0.5),
                  logo.animate.set_opacity(0.85), run_time=0.7)
        caption("You are at the center — stars lie all around, at every distance")
        self.play(LaggedStart(*[FadeIn(d) for d in star_mobs],
                              lag_ratio=0.03), run_time=1.7)
        self.wait(0.5)

        # Beat 2 — only direction survives: collapse onto the celestial sphere.
        caption("All we recover is direction — so we plot them on a sphere")
        self.play(*[m.animate.move_to(s["sphere"])
                    for m, s in zip(star_mobs, stars)],
                  Create(graticule), run_time=1.7)
        self.wait(0.2)
        self.stop_ambient_camera_rotation()

        # Beat 3 & 4 — settle to face the patch, hold up a flat pane, and let the
        # inward light mark where it crosses.
        caption("Their light reaches us — catch one patch on a flat pane")
        self.move_camera(phi=64 * DEGREES, theta=-90 * DEGREES, run_time=1.0,
                         added_anims=[FadeIn(pane)])
        caption("Where each ray crosses the pane is its place on the map")
        anims = [LaggedStart(Create(rays[i]), FadeIn(landed[i], scale=0.6),
                             lag_ratio=0.55) for i in order]
        self.play(LaggedStart(*anims, lag_ratio=0.05), run_time=3.2)
        self.wait(0.5)

        # Beat 5 — fly to the eye and look OUT through the pane; dissolve to the
        # REAL sph TAN chart (that outward view is exactly what the chart is).
        # The far stars, rays, and dome have done their job — fade them as we
        # fly in, leaving just the map on the pane. Aim at the pane and look OUT
        # through it (camera on the eye's side → the true outward orientation).
        self.move_camera(
            phi=168 * DEGREES, theta=-90 * DEGREES, zoom=2.3,
            frame_center=R_WIN * OUT, run_time=2.0,
            added_anims=[FadeOut(graticule), FadeOut(VGroup(*rays)),
                         FadeOut(VGroup(*star_mobs))])
        self.add_fixed_in_frame_mobjects(tan_png)
        tan_png.set_opacity(0.0)
        caption("The gnomonic (TAN) chart — the view from here, looking out",
                run_time=0.01)
        self.play(FadeOut(VGroup(eye, pane, *landed)),
                  tan_png.animate.set_opacity(1.0), run_time=1.7)
        self.wait(1.3)

        # Beat 6 (coda) — Orion is the anchor. Keep its stars + asterism fixed
        # (it is centered in every frame, so it never moves) while the frame
        # zooms out to a bare all-sky grid, then drape the sky image on.
        # 6a — the TAN chart shrinks into its place on the whole-sky grid.
        self.add_fixed_in_frame_mobjects(grid_png)
        grid_png.set_opacity(0.0)
        caption("Zoom out to the whole-sky grid — Orion is one small patch",
                run_time=0.01)
        self.play(tan_png.animate.scale(0.45).set_opacity(0.0),
                  grid_png.animate.set_opacity(1.0), run_time=1.8)
        self.wait(1.5)
        # 6b — drape the real sky image over the same grid; Orion stays put.
        self.add_fixed_in_frame_mobjects(raster_png)
        raster_png.set_opacity(0.0)
        caption("Drape the real sky image over the same grid", run_time=0.01)
        self.play(raster_png.animate.set_opacity(1.0), run_time=1.7)
        self.wait(1.6)
        # Fade back to bare navy so the loop is seamless (navy -> navy).
        self.play(FadeOut(raster_png), FadeOut(grid_png), FadeOut(self._cap),
                  FadeOut(logo), run_time=0.7)
        self.wait(0.3)
