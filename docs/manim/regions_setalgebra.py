"""Manim explainer for tutorial #7 — *set algebra as an animated Venn* (demo B).

Teaches ``CompoundRegion`` set algebra the way a textbook Venn diagram does — two
overlapping sets A and B — but built for comparison and construction:

1. Each operation (union, intersection, difference) fills in the center, then
   shrinks into a corner and persists, one color each.
2. The three corners then fly onto **one real sph globe**, each landing at its own
   sky position — off-center pairs visibly stretched by the projection toward the
   limb.
3. A coda assembles the notebook's worked survey footprint on the all-sky frame,
   one operation at a time: the box drops in, then the galactic band and the
   exclusion hole flash in their own colors and carve it, piece by piece.

This is the manim side only. It draws *no* science map: the globe and every
build-up frame is a genuine sph render imported as an ``ImageMobject``
(``assets/trio_globe.png``, ``assets/build_*.png``, ``assets/{band,hole}_zone.png``),
all produced by ``make_regions_assets.py`` in the astropy env. The abstract Venn
shapes are manim's own ``Union``/``Intersection``/``Difference`` of two circles —
pure typography, not a sky map. See ``README.md``.

Build (in the optional ManimCE env, from the repo root)::

    manim -qm --fps 30 --media_dir docs/manim/media docs/manim/regions_setalgebra.py SetAlgebra

then re-encode the mp4 for the web and grab a poster still (see ``README.md`` for
the exact ffmpeg lines and the `<video>` embed). ManimCE 0.19.x.
"""
from __future__ import annotations

import os

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Circle,
    Difference,
    FadeIn,
    FadeOut,
    ImageMobject,
    Intersection,
    Scene,
    Text,
    Union,
    VGroup,
)

SKY = "#16203A"            # shared navy canvas (matches the sph PNGs)
INK = "#D7DEE8"            # light label text / set outlines on navy
GREEN = "#5E8C7E"          # union   ┐ the uranometria tones the notebook's
GOLD = "#C29B3C"           # inter   │ §4 gallery uses for these same ops
RUST = "#8A4540"           # diff    ┘

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

R = 1.65                   # Venn circle radius (center stage)
DX = 0.95                  # half the center separation (circles overlap)


class SetAlgebra(Scene):
    def construct(self):
        self.camera.background_color = SKY

        # Persistent skyplothelper wordmark, bottom-right — branding for anyone who
        # downloads/reuses the clip. The mark is itself a minimal all-sky frame, so it
        # sits in-theme on the navy. add_foreground_mobject keeps it above every PNG
        # that fades in later; it never fades, so the navy loop point carries it too.
        logo = ImageMobject(os.path.join(HERE, "..", "_static", "logo",
                                         "logo_6_wordmark-dark_mark.png"))
        logo.width = 3.0
        logo.to_corner(RIGHT + DOWN, buff=0.28).set_opacity(0.9)
        self.add_foreground_mobject(logo)

        # Central sets. A on the LEFT, B on the RIGHT — the same layout as the sph
        # globe (where +lon falls left on an east-left frame).
        A = Circle(radius=R).set_stroke(INK, 3).move_to(LEFT * DX)
        B = Circle(radius=R).set_stroke(INK, 3).move_to(RIGHT * DX)
        labA = Text("A", font="sans-serif", color=INK).scale(0.6).next_to(
            A, LEFT, buff=0.12).shift(UP * 1.0)
        labB = Text("B", font="sans-serif", color=INK).scale(0.6).next_to(
            B, RIGHT, buff=0.12).shift(UP * 1.0)
        core = VGroup(A, B, labA, labB)

        self._cap = None

        def caption(txt, run_time=0.6):
            old = self._cap
            if old is not None:
                self.play(FadeOut(old), run_time=run_time * 0.4)
                self.remove(old)
            new = Text(txt, font="sans-serif", color=INK).scale(0.55)
            new.to_edge(UP, buff=0.32).set_opacity(0.0)
            self.play(new.animate.set_opacity(1.0), run_time=run_time * 0.6)
            self._cap = new

        def img(name, **kw):
            im = ImageMobject(os.path.join(ASSETS, name))
            for k, v in kw.items():
                setattr(im, k, v)
            return im

        # ---- intro ----------------------------------------------------------
        self.play(FadeIn(core), run_time=1.0)
        caption("Two regions on the sky — sets A and B")
        self.wait(0.6)

        # ---- one beat per operation: fill in the center, then park in a corner.
        # corner = where it will later land on the globe (union top-left, etc.).
        corner = {"green": LEFT * 4.7 + UP * 2.3,     # union     → globe upper-left
                  "gold": RIGHT * 4.7 + UP * 2.3,     # intersect → globe upper-right
                  "rust": DOWN * 2.85}                # diff      → globe bottom
        glyph = {"green": "∪", "gold": "∩", "rust": "−"}
        ops = [
            ("A  ∪  B   —   union", Union(A.copy(), B.copy()), GREEN, "green"),
            ("A  ∩  B   —   intersection", Intersection(A.copy(), B.copy()), GOLD, "gold"),
            ("A  −  B   —   difference", Difference(A.copy(), B.copy()), RUST, "rust"),
        ]
        parked = []
        for label, shape, color, key in ops:
            shape.set_stroke(color, 3).set_fill(color, 0.62)
            caption(label)
            self.play(FadeIn(shape), run_time=0.7)
            self.wait(0.6)
            # a self-contained mini-Venn (outlines + fill + op glyph) for the corner
            mini = VGroup(
                Circle(radius=R).set_stroke(INK, 2).move_to(LEFT * DX),
                Circle(radius=R).set_stroke(INK, 2).move_to(RIGHT * DX),
                shape.copy(),
                Text(glyph[key], font="sans-serif", color=color).scale(0.8).move_to(
                    DOWN * (R + 0.55)),
            ).scale(0.34).move_to(corner[key])
            self.play(FadeOut(shape), FadeIn(mini), run_time=0.8)
            parked.append((mini, key))
        self.wait(0.4)

        # ---- gather the three corners onto ONE real sph globe ---------------
        caption("The same three, as real regions on a sphere")
        globe_pos = {"green": LEFT * 1.7 + UP * 1.7,
                     "gold": RIGHT * 1.7 + UP * 1.7,
                     "rust": DOWN * 1.9}
        self.play(FadeOut(core),
                  *[m.animate.move_to(globe_pos[k]).scale(1.7) for m, k in parked],
                  run_time=1.2)
        trio = img("trio_globe.png", height=6.7)
        self.play(FadeOut(VGroup(*[m for m, _ in parked])), FadeIn(trio),
                  run_time=1.3)
        self.wait(1.6)

        # ---- coda: build the worked footprint one operation at a time -------
        frame = img("build_frame.png", width=12.2)
        self.play(FadeOut(trio), FadeIn(frame), run_time=1.1)
        caption("Compose them — a survey footprint, built up")

        box = img("build_box.png", width=12.2)
        self.play(FadeIn(box), run_time=0.9)          # over the empty frame
        self.wait(0.7)

        # Each cut is two cross-fades on same-geometry frames: the current footprint
        # → the footprint with the cut zone laid over it (its colored contribution)
        # → the carved result. All opaque, so they register cleanly.
        def carve(zone_name, result_name, cap_txt):
            zone = img(zone_name, width=12.2)
            result = img(result_name, width=12.2)
            caption(cap_txt)
            self.play(FadeOut(self._prev), FadeIn(zone), run_time=0.7)   # contribution
            self.wait(0.6)
            self.play(FadeOut(zone), FadeIn(result), run_time=0.9)       # carve
            self._prev = result

        self._prev = box
        carve("build_box_bandzone.png", "build_box_band.png", "minus the galactic plane")
        carve("build_bmb_holezone.png", "build_final.png", "minus a bright-source hole")
        self.wait(1.7)

        # back to bare navy so the loop is seamless
        self.play(FadeOut(self._prev), FadeOut(frame), FadeOut(self._cap),
                  run_time=0.8)
        self.wait(0.3)
