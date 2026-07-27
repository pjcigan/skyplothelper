"""Manim explainer for tutorial #15 — *the perceived color of a star* (demo E).

A star marker morphs down the main sequence, O to M, beside its **blackbody
spectrum**. As the temperature falls the spectrum's peak slides from the blue end
toward the red. Three fixed **eye-response curves (R, G, B)** — the tristimulus
receptors — sit over the visible band; the star's color is what those three
stimuli add up to. Even for a cool star the blue/green receptors still catch
plenty of light, so the sum stays a *pale* orange rather than a deep red. The
three stimulus bars (lower left) show that balance shifting as the star cools.

The marker's color at every instant is a real ``sph.teff_to_rgb`` value (from
``assets/startype_stops.json``); the spectrum curve is textbook Planck physics
computed in-scene; the fill uses a luminance-even wavelength LUT from the JSON. A
coda dissolves into the notebook's real perceived-color all-sky chart
(``assets/startype_allsky.jpg``). Both assets come from
``make_startype_assets.py`` in the astropy env. A skyplothelper wordmark sits in
the corner for attribution if the file is reused. See ``README.md``.

Build (in the ManimCE env, from the repo root)::

    manim -qm --fps 30 --media_dir docs/manim/media docs/manim/startype.py StarType

then re-encode the mp4 and grab a poster (see ``README.md``). ManimCE 0.19.x, on
the shared navy sky canvas.
"""
from __future__ import annotations

import json
import os

import numpy as np
from manim import (
    DOWN,
    UP,
    Axes,
    Circle,
    FadeIn,
    FadeOut,
    ImageMobject,
    Line,
    ManimColor,
    Rectangle,
    Scene,
    Text,
    Triangle,
    ValueTracker,
    VGroup,
    VMobject,
    always_redraw,
    interpolate_color,
    rate_functions,
)

SKY = "#16203A"
INK = "#D7DEE8"
FAINT = "#4E6188"
R_COL, G_COL, B_COL = "#F2665A", "#5FC97A", "#5C9BF2"     # the three receptors
HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
LOGO = os.path.join(HERE, "..", "_static", "logo", "logo_6_wordmark-dark_mark.png")

WL_LO, WL_HI = 320.0, 1000.0          # plotted wavelength window (nm)
VIS_LO, VIS_HI = 380.0, 700.0         # the eye's visible band
HC_OVER_K = 0.0143877696             # h c / k_B  (m K)

# The three tristimulus receptors as clean response bells (illustrative L/M/S-ish
# peaks); enough to show "three stimuli, and their balance sets the color."
RGB_RESP = [(600.0, 44.0, R_COL), (545.0, 40.0, G_COL), (450.0, 34.0, B_COL)]


def planck_norm(wl_nm, T):
    """Blackbody spectral radiance vs wavelength, normalized to unit peak."""
    wl = np.asarray(wl_nm) * 1e-9
    B = (1.0 / wl**5) / (np.exp(HC_OVER_K / (wl * T)) - 1.0)
    return B / B.max()


def _bell(wl_nm, mu, sig):
    return np.exp(-0.5 * ((np.asarray(wl_nm) - mu) / sig) ** 2)


def _load():
    with open(os.path.join(ASSETS, "startype_stops.json")) as fh:
        return json.load(fh)


def _disp_radius(r_sun):
    return 0.22 + 1.05 * (r_sun / 9.0) ** 0.5


class StarType(Scene):
    def construct(self):
        self.camera.background_color = SKY
        d = _load()
        stops = d["stops"]
        n = len(stops)
        sun_i = d["sun_index"]
        t = ValueTracker(0.0)

        # wavelength -> luminance-even fill color (LUT from the asset generator).
        lut_wl = np.array([c[0] for c in d["wl_colors"]])
        lut_hex = [c[1] for c in d["wl_colors"]]

        def fill_color(w):
            if w < VIS_LO or w > VIS_HI:
                return FAINT
            return lut_hex[int(np.argmin(np.abs(lut_wl - w)))]

        def _interp(x, key):
            lo = max(0, min(n - 1, int(x)))
            hi = min(n - 1, lo + 1)
            f = x - lo
            return (1 - f) * stops[lo][key] + f * stops[hi][key], lo, hi, f

        def star_color(x):
            _, lo, hi, f = _interp(x, "teff")
            return interpolate_color(ManimColor(stops[lo]["color"]),
                                     ManimColor(stops[hi]["color"]), f)

        def temp_at(x):
            return _interp(x, "teff")[0]

        def radius_at(x):
            r, lo, hi, f = _interp(x, "r_sun")
            return _disp_radius((1 - f) * stops[lo]["r_sun"] + f * stops[hi]["r_sun"])

        STAR_POS = [-4.8, 1.55, 0]

        # --- the morphing star: a clean solid disk in the star's own color ----
        star = always_redraw(lambda: Circle(
            radius=radius_at(t.get_value()), stroke_width=0,
            fill_color=star_color(t.get_value()), fill_opacity=1.0).move_to(STAR_POS))

        # --- readout: class · temperature · perceived hue --------------------
        def make_label():
            s = stops[max(0, min(n - 1, int(round(t.get_value()))))]
            return Text(f"Class {s['cls']}    {s['teff']:,} K    {s['cue']}",
                        font="sans-serif", color=INK).scale(0.5).to_edge(UP, buff=0.5)

        label = always_redraw(make_label)

        # --- the blackbody spectrum panel ------------------------------------
        axes = Axes(x_range=[WL_LO, WL_HI, 200], y_range=[0, 1.08, 1],
                    x_length=7.4, y_length=3.6, tips=False,
                    axis_config={"color": FAINT, "stroke_width": 2},
                    y_axis_config={"include_ticks": False}).move_to([2.5, 0.5, 0])
        wl_grid = np.linspace(WL_LO, WL_HI, 200)
        bar_i = np.arange(0, len(wl_grid) - 1, 3)          # ~66 colored bars

        def spectrum_body():
            T = temp_at(t.get_value())
            B = planck_norm(wl_grid, T)
            bars = VGroup()
            for i in bar_i:
                w0 = wl_grid[i]
                w1 = wl_grid[i + 3] if i + 3 < len(wl_grid) else wl_grid[-1]
                b = B[i]
                p_lo = axes.c2p(w0, 0)
                p_hi = axes.c2p(w1, b)
                inside = VIS_LO <= w0 <= VIS_HI
                rect = Rectangle(width=abs(p_hi[0] - p_lo[0]),
                                 height=max(abs(p_hi[1] - p_lo[1]), 1e-3),
                                 stroke_width=0, fill_color=fill_color(w0),
                                 fill_opacity=0.92 if inside else 0.12)
                rect.move_to([(p_lo[0] + p_hi[0]) / 2, (p_lo[1] + p_hi[1]) / 2, 0])
                bars.add(rect)
            curve = VMobject().set_points_as_corners(
                [axes.c2p(w, b) for w, b in zip(wl_grid, B)]).set_stroke(INK, 2.0)
            return VGroup(bars, curve)

        spectrum = always_redraw(spectrum_body)

        # The three receptor curves are fixed (they don't move with T) and drawn
        # IN FRONT of the fill; the whole point is that they stay put while the
        # spectrum slides beneath them.
        resp_curves = VGroup()
        for mu, sig, col in RGB_RESP:
            c = VMobject().set_points_as_corners(
                [axes.c2p(w, 0.84 * _bell(w, mu, sig)) for w in wl_grid])
            c.set_stroke(col, 2.6, opacity=0.95).set_fill(opacity=0.0)
            resp_curves.add(c)
        resp_lbl = VGroup(
            Text("the eye's three receptors:", font="sans-serif", color=INK).scale(0.34),
            Text("R", font="sans-serif", color=R_COL).scale(0.34),
            Text("G", font="sans-serif", color=G_COL).scale(0.34),
            Text("B", font="sans-serif", color=B_COL).scale(0.34),
        ).arrange(buff=0.16).next_to(axes, UP, buff=0.12)
        xlbl = Text("wavelength (nm)", font="sans-serif", color=INK).scale(0.36) \
            .next_to(axes, DOWN, buff=0.2)
        xticks = VGroup(*[
            Text(str(w), font="sans-serif", color=FAINT).scale(0.3)
            .next_to(axes.c2p(w, 0), DOWN, buff=0.12)
            for w in (400, 600, 800)])
        panel = VGroup(axes, resp_curves, resp_lbl, xlbl, xticks)

        # --- stimulus-balance bars (how much each receptor catches) ----------
        vis_mask = (wl_grid >= VIS_LO) & (wl_grid <= VIS_HI)
        vwl = wl_grid[vis_mask]

        def stimuli(T):
            B = planck_norm(wl_grid, T)[vis_mask]
            s = np.array([getattr(np, 'trapezoid', np.trapz)(B * _bell(vwl, mu, sig), vwl)
                          for mu, sig, _ in RGB_RESP])
            return s / s.max()                     # relative balance (color = ratios)

        BAR_X = [-5.5, -5.05, -4.6]
        BAR_BASE, BAR_H, BAR_W = -1.75, 1.35, 0.3

        def make_bars():
            s = stimuli(temp_at(t.get_value()))
            g = VGroup()
            for x, val, (_, _, col) in zip(BAR_X, s, RGB_RESP):
                h = max(0.02, val * BAR_H)
                r = Rectangle(width=BAR_W, height=h, stroke_width=0,
                              fill_color=col, fill_opacity=0.95)
                r.move_to([x, BAR_BASE + h / 2, 0])
                g.add(r)
            return g

        bars = always_redraw(make_bars)
        bars_axis = Line([BAR_X[0] - 0.28, BAR_BASE, 0], [BAR_X[-1] + 0.28, BAR_BASE, 0],
                         stroke_color=FAINT, stroke_width=2)
        bars_lbl = Text("their balance = the star's color", font="sans-serif",
                        color=INK).scale(0.32).next_to(bars_axis, DOWN, buff=0.14)

        # --- the O..M spectral rail with a sliding pointer -------------------
        X0, X1, Y = -5.7, 5.7, -3.25
        rail = Line([X0, Y, 0], [X1, Y, 0], stroke_color=FAINT, stroke_width=2)
        first_idx = {}
        for i, s in enumerate(stops):
            first_idx.setdefault(s["cls"], i)
        letters = VGroup()
        for cls in d["classes"]:
            x = X0 + (X1 - X0) * (first_idx.get(cls, 0) / (n - 1))
            letters.add(Text(cls, font="sans-serif", color=INK).scale(0.5)
                        .move_to([x, Y - 0.4, 0]))
        pointer = always_redraw(lambda: Triangle(
            color=star_color(t.get_value()), fill_color=star_color(t.get_value()),
            fill_opacity=1.0).scale(0.12).rotate(np.pi).move_to(
                [X0 + (X1 - X0) * (t.get_value() / (n - 1)), Y + 0.26, 0]))

        # --- skyplothelper wordmark, corner watermark (persists incl. coda) --
        logo = ImageMobject(LOGO)
        logo.width = 2.5
        logo.to_corner([1, 1, 0], buff=0.28).set_opacity(0.6)   # top-right

        # ---- beats ----------------------------------------------------------
        self.add(logo)
        self.play(FadeIn(star), FadeIn(label), FadeIn(panel), FadeIn(spectrum),
                  FadeIn(bars), FadeIn(bars_axis), FadeIn(bars_lbl),
                  FadeIn(rail), FadeIn(letters), FadeIn(pointer), run_time=1.0)
        self.wait(0.4)
        cap = Text("a star glows as a blackbody — the eye sums its light into one color",
                   font="sans-serif", color=INK).scale(0.42).move_to([0.4, -2.35, 0])
        self.play(FadeIn(cap), run_time=0.6)
        self.wait(1.3)
        self.play(FadeOut(cap), run_time=0.5)

        # O -> G (the Sun): the peak slides in from the blue end toward green.
        self.play(t.animate.set_value(sun_i), run_time=5.0,
                  rate_func=rate_functions.ease_in_out_sine)
        cap = Text("our Sun — the spectrum sits roughly level across the visible",
                   font="sans-serif", color=INK).scale(0.42).move_to([0.4, -2.35, 0])
        self.play(FadeIn(cap), run_time=0.6)
        self.wait(1.5)
        self.play(FadeOut(cap), run_time=0.5)

        # G -> M: peak slides into the red, but the eye still catches blue-green.
        self.play(t.animate.set_value(n - 1), run_time=3.8,
                  rate_func=rate_functions.ease_in_out_sine)
        cap = Text("peak in the red — but the blue-green receptors still fire: pale orange",
                   font="sans-serif", color=INK).scale(0.42).move_to([0.4, -2.35, 0])
        self.play(FadeIn(cap), run_time=0.6)
        self.wait(1.8)

        # ---- coda: the same function over the whole real sky ----------------
        for m in (star, label, spectrum, bars, pointer):
            m.clear_updaters()
        allsky = ImageMobject(os.path.join(ASSETS, "startype_allsky.jpg"))
        allsky.width = 12.6
        coda = Text("every naked-eye star, colored exactly this way",
                    font="sans-serif", color=INK).scale(0.5).to_edge(UP, buff=0.4)
        self.add(allsky)
        allsky.set_opacity(0.0)
        self.bring_to_front(logo)                     # keep the watermark on top
        self.play(FadeOut(VGroup(star, label, spectrum, panel, bars, bars_axis,
                                 bars_lbl, pointer, rail, letters, cap)),
                  allsky.animate.set_opacity(1.0), FadeIn(coda), run_time=1.6)
        self.wait(2.2)

        # Fade back to bare navy so the loop is seamless (navy -> navy).
        self.play(FadeOut(allsky), FadeOut(coda), run_time=0.7)
        self.wait(0.3)
