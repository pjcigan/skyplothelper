"""manim explainers for tutorial #11 — the two globe-motion demos.

* ``EulerTrio``  (demo C) — the three physical angles that pose every globe made
  tangible: a wireframe body swept in turn by ``rotation`` (spin about the pole),
  ``obliquity`` (tilt the pole toward the viewer) and ``perspective`` (precession
  of the tilted pole), then a dissolve into the *real* sph Blue-Marble globe posed
  at exactly those angles, which then re-runs the three motions draped.
* ``Precession`` (demo D) — the three axial motions of a spinning body shown one at
  a time (spin, nutation, then precession with the pole rod sweeping a traced cone),
  then all three at once for one cycle, dissolving into the real Black-Marble
  look-down globe with its spin axis, which repeats that combined motion draped.
  Its nutation amplitude and precession rate are heavily exaggerated to read — the
  notebook says so above the video.

**How the wireframe stays honest.** The schematic globe is never rotated by raw
3-D angles; every frame it is *posed from sph parameters* — ``euler_to_fits_ortho``
(reimplemented in pure numpy, verified identical to skyplothelper) gives
``(center_lon, center_lat, lonpole)``, and ``orientation_from_sph`` turns that into
the body orientation that reproduces ``make_planet_frame``'s view. So the wireframe
and the real sph cut match *by construction* for any pose — including the precession
beat, where the re-aiming frame makes the tilted pole sweep a circle. The genuine
sph renders it dissolves into come from ``make_globe_motions_assets.py`` (astropy
env). manim is only the camera, typographer, and brand mark.

Build (in the optional ManimCE env, from the repo root)::

    manim -qm --fps 30 --media_dir docs/manim/media docs/manim/globe_motions.py EulerTrio
    manim -qm --fps 30 --media_dir docs/manim/media docs/manim/globe_motions.py Precession

then re-encode each mp4 for the web and grab a poster still (see ``README.md``).
ManimCE 0.19.x, on the shared navy sky canvas so the sph renders blend in.
"""
from __future__ import annotations

import os
from glob import glob

import numpy as np
from manim import (
    DEGREES,
    ORIGIN,
    UP,
    DashedVMobject,
    Dot3D,
    FadeIn,
    FadeOut,
    ImageMobject,
    Line3D,
    Sphere,
    Text,
    ThreeDScene,
    TracedPath,
    ValueTracker,
    VGroup,
    VMobject,
    smooth,
)

SKY = "#16203A"
BODY = "#1B2949"           # globe fill: a touch above the navy so the disk reads
GRID = "#5A6E97"           # faint graticule
EQ_C = "#C29B3C"           # gold equator  (matches the sph cut)
PM_C = "#5E8C7E"           # green prime meridian
POLE_C = "#E8DDB5"         # wheat spin axis / pole rod / landmark
LABEL = "#D7DEE8"
HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
LOGO = os.path.join(HERE, "..", "_static", "logo", "logo_6_wordmark-dark_mark.png")

R = 2.2                    # globe radius in scene units


# ---------------------------------------------------------------------------
# Posing the wireframe from sph parameters (so it matches make_planet_frame).
# ---------------------------------------------------------------------------
def _Rz(a):
    a = np.radians(a)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])


def _Rx(a):
    a = np.radians(a)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def euler_to_fits_ortho(rotation, obliquity, perspective):
    """Pure-numpy twin of ``skyplothelper.euler_to_fits_ortho`` (astropy PASSIVE
    matrices, same extraction) — verified identical to 0.0°. Returns
    ``(center_lon, center_lat, lonpole)`` in degrees."""
    m = _Rz(rotation) @ _Rx(obliquity) @ _Rz(perspective)
    clon = np.degrees(np.arctan2(-m[0, 1], m[1, 1]))
    clat = -np.degrees(np.arcsin(np.clip(m[2, 1], -1.0, 1.0)))
    lonpole = -np.degrees(np.arctan2(-m[2, 0], m[2, 2]))
    return clon, clat, lonpole


def _unit(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def orientation_from_sph(clon, clat, lonpole):
    """Body orientation (3×3) that makes the wireframe reproduce
    ``make_planet_frame(center_LONdeg=clon, center_LATdeg=clat, lonpole=lonpole)``.

    The manim camera views +X (toward us = sub-observer), +Y screen-right (east),
    +Z up (north). So we send the sub-observer's local (east, north, out) frame onto
    (Y, Z, X) and roll about the view axis by ``lonpole``. Calibrated against sph
    renders."""
    lo, la = np.radians(clon), np.radians(clat)
    s = _unit(clon, clat)                                    # sub-observer -> +X
    e = np.array([-np.sin(lo), np.cos(lo), 0.0])             # east        -> +Y
    n = np.array([-np.sin(la) * np.cos(lo),
                  -np.sin(la) * np.sin(lo), np.cos(la)])     # north       -> +Z
    r0 = np.column_stack([s, e, n]).T
    roll = np.radians(lonpole)
    c, si = np.cos(roll), np.sin(roll)
    rx = np.array([[1, 0, 0], [0, c, -si], [0, si, c]])      # roll about the view axis
    return rx @ r0


def _xyz(lon_deg, lat_deg, radius=R):
    return radius * _unit(lon_deg, lat_deg)


# ---------------------------------------------------------------------------
# The wireframe, rebuilt each frame at a given orientation so its curves can be
# split near/far. (manim's default renderer sorts whole mobjects by depth rather
# than per-pixel, so the near/far split has to be explicit — which is also exactly
# how sph's highlight_great_circle draws a ring: near half solid, far half dashed.)
# ---------------------------------------------------------------------------
def _wpts(lonlats, m, radius):
    """World points for body (lon, lat) samples at orientation ``m``."""
    return np.array([m @ _xyz(lo, la, radius) for lo, la in lonlats])


def camera_dir(phi_deg, theta_deg):
    """The unit vector from the origin toward a manim ThreeDScene camera at
    ``(phi, theta)`` — i.e. which side of the body faces the viewer."""
    p, t = np.radians(phi_deg), np.radians(theta_deg)
    return np.array([np.sin(p) * np.cos(t), np.sin(p) * np.sin(t), np.cos(p)])


VIEW_EULER = camera_dir(90, 0)          # EulerTrio's sph-matching camera (+X)


def _runs(pts, closed, view=VIEW_EULER):
    """Split a sampled curve into contiguous near (facing ``view``) and far runs;
    returns [(points, is_near), ...]."""
    near = pts @ np.asarray(view) >= 0
    out, start = [], 0
    for i in range(1, len(pts)):
        if near[i] != near[start]:
            out.append((pts[start:i + 1], bool(near[start])))   # share the crossing
            start = i
    out.append((pts[start:], bool(near[start])))
    if closed and len(out) > 1 and out[0][1] == out[-1][1]:     # merge the wrap seam
        out[0] = (np.vstack([out[-1][0], out[0][0]]), out[0][1])
        out.pop()
    return out


def _curve(pts, color, width, opacity=1.0, dashed=False):
    if len(pts) < 2:
        return VGroup()
    vm = VMobject().set_points_as_corners([np.asarray(p) for p in pts])
    vm.set_stroke(color, width, opacity=opacity)
    return DashedVMobject(vm, num_dashes=16, dashed_ratio=0.5) if dashed else vm


def _rotA(axis, deg):
    """Active (right-hand) rotation matrix about ``axis`` — Rodrigues."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    x, y, z = a
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


def spin_nut_prec(spin, tilt, prec):
    """Body orientation for the three axial motions, composed the way they act
    physically: spin turns the body about its own pole, ``tilt`` (obliquity plus any
    nutation nod) leans that pole, and ``prec`` carries the leaned pole around the
    vertical — so the pole traces a cone under precession, nods under nutation, and
    stays put under spin. Same decomposition ``make_globe_angles`` parametrizes."""
    return _rotA([0, 0, 1], prec) @ _rotA([1, 0, 0], tilt) @ _rotA([0, 0, 1], spin)


def make_sphere():
    """A flat, faint body tint behind the wireframe — added once, never posed.

    Shading is turned OFF on purpose. A *lit, opaque* sphere carries a fixed bright
    hemisphere and hard-occludes the far-side graticule, so as the wire turns only
    the near-side lines sweep across a motionless disk — the body reads as a
    stationary surface with a spinning cage over it. Flattened, it becomes a
    featureless backdrop and lets the wireframe show its far side (faint) — the same
    near/far split the Precession scene relies on — so the rotation lives in the
    wireframe and the globe reads as turning as one piece."""
    body = Sphere(radius=R, resolution=(32, 32))
    body.set_fill(BODY, opacity=1.0)
    body.set_stroke(width=0)
    body.set_shade_in_3d(False)     # no fixed lit hemisphere, no hard far-side
    return body                     # occlusion: the turn lives in the wireframe


def posed_wire(m, view=VIEW_EULER):
    """The graticule + gold equator + green prime meridian + landmark at body
    orientation ``m``, split near/far against the camera direction ``view``. The far side of every curve is drawn faint (graticule) or
    dashed (the two highlights), so the sphere reads solid and depth is unambiguous
    — matching how the sph cut draws them."""
    g = VGroup()
    for lon in range(0, 360, 30):
        pts = _wpts([(lon, la) for la in np.linspace(-88, 88, 40)], m, R * 1.004)
        for run, near in _runs(pts, closed=False, view=view):
            g.add(_curve(run, GRID, 1.2, opacity=0.55 if near else 0.18))
    for lat in range(-60, 61, 30):
        if lat == 0:
            continue
        pts = _wpts([(lo, lat) for lo in np.linspace(0, 360, 90)], m, R * 1.004)
        for run, near in _runs(pts, closed=True, view=view):
            g.add(_curve(run, GRID, 1.2, opacity=0.55 if near else 0.18))
    eq = _wpts([(lo, 0.0) for lo in np.linspace(0, 360, 121)], m, R * 1.006)
    for run, near in _runs(eq, closed=True, view=view):
        g.add(_curve(run, EQ_C, 3.4, opacity=1.0 if near else 0.75, dashed=not near))
    pm = _wpts([(0.0, la) for la in np.linspace(-89, 89, 90)], m, R * 1.006)
    for run, near in _runs(pm, closed=False, view=view):
        g.add(_curve(run, PM_C, 3.4, opacity=1.0 if near else 0.75, dashed=not near))
    lm = m @ _xyz(0, 0, R * 1.01)
    if lm @ np.asarray(view) >= 0:       # the landmark hides when it swings behind
        g.add(Dot3D(lm, radius=0.06, color=POLE_C))
    return g


class _GlobeScene(ThreeDScene):
    """Shared setup: navy canvas, the sph-matching camera, captions, brand mark."""

    def setup_scene(self):
        self.camera.background_color = SKY
        # Camera on +X looking at the origin with +Z up — the sph identity globe
        # view (sub-observer 0,0, north up). orientation_from_sph then carries the
        # body to any make_planet_frame pose.
        self.set_camera_orientation(phi=90 * DEGREES, theta=0 * DEGREES, zoom=1.05)
        self._cap = None
        self._R = np.eye(3)                 # the globe's current orientation
        # Brand mark, pinned bottom-right in screen space, on every frame.
        logo = ImageMobject(LOGO)
        logo.width = 2.5
        logo.to_corner(np.array([1, -1, 0]), buff=0.28).set_opacity(0.85)
        self.add_fixed_in_frame_mobjects(logo)
        self._logo = logo

    def caption(self, txt, run_time=0.7):
        old = self._cap
        if old is not None:
            self.play(FadeOut(old), run_time=run_time * 0.4)
            self.remove(old)
        new = Text(txt, font="sans-serif", color=LABEL).scale(0.5)
        new.to_edge(UP, buff=0.35).set_opacity(0.0)
        self.add_fixed_in_frame_mobjects(new)
        self.play(new.animate.set_opacity(1.0), run_time=run_time * 0.6)
        self._cap = new

    def pose(self, wire, euler_fn, t0, t1, run_time, rate_func=smooth):
        """Animate ``wire`` through the sph poses ``euler_fn(t)`` for t: t0→t1,
        rebuilding it each frame so it tracks exactly the views make_planet_frame
        would render (and so the near/far split follows the motion)."""
        tr = ValueTracker(float(t0))

        def upd(w):
            w.become(posed_wire(orientation_from_sph(*euler_fn(tr.get_value()))))

        wire.add_updater(upd)
        self.play(tr.animate.set_value(float(t1)), run_time=run_time,
                  rate_func=rate_func)
        wire.remove_updater(upd)
        self._R = orientation_from_sph(*euler_fn(t1))

    def dissolve_to(self, png_name, run_time=1.8, height=5.9):
        img = ImageMobject(os.path.join(ASSETS, png_name))
        img.height = height
        self.add_fixed_in_frame_mobjects(img)
        img.set_opacity(0.0)
        self.play(img.animate.set_opacity(1.0), run_time=run_time)
        # Keep the caption and brand mark above the (smaller) cut image.
        self.add_fixed_in_frame_mobjects(self._cap, self._logo)
        return img

    def flipbook(self, holder, prefix, run_time):
        """Flip ``holder`` through the real sph recap frames ``<prefix>_NN.jpg`` so
        the *draped* globe is seen performing the motions. Returns the last frame's
        mobject (each frame is its own ImageMobject)."""
        frames = sorted(glob(os.path.join(ASSETS, f"{prefix}_*.jpg")))
        if not frames:
            return holder
        dt = run_time / len(frames)
        cur = holder
        for path in frames:
            nxt = ImageMobject(path)
            nxt.height = cur.height
            nxt.move_to(cur)
            self.add_fixed_in_frame_mobjects(nxt)
            self.add_fixed_in_frame_mobjects(self._cap, self._logo)
            self.remove(cur)
            cur = nxt
            self.wait(dt)
        return cur


class EulerTrio(_GlobeScene):
    # rotation, obliquity, perspective the beats sweep to; the cut PNG is posed at
    # exactly these in make_globe_motions_assets.py (EULER_TRIPLE).
    ROT, OBL, PER = 45.0, 30.0, 60.0

    def construct(self):
        self.setup_scene()
        body = make_sphere()
        wire = posed_wire(np.eye(3))
        self.play(FadeIn(body, scale=0.6), FadeIn(wire, scale=0.6), run_time=0.9)
        self.caption("Three angles pose any globe — start it upright")
        self.wait(0.4)

        # 1 — rotation: spin about the pole (the sub-observer longitude turns).
        self.caption("rotation — spin about the pole")
        self.pose(wire, lambda t: euler_to_fits_ortho(t, 0.0, 0.0), 0, self.ROT, 1.6)
        self.wait(0.3)

        # 2 — obliquity: tilt the pole toward the viewer.
        self.caption("obliquity — tilt the pole toward you")
        self.pose(wire, lambda t: euler_to_fits_ortho(self.ROT, t, 0.0), 0, self.OBL, 1.6)
        self.wait(0.3)

        # 3 — perspective: precess the tilted pole (the re-aiming frame sweeps the
        # pole around a circle, not a spin about it).
        self.caption("perspective — precess the tilted pole")
        self.pose(wire, lambda t: euler_to_fits_ortho(self.ROT, self.OBL, t),
                  0, self.PER, 1.9)
        self.wait(0.5)

        # Cut to the real sph globe posed at exactly (rotation, obliquity, perspective).
        self.caption("the real skyplothelper globe, posed at those three angles")
        img = self.dissolve_to("globe_euler_posed.png")
        self.play(FadeOut(body), FadeOut(wire), run_time=0.6)
        self.wait(1.2)

        # And the same three motions run on the draped globe itself — a strip of real
        # sph frames, so the connection to your own code is unmistakable.
        self.caption("…and the same three motions, on the draped globe")
        img = self.flipbook(img, "euler_recap", run_time=3.4)
        self.wait(0.8)
        self.play(FadeOut(img), FadeOut(self._cap), run_time=0.7)
        self.wait(0.3)


VIEW_PREC = camera_dir(54, -72)     # the Precession scene's oblique look-down camera


def _rod(m, length, view=VIEW_PREC):
    """The spin axis at orientation ``m``: the half that points toward the viewer is
    solid, the half running away behind the body is faint — so the rod reads as
    skewering the globe rather than floating over it."""
    plus, minus = m @ _xyz(0, 90, length), m @ _xyz(0, -90, length)
    near_plus = float(plus @ np.asarray(view)) >= 0
    g = VGroup()
    for end, near in ((plus, near_plus), (minus, not near_plus)):
        seg = Line3D(ORIGIN, end, color=POLE_C, thickness=0.05)
        seg.set_opacity(1.0 if near else 0.3)
        g.add(seg)
    # Keep the rod in the same flat 2-D layer as the unshaded body sphere and the
    # wireframe. A 3-D Line3D renders *beneath* the flat (shade_in_3d=False) sphere,
    # so when the axis stands upright over the disk its near half is clipped away —
    # visible whenever a beat pauses. Depth here is the near/far opacity split above,
    # exactly as the wireframe does it, not renderer z-sorting.
    g.set_shade_in_3d(False)
    return g


class Precession(_GlobeScene):
    ROD = R * 1.7            # spin-axis half-length (well clear of the sphere)
    OBL = 26.0               # the body's axial tilt for the demo
    NUT = 7.0                # nutation amplitude — exaggerated to read (see caption)

    def _body_at(self, m):
        return VGroup(posed_wire(m, view=VIEW_PREC), _rod(m, self.ROD))

    def motion(self, holder, tip, fn, t0, t1, run_time, rate_func=smooth):
        """Drive the body through orientations ``fn(t)``, rebuilding the wireframe
        and rod each frame (so the near/far split tracks the motion) and carrying the
        axis tip along for the traced precession circle."""
        tr = ValueTracker(float(t0))

        def upd_body(h):
            h.become(self._body_at(fn(tr.get_value())))

        def upd_tip(d):
            d.move_to(fn(tr.get_value()) @ _xyz(0, 90, self.ROD))

        holder.add_updater(upd_body)
        tip.add_updater(upd_tip)
        self.play(tr.animate.set_value(float(t1)), run_time=run_time,
                  rate_func=rate_func)
        holder.remove_updater(upd_body)
        tip.remove_updater(upd_tip)

    def construct(self):
        self.setup_scene()
        # An oblique look-down so the precession traces a visible circle, not a
        # foreshortened pendulum (the lesson #17 §4's equator-on view can't show).
        self.set_camera_orientation(phi=54 * DEGREES, theta=-72 * DEGREES, zoom=0.95)
        body = make_sphere()
        holder = self._body_at(spin_nut_prec(0, self.OBL, 0))
        tip = Dot3D(spin_nut_prec(0, self.OBL, 0) @ _xyz(0, 90, self.ROD),
                    radius=0.10, color=POLE_C)
        tip.set_shade_in_3d(False)      # same flat layer as the rod (see _rod)
        self.play(FadeIn(body, scale=0.6), FadeIn(holder, scale=0.6), FadeIn(tip),
                  run_time=1.1)
        self.caption("A spinning body has three axial motions")
        self.wait(0.4)

        # Spin — the body turns about its own axis; the axis itself stays put.
        self.caption("spin — the body turns about its axis")
        self.motion(holder, tip, lambda t: spin_nut_prec(t, self.OBL, 0),
                    0, 360, 2.6, rate_func=lambda x: x)
        self.wait(0.3)

        # Nutation — a small nod of the axis (amplitude exaggerated to read).
        self.caption("nutation — the axis nods")
        self.motion(holder, tip,
                    lambda t: spin_nut_prec(0, self.OBL + self.NUT * np.sin(np.radians(t)), 0),
                    0, 360, 2.2, rate_func=lambda x: x)
        self.wait(0.3)

        # Precession — the axis sweeps a cone; a traced path draws the tip's circle.
        self.caption("precession — the axis sweeps a cone")
        circle = TracedPath(tip.get_center, stroke_color="#F2E4B8", stroke_width=3)
        self.add(circle)
        self.motion(holder, tip, lambda t: spin_nut_prec(0, self.OBL, t),
                    0, 360, 3.6, rate_func=lambda x: x)
        self.wait(0.4)
        self.remove(circle)

        # All three at once — one cycle of the real thing.
        self.caption("all three together — one cycle")
        self.motion(
            holder, tip,
            lambda t: spin_nut_prec(3 * t, self.OBL + self.NUT * np.sin(np.radians(t)), t),
            0, 360, 4.2, rate_func=lambda x: x)
        self.wait(0.4)

        # Cut to the real sph look-down globe with its spin axis, then let the draped
        # globe run that same combined motion.
        self.caption("the real skyplothelper globe and its spin axis")
        img = self.dissolve_to("globe_precession.png")
        self.play(FadeOut(body), FadeOut(holder), FadeOut(tip), run_time=0.6)
        self.wait(1.2)
        self.caption("…and the same three motions, on the draped globe")
        img = self.flipbook(img, "precess_recap", run_time=4.4)
        self.wait(0.8)
        self.play(FadeOut(img), FadeOut(self._cap), run_time=0.7)
        self.wait(0.3)
