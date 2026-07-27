"""Golden-file regression guard for ``aim_angles()``.

A captured baseline of ``aim_angles()`` outputs over a grid of markers, modes,
flip settings, targets, and ``rest_elev`` values. Unlike the self-consistency
checks in ``test_instruments.py`` (which verify the geometry agrees with itself),
this pins the exact numeric contract against a committed baseline — so a future
change to the solve is caught even if it stays internally consistent.

The axes use ``set_aspect('equal')`` so the data->display map is a pure equal
scale: the captured angles depend only on the data-space geometry, not on the
figure's margins/DPI, and stay stable across matplotlib versions.

Regenerate the baseline ONLY for an intended, reviewed contract change:

    python tests/test_aim_angles_golden.py --regen
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import skyplothelper as sph  # noqa: E402

GOLDEN = pathlib.Path(__file__).with_name("test_aim_angles_golden.json")

SITE = (5.0, 5.0)
GLOBE_CENTER = (5.0, 0.0)
# ``None`` = omit rest_elev entirely, so the *default* is captured too — a silent
# change to the default value is otherwise invisible to a grid that always passes
# rest_elev explicitly.
REST_ELEVS = (None, 30.0, 45.0, 60.0, 90.0, 120.0)
TARGETS = [(9, 9), (1, 9), (1, 1), (9, 1), (5, 9.5), (0.5, 5), (5, 0.2), (7.3, 6.1)]


def _axes():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set(xlim=(0, 10), ylim=(0, 10))
    ax.set_aspect("equal")
    fig.canvas.draw()
    return fig, ax


def _grid():
    fig, ax = _axes()
    rows = {}
    for marker in ("antenna", "telescope"):
        # 'aimed' ignores flip (a planted-only concept), so vary it only there.
        specs = [("aimed", "auto")] + [("planted", f) for f in ("auto", True, False)]
        for mode, flip in specs:
            for rest in REST_ELEVS:
                for tx, ty in TARGETS:
                    kw = dict(marker=marker, mode=mode, flip=flip,
                              target_coords="data")
                    if rest is not None:            # None -> exercise the default
                        kw["rest_elev"] = rest
                    if mode == "planted":
                        kw["globe_center"] = GLOBE_CENTER
                    r = sph.aim_angles(ax, SITE, (tx, ty), **kw)
                    rlabel = "default" if rest is None else f"{rest:g}"
                    key = f"{marker}|{mode}|{flip}|{rlabel}|{tx},{ty}"
                    rows[key] = {k: (None if v is None else round(float(v), 10))
                                 for k, v in r.items()}
    plt.close(fig)
    return rows


def test_aim_angles_matches_golden():
    golden = json.loads(GOLDEN.read_text())["cases"]
    live = _grid()
    assert set(live) == set(golden), "grid shape changed — regen the golden if intended"
    # Non-vacuous guard: the grid must actually exercise a spread of outputs, so
    # an all-constant regression can't pass trivially.
    rots = {c["rotation"] for c in live.values() if c["rotation"] is not None}
    assert len(rots) > 15, "grid collapsed — the test is not exercising the solve"
    for key, exp in golden.items():
        got = live[key]
        assert set(got) == set(exp), (key, "fields changed")
        for field, ev in exp.items():
            gv = got[field]
            if ev is None or gv is None:
                assert ev == gv, (key, field, ev, gv)
            else:
                # Circular distance, not a linear tolerance: every field here
                # is an angle in degrees, so a value at the +180/-180 seam must
                # compare equal to its wrapped twin. A plain isclose would fail
                # there spuriously if a future grid target landed on the seam
                # (none of the current 8 do). Still fails genuine differences —
                # verified against 0.01° and 2° cases near the seam.
                d = abs((gv - ev + 180.0) % 360.0 - 180.0)
                assert d < 1e-8, (key, field, ev, gv)


if __name__ == "__main__":
    import sys
    if "--regen" in sys.argv:
        payload = {"note": "regen via --regen only on an intended contract change",
                   "cases": _grid()}
        GOLDEN.write_text(json.dumps(payload, indent=0))
        print("wrote", GOLDEN, "with", len(payload["cases"]), "cases")
    else:
        test_aim_angles_matches_golden()
        print("OK")
