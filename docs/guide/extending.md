# Writing a custom overlay

skyplothelper is deliberately small at the core: every built-in decoration
draws onto a frame through the same three primitives, and they are public, so
your own overlay is a first-class citizen rather than a workaround. This page
is the contract — what to consult for the frame, the projection, and the
color, on each backend — so an extension behaves like the bundled ones instead
of guessing.

Whatever you draw, three questions come up, and each has one right answer to
consult:

| Concern | matplotlib | plotly |
|---|---|---|
| **What frame is this?** | {func}`~skyplothelper.to_lonlat` (`ax=`) | `fig.layout.meta["sph_frame"]` |
| **Where does (lon, lat) land?** | draw in world coords via {func}`~skyplothelper.world_transform` | {func}`~skyplothelper.plotly.project` with the figure's projection setup |
| **What color is theme-correct?** | `None` → `rcParams["text.color"]` | `fig.layout.meta["sph_fg"]` |

## The matplotlib side

On a WCSAxes you never convert coordinates to pixels yourself — you hand
`(lon, lat)` in the axes' own frame to the **world transform** and matplotlib
places them. {func}`~skyplothelper.to_lonlat` normalizes whatever the caller
passed (a `SkyCoord`, a tuple, arrays) into that frame; passing `ax=` converts
into the axes' frame rather than silently assuming ICRS. Follow the theme by
leaving `color=None` and resolving it to the appropriate rcParam — the same
convention the bundled decorations use (see {doc}`styling`):

```python
import matplotlib.pyplot as plt
import skyplothelper as sph

def mark_targets(ax, coords, *, color=None, size=60):
    """Draw an open ring at each target, following the frame and theme."""
    lon, lat = sph.to_lonlat(coords, ax=ax)      # normalize into the axes' frame
    if color is None:
        color = plt.rcParams["text.color"]       # None -> follow the theme
    tr = sph.world_transform(ax)                  # the (lon, lat) -> display transform
    ax.scatter(lon, lat, s=size, facecolor="none", edgecolor=color, transform=tr)
    return lon, lat
```

```python
from astropy.coordinates import SkyCoord
import astropy.units as u

c = SkyCoord([10, 120, 250] * u.deg, [20, -10, 40] * u.deg)
with sph.style_context(theme="dark_sky"):
    fig, ax = sph.allsky_figure(projection="AIT")
    mark_targets(ax, c)          # rings come out light, automatically
```

Because the coordinates go through `to_lonlat(ax=ax)` and the transform is the
axes' own, the same function draws correctly on an equatorial map, a galactic
one, or a tilted globe — you wrote no projection math.

## The plotly side

A plotly figure has no live axes to carry state, so {func}`~skyplothelper.plotly.make_figure`
stamps its setup onto `fig.layout.meta` — a free-form slot plotly preserves
through serialization. Your overlay **reads that meta** rather than repeating
the projection, and calls the projection primitive to get canvas coordinates.
The keys are `sph_projection`, `sph_center`, `sph_lat_center`, `sph_direction`,
`sph_frame`, and the resolved theme colors `sph_fg` / `sph_bg`:

```python
import skyplothelper as sph
import skyplothelper.plotly as sphpl

def add_target_marks(fig, coords, *, color=None):
    meta = fig.layout.meta
    lon, lat = sph.to_lonlat(coords, frame=meta["sph_frame"])   # into the figure's frame
    x, y = sphpl.project(lon, lat,
                         projection=meta["sph_projection"], center=meta["sph_center"],
                         lat_center=meta["sph_lat_center"], frame=meta["sph_frame"],
                         direction=meta["sph_direction"])       # -> canvas (x, y)
    if color is None:
        color = meta["sph_fg"]                                  # theme foreground
    fig.add_scatter(x=x, y=y, mode="markers",
                    marker=dict(color=color, size=10, symbol="circle-open"))
    return x, y
```

```python
fig = sphpl.make_figure(projection="AIT", frame="galactic", theme="dark")
add_target_marks(fig, c)         # projected into galactic, marker in the theme fg
```

`sph_frame` is `None` on a figure that never declared one; `to_lonlat` with
`frame=None` preserves the coordinate's own frame, so the pattern above works
either way without a special case.

```{note}
{func}`~skyplothelper.plotly.project` is a thin plotly-side wrapper exposing the
same core projection arguments as the matplotlib {func}`~skyplothelper.project`
(they are separate functions, and `sph.project` additionally accepts the
conic `pv2_1`/`pv2_2` parameters). It lives in the `skyplothelper.plotly`
namespace so an overlay written against that backend needs only one import. It
takes the projection setup as keyword arguments; there is no figure argument,
which is why you read the setup off `fig.layout.meta` and pass it through.
```

## The contract, in one line each

- **Never assume ICRS.** Route caller coordinates through
  {func}`~skyplothelper.to_lonlat` so a galactic or ecliptic frame is honored.
- **Never compute pixels.** Use {func}`~skyplothelper.world_transform` (mpl) or
  {func}`~skyplothelper.plotly.project` (plotly); both share the one projection
  pipeline that every built-in overlay uses, so seams and singular points are
  handled the same way.
- **Default `color=None` to the theme**, not to a literal — `rcParams["text.color"]`
  on matplotlib, `meta["sph_fg"]` on plotly — so your overlay is legible on a
  dark background without the caller passing colors. See {doc}`styling`.

The building blocks referenced here are in the API reference:
{func}`~skyplothelper.to_lonlat` and {func}`~skyplothelper.world_transform`
in {doc}`/api/frames`, and {func}`~skyplothelper.plotly.project` in
{doc}`/api/plotly`.
