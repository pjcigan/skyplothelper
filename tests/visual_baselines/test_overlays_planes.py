"""pytest-mpl wrappers for the overlays.planes gallery."""

from _helpers import make_panel_tests
from render_overlays_planes import PANELS

make_panel_tests(globals(), PANELS)
