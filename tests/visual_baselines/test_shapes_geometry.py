"""pytest-mpl wrappers for the geometry.shapes gallery."""

from _helpers import make_panel_tests
from render_shapes_geometry import PANELS

make_panel_tests(globals(), PANELS)
