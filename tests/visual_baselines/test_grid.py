"""pytest-mpl wrappers for the grid gallery."""

from _helpers import make_panel_tests
from render_grid import PANELS

make_panel_tests(globals(), PANELS)
