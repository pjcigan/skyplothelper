"""pytest-mpl wrappers for the ticks gallery."""

from _helpers import make_panel_tests
from render_ticks import PANELS

make_panel_tests(globals(), PANELS)
