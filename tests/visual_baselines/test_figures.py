"""pytest-mpl wrappers for the figures gallery."""

from _helpers import make_panel_tests
from render_figures import PANELS

make_panel_tests(globals(), PANELS)
