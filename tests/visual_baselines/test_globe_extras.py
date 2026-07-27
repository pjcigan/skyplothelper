"""pytest-mpl wrappers for the globe extras (baselines + insets) gallery."""

from _helpers import make_panel_tests
from render_globe_extras import PANELS

make_panel_tests(globals(), PANELS)
