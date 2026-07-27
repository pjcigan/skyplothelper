"""pytest-mpl wrappers for the globe.plotting gallery."""

from _helpers import make_panel_tests
from render_globe_plotting import PANELS

make_panel_tests(globals(), PANELS)
