"""pytest-mpl wrappers for the globe.decorations gallery."""

from _helpers import make_panel_tests
from render_decorations_globe import PANELS

make_panel_tests(globals(), PANELS)
