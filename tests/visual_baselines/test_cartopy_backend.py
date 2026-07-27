"""pytest-mpl wrappers for the cartopy_backend gallery."""

from _helpers import make_panel_tests
from render_cartopy_backend import PANELS

make_panel_tests(globals(), PANELS)
