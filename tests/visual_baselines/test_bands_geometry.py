"""pytest-mpl wrappers for the geometry.bands gallery."""

from _helpers import make_panel_tests
from render_bands_geometry import PANELS

make_panel_tests(globals(), PANELS)
