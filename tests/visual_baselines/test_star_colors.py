"""pytest-mpl wrappers for the perceived-star-color sequences gallery."""

from _helpers import make_panel_tests
from render_star_colors import PANELS

make_panel_tests(globals(), PANELS)
