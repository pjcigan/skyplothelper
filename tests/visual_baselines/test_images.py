"""pytest-mpl wrappers for the images gallery."""

from _helpers import make_panel_tests
from render_images import PANELS

make_panel_tests(globals(), PANELS)
