"""pytest-mpl wrappers for the co-visibility (skyplothelper.visibility) gallery."""

from _helpers import make_panel_tests
from render_covisibility import PANELS

make_panel_tests(globals(), PANELS)
