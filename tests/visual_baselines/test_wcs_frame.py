"""pytest-mpl wrappers for the wcs_frame gallery."""

from _helpers import make_panel_tests
from render_wcs_frame import PANELS

make_panel_tests(globals(), PANELS)
