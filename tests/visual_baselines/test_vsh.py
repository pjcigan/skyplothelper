"""pytest-mpl wrappers for the VSH forward-model gallery."""

from _helpers import make_panel_tests
from render_vsh import PANELS

make_panel_tests(globals(), PANELS)
