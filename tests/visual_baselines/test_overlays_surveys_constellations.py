"""pytest-mpl wrappers for the surveys + constellations gallery."""

from _helpers import make_panel_tests
from render_overlays_surveys_constellations import PANELS

make_panel_tests(globals(), PANELS)
