"""pytest-mpl wrappers for the cone.plotting gallery."""

from _helpers import make_panel_tests
from render_cone_plotting import PANELS

make_panel_tests(globals(), PANELS)
