"""pytest-mpl wrappers for the data_plots gallery."""

from _helpers import make_panel_tests
from render_data_plots import PANELS

make_panel_tests(globals(), PANELS)
