"""pytest-mpl wrappers for the HEALPix numerical-correctness gallery."""

from _helpers import make_panel_tests
from render_healpix_numerical import PANELS

make_panel_tests(globals(), PANELS)
