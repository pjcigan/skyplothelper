"""pytest-mpl wrappers for the overlays.annotations gallery."""

from _helpers import make_panel_tests
from render_annotations import PANELS

make_panel_tests(globals(), PANELS)
