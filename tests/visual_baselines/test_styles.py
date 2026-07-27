"""pytest-mpl wrappers for the style-layers gallery.

Covers the cycle palettes, rc themes, annotation palettes (incl. the
renamed ``denim``), the WCSAxes styling bridge, the composed all-sky shot,
and the ``set_base_style`` presets / ``MONO_STACK`` demos.
"""

from _helpers import make_panel_tests
from render_style import PANELS

make_panel_tests(globals(), PANELS)
