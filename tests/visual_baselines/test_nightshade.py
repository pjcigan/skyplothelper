"""Visual baselines for the day/night terminator blend.

Panels register only when the example day/night world maps are present in
``examples/data/`` (they're untracked), so this file generates no tests on
clones that lack the imagery.
"""

from _helpers import make_panel_tests
from render_nightshade import PANELS

make_panel_tests(globals(), PANELS)
