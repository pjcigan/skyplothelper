# Merge-verification visual gallery

Standalone Python scripts that render every public plotting function in
``skyplothelper`` to PNG (or display interactively). The intent is to
let a human eyeball that the merge didn't break any visual output —
assertions can't catch a plot that draws but draws *wrong*.

Lives under ``tests/integration/`` because it shares the same
**temporary** lifecycle: once the merge is verified to be clean, this
directory can be ``rm -rf``-ed wholesale. These are NOT canonical
tutorials (those will go in ``examples/`` later, more cultivated).

## Running

Each script saves PNGs to ``output/`` by default:

```bash
# Render one module group
python tests/integration/visual/render_globe_plotting.py

# Show interactively instead of saving
python tests/integration/visual/render_globe_plotting.py --show

# Render everything (calls all render_*.py scripts in --save mode)
python tests/integration/visual/render_all.py
```

After running, browse ``tests/integration/visual/output/`` —
PNGs are named ``<area>_<NN>_<short_description>.png`` so they sort
into a sensible order.

## What's NOT here

- **Assertion-based tests.** Those live in
  ``tests/integration/test_*.py`` and run via pytest as usual.
- **Visual-baseline diffing.** No pytest-mpl integration in this
  directory; that's a future-tests concern.
- **Tutorials.** Those will be cultivated examples in ``examples/``,
  not raw "did it draw?" smoke renders.

## Output dir is gitignored

``output/*.png`` is in ``.gitignore``. The directory exists (via
``output/.gitkeep``) but its contents are never committed — they're
regenerable from the scripts.
