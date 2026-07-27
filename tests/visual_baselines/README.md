# Visual regression suite (pytest-mpl)

Local-only suite of pytest-mpl tests that compare gallery panels
against reference PNGs. Used as a regression guard during the
region-renderer API rename + default-flip work (see
[REGION_API_PROPOSAL.md](../../.claude/REGION_API_PROPOSAL.md)).

**Excluded from the default `pytest` run** by the
`addopts = "--ignore=tests/visual_baselines"` line in
`pyproject.toml`. The standard 779 assertion tests cover cross-platform
correctness; this suite covers pixel-level appearance and is intended
to run only on Phil's local dev machine.

Baselines and result PNGs are gitignored.

## Running the comparison

```bash
pytest tests/visual_baselines --mpl
```

`--mpl` enables image comparison. Without it, the tests still execute
each figure-builder (smoke check) but skip the comparison step.

## Regenerating baselines

After an intentional visual change, refresh the baselines from the
current code state:

```bash
pytest tests/visual_baselines --mpl-generate-path=tests/visual_baselines/baseline
```

This writes one PNG per test into the gitignored `baseline/` dir.
Subsequent `--mpl` runs compare against these.

## Tolerance

Default RMS tolerance is 10 (set in `conftest.py` decorators per test).
Tighten per-test where strict matching is desired; loosen when a panel
is known to be sub-pixel-noisy (e.g. anti-aliased text labels).

## Adding a new test

Each test wraps a figure-builder from one of the
`tests/merge_verification/visual/render_*.py` scripts. The render
script must expose a `PANELS = {name: builder, ...}` dictionary
where each builder is a no-arg function returning a `Figure`. Then
in this directory:

```python
import pytest
from render_overlays_surveys_constellations import PANELS

@pytest.mark.mpl_image_compare(filename="surveys_01_imaging_optical_ir.png",
                               tolerance=10)
def test_surveys_01():
    return PANELS["surveys_01_imaging_optical_ir"]()
```

One test per panel, explicit filename. The merge-verification render
scripts retain their standalone `main()` entrypoint (so
`python render_*.py` still works for manual gallery regen) — the
pytest-mpl suite just consumes the same builders programmatically.
