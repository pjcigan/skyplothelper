# Merge verification tests

These tests exist to verify that the four-source merge (sph_dev_v5,
wcs_geometry_v7, wcs_tilted_globe, cone_frame → skyplothelper 1.0.0)
preserved the **expected runtime behavior** of every public plotting
function, decoration, annotation, and rendering helper.

## What these tests are

- **Behavior verification.** Each test checks that calling a public
  function produces the right kind of output: returns the expected
  artist type, adds the right number of children, propagates key
  parameters (color, count, label) through to the resulting artist.
- **Coverage of smoke-only modules.** The canonical test suite
  (`tests/test_*.py`) has light or zero assertions for several
  plotting / decoration / annotation modules. This directory fills
  those gaps for the duration of the post-merge verification window.

## What these tests are NOT

- **NOT** edge-case or regression tests (those go in the canonical
  test suite once they're identified).
- **NOT** visual-baseline (pytest-mpl) tests.
- **NOT** numerical-precision tests for math-heavy modules — those
  already exist in `tests/test_core.py`, `tests/test_coords.py`, etc.

## Lifecycle

This directory is **temporary**. Once Phil is satisfied that the merge
is clean and we move on to writing the long-term test suite, these
files can be:

1. **Deleted wholesale** (`rm -rf tests/merge_verification`) — the
   canonical test suite remains intact, the verification tests retire.
2. **Promoted selectively** by moving useful checks into the canonical
   `tests/test_<module>.py` files.

Either way, no cross-references from outside this directory should
exist. Keep the verification tests self-contained.

## File layout

Flat. One test file per (module, area) pair:

```
tests/merge_verification/
├── README.md                  (this file)
├── __init__.py                (empty)
├── test_plotting_globe.py     (Group 1a)
├── test_plotting_cone.py      (Group 1b)
├── test_annotations.py        (Group 1c)
├── test_decorations_globe.py  (Group 1d)
├── test_bands_geometry.py     (Group 1e)
... [more added group by group]
```

## Running

```bash
# Just the verification tests
pytest tests/merge_verification -v

# Everything except verification
pytest --ignore=tests/merge_verification

# Full suite (default)
pytest
```
