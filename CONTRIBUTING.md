# Contributing to skyplothelper

Thanks for your interest! Bug reports, ideas, and patches are all genuinely
welcome, and the notes below cover the development workflow.

## What to expect

skyplothelper is maintained by a single developer with limited time, so it's
worth setting expectations before you invest effort:

- **Responses come as time allows.** Issues and pull requests may sit for a
  while — sometimes weeks — before they get a proper look. A quiet stretch
  means a busy period, not disinterest or rejection.
- **Not every report can be fixed, and not every pull request merged.** Some
  requests fall outside what one person can maintain long-term. If something
  is declined, that's a judgment about sustainable scope, not about the
  quality of your idea or your work.
- **For anything non-trivial, please open an issue before writing code.** A
  short conversation up front can spare you from building something that
  turns out not to fit, or that overlaps with work already underway.
- **Small, focused, tested changes are the easiest to accept** — and a bug
  report with a minimal reproducer is the single most useful thing you can
  send, since it removes most of the work of confirming the problem.

None of this is meant to wave you off; the project is better for the
contributions it gets. It's simply an honest picture of the pace, so nobody
is left wondering.

## Questions, bugs, and ideas are different things

A quick triage, since each needs something different:

- **"How do I do X?"** — usage questions are welcome in
  [Discussions](https://github.com/pjcigan/skyplothelper/discussions);
  please post them there rather than opening an issue, so the tracker stays
  focused on bugs and features. They may go unanswered during busy stretches,
  and the docs will usually get you an answer faster than I can: the [user
  guide](https://skyplothelper.readthedocs.io/en/latest/guide/index.html)
  explains each subsystem, the
  [tutorials](https://skyplothelper.readthedocs.io/en/latest/tutorials/index.html)
  are runnable end-to-end examples, and `sph.overview()` /
  `sph.recipes('<keyword>')` print an orientation map and copy-paste recipes
  without leaving your Python session.
- **"This behaves incorrectly."** — that's a bug; please open an issue with a
  minimal reproducer (below). These get the most attention, because a clear
  reproducer makes a fix cheap.
- **"It would be useful if…"** — a feature request; open an issue describing
  the *use case* rather than a proposed implementation, so the discussion can
  start from the problem you're trying to solve.

## Reporting bugs

Open an issue on GitHub with:

- A minimal reproducer (script or notebook cell)
- The exact error / traceback or unexpected output (with figure if relevant)
- Your Python, matplotlib, astropy, and skyplothelper versions

## Suggesting enhancements

Open an issue describing the use case before sending a PR for
non-trivial changes — there may be ongoing work in the same area that
you'd want to coordinate with, and it's the cheapest point at which to
find out whether something fits the scope the project can carry.

## Development setup

```bash
git clone https://github.com/pjcigan/skyplothelper
cd skyplothelper
pip install -e .[dev]
pytest tests/
ruff check .
```

If you're working on optional-dependency code paths, install the
relevant extra (e.g., `pip install -e .[healpix,dev]`).

## Pull requests

- Branch from `main`. Use a descriptive branch name (`fix/...`,
  `feature/...`, `docs/...`).
- Keep PRs focused — one logical change per PR. Bundle small refactors
  with the change that motivates them; avoid drive-by reformatting.
- Add or update tests for behavior changes. Tests live in `tests/`.
- Run `pytest tests/` and `ruff check .` locally; CI must pass before
  merge.
- Update `CHANGELOG.md` under the `[Version]` section.
- Opening a PR isn't a commitment on either side: review may take a
  while, and larger changes may need discussion or revision first — see
  [What to expect](#what-to-expect).

## Code style

- Follow the existing patterns in the codebase. `ruff` configuration in
  `pyproject.toml` is the source of truth for lint/format rules.
- The package is fully type-annotated and checked with `mypy` in strict
  mode (enforced in CI). Keep new and changed code annotated and
  mypy-clean.
- Keep public API docstrings informative (NumPy style) — they feed the
  Sphinx API reference.

## Commit messages

- Imperative, present tense: "Add coastlines overlay", not "Added
  coastlines overlay".
- Reference issue numbers where applicable: `Fix tick label sign in
  apply_offset_ticks (#42)`.

## Code of conduct

Be respectful, collaborative, and assume good intent. Constructive
disagreement is welcome; personal attacks are not.
