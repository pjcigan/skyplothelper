"""Run every render_*.py gallery script and report per-script status.

Designed for matrix testing across conda envs / Python versions:
auto-discovers all sibling ``render_*.py`` scripts, runs each as a
subprocess so a failure in one doesn't kill the rest, and prints a
per-script PASS/FAIL summary plus environment metadata (Python
version + key library versions) at the start.

Usage
-----
    # From the repo root, for each conda env:
    bash -ic 'cenv && python tests/integration/visual/render_all.py'
    bash -ic 'cenv && conda activate py312 && python tests/integration/visual/render_all.py'

    # Optionally clear the output dir before re-rendering:
    python tests/integration/visual/render_all.py --clean

    # Filter to a subset (regex match against script stem):
    python tests/integration/visual/render_all.py --filter "ticks|wcs"

Notes
-----
- Failing scripts do not abort the run; they are reported in the
  summary with their exit code.
- Per-script timing is reported. Total wall time is printed at the
  end.
- Output PNGs land in ``output/`` (the standard gallery location).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _SCRIPT_DIR / "output"


def _discover_scripts() -> list[Path]:
    """Return sorted list of render_*.py scripts in this directory,
    excluding render_all.py itself."""
    scripts = sorted(_SCRIPT_DIR.glob("render_*.py"))
    return [p for p in scripts if p.name != "render_all.py"]


def _print_environment() -> None:
    """Print Python version + key library versions for matrix-test
    record-keeping."""
    print("=" * 70)
    print("  Environment")
    print("=" * 70)
    print(f"  Python: {sys.version.split()[0]}  ({sys.executable})")
    libs = ["numpy", "matplotlib", "astropy", "shapely",
            "healpy", "cartopy", "reproject", "scipy"]
    for lib in libs:
        try:
            version = importlib.metadata.version(lib)
            print(f"  {lib:<14s} {version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"  {lib:<14s} (not installed)")
    print()


def _run_one(script: Path, env: dict) -> tuple[bool, float, str]:
    """Run a single render script and return (ok, elapsed_seconds, tail).

    `tail` is up to 25 lines of stderr+stdout from the script, useful
    when something fails. If the script succeeded we just keep the
    last few lines so the summary stays terse.
    """
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, env=env,
    )
    elapsed = time.monotonic() - t0
    ok = proc.returncode == 0
    output = (proc.stdout + proc.stderr).strip().splitlines()
    tail_lines = 25 if not ok else 5
    tail = "\n".join(output[-tail_lines:]) if output else "(no output)"
    return ok, elapsed, tail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--clean", action="store_true",
        help="Delete all PNGs in output/ before re-rendering.",
    )
    parser.add_argument(
        "--filter", default=None, metavar="REGEX",
        help="Only run scripts whose stem matches this regex.",
    )
    args = parser.parse_args()

    scripts = _discover_scripts()
    if args.filter is not None:
        pattern = re.compile(args.filter)
        scripts = [s for s in scripts if pattern.search(s.stem)]
        if not scripts:
            print(f"No render_*.py scripts match {args.filter!r}.",
                  file=sys.stderr)
            return 1

    _print_environment()

    if args.clean and _OUTPUT_DIR.exists():
        n_removed = 0
        for f in _OUTPUT_DIR.glob("*.png"):
            f.unlink()
            n_removed += 1
        print(f"  --clean: removed {n_removed} prior PNG(s) from "
              f"{_OUTPUT_DIR}\n")

    _OUTPUT_DIR.mkdir(exist_ok=True)
    n_pngs_before = len(list(_OUTPUT_DIR.glob("*.png")))

    # Subprocess env: ensure cwd is the repo root so the package import
    # resolves to the local skyplothelper rather than a stray shadowing
    # module file. Pass-through environment otherwise.
    env = dict(os.environ)

    print("=" * 70)
    print(f"  Running {len(scripts)} render script(s)")
    print("=" * 70)

    results: list[tuple[Path, bool, float, str]] = []
    t0 = time.monotonic()
    for script in scripts:
        print(f"\n>>> {script.name}", flush=True)
        ok, elapsed, tail = _run_one(script, env)
        marker = "PASS" if ok else "FAIL"
        print(f"    [{marker}] {elapsed:5.1f}s")
        if not ok:
            print("    --- last lines of output ---")
            for line in tail.splitlines():
                print(f"    {line}")
            print("    --- end ---")
        results.append((script, ok, elapsed, tail))
    total_time = time.monotonic() - t0

    n_pngs_after = len(list(_OUTPUT_DIR.glob("*.png")))

    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    n_pass = sum(1 for _, ok, _, _ in results if ok)
    n_fail = sum(1 for _, ok, _, _ in results if not ok)
    print(f"  scripts:        {len(results)} total, "
          f"{n_pass} passed, {n_fail} failed")
    print(f"  output PNGs:    {n_pngs_before} → {n_pngs_after} "
          f"(net +{n_pngs_after - n_pngs_before})")
    print(f"  wall time:      {total_time:.1f}s")

    if n_fail:
        print()
        print("  Failed scripts:")
        for script, ok, elapsed, _ in results:
            if not ok:
                print(f"    - {script.name}  ({elapsed:.1f}s)")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
