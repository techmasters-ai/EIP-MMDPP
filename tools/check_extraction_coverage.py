#!/usr/bin/env python
"""CLI: check extraction coverage for all ontology bundles.

Usage:
    python tools/check_extraction_coverage.py [bundle_key ...]

If no bundle keys are given, all directories under ontology_bundles/ that
contain a manifest.yaml are checked.

Exit code 0 = all pass, 1 = one or more bundles failed.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is on sys.path regardless of working directory.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from tools.extraction_coverage.rules import check_bundle  # noqa: E402


def _find_bundles(root: Path) -> list[Path]:
    bundles_root = root / "ontology_bundles"
    return sorted(
        d for d in bundles_root.iterdir()
        if d.is_dir()
        and not d.name.startswith("_")
        and (d / "manifest.yaml").exists()
    )


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    args = argv if argv is not None else sys.argv[1:]

    if args:
        bundles = [repo_root / "ontology_bundles" / key for key in args]
    else:
        bundles = _find_bundles(repo_root)

    if not bundles:
        print("No bundles found.", file=sys.stderr)
        return 1

    overall_ok = True
    for bundle_path in bundles:
        bundle_key = bundle_path.name
        try:
            errors, warnings = check_bundle(bundle_path)
        except Exception as exc:
            print(f"FAIL {bundle_key}: unexpected error: {exc}", file=sys.stderr)
            overall_ok = False
            continue

        for w in warnings:
            print(f"  WARN  {w}")

        if errors:
            overall_ok = False
            print(f"FAIL {bundle_key}")
            for e in errors:
                print(f"  ERROR {e}")
        else:
            if warnings:
                print(f"PASS {bundle_key}  ({len(warnings)} warning(s))")
            else:
                print(f"PASS {bundle_key}")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
