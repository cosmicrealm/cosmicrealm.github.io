#!/usr/bin/env python3
"""Run every numbered LLM mechanism example in an isolated process."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    example_files = sorted(ROOT.glob("[0-9][0-9]_*.py"))
    if len(example_files) != 12:
        print(f"FAIL: expected 12 examples, found {len(example_files)}", file=sys.stderr)
        return 1

    failures: list[tuple[str, str]] = []
    for path in example_files:
        completed = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        status = "PASS" if completed.returncode == 0 else "FAIL"
        print(f"[{status}] {path.name}")
        if completed.stdout.strip():
            print(completed.stdout.rstrip())
        if completed.returncode:
            failures.append((path.name, completed.stderr.rstrip()))

    if failures:
        print("\nExample failures:", file=sys.stderr)
        for name, stderr in failures:
            print(f"- {name}: {stderr}", file=sys.stderr)
        return 1

    print("\n12/12 examples passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
