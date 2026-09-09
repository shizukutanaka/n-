from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_public_package_imports_work_in_fresh_interpreters() -> None:
    root = Path(__file__).resolve().parents[1]
    modules = (
        "nmesh.orchestrate",
        "nmesh.spec",
        "nmesh.bench",
        "nmesh.planner",
        "nmesh.cli",
    )
    for module in modules:
        result = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"import {module} failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
