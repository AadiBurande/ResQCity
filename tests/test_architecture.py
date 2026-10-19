"""Module-boundary checks (master plan section 2).

Most important: decision-making code (fusion, forecasting, optimization, backend) must never import
the simulator's ground truth. Policies may only see Events and Calls, otherwise results leak.
A declared "oracle" ablation, if ever needed, must live in sim/ and be named as such.
"""

import ast
from pathlib import Path

import pytest

from common.config import PROJECT_ROOT

PACKAGES = [
    "common",
    "geo",
    "vision",
    "streams",
    "fusion",
    "forecasting",
    "optimization",
    "sim",
    "backend",
    "experiments",
    "paper",
    "tools",
]

# package -> project packages it must NOT import
FORBIDDEN: dict[str, set[str]] = {
    "common": set(PACKAGES) - {"common"},
    "geo": {
        "vision",
        "streams",
        "fusion",
        "forecasting",
        "optimization",
        "sim",
        "backend",
        "experiments",
    },
    "vision": set(PACKAGES) - {"vision", "common"},
    "fusion": {"sim", "vision", "backend", "optimization", "experiments"},
    "forecasting": {"sim", "vision", "backend", "optimization", "experiments"},
    "optimization": {"sim", "vision", "backend", "experiments"},
    "sim": {"vision", "backend", "experiments"},
    "backend": {"sim", "experiments"},
}
GROUND_TRUTH_MODULES = {"sim.groundtruth", "sim.generator"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def _files(package: str) -> list[Path]:
    return sorted((PROJECT_ROOT / package).rglob("*.py"))


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_package_boundaries(package: str):
    violations = []
    for path in _files(package):
        for module in _imports(path):
            top = module.split(".")[0]
            if top in FORBIDDEN[package]:
                violations.append(f"{path.relative_to(PROJECT_ROOT)} imports {module}")
    assert not violations, "\n".join(violations)


def test_ground_truth_only_used_inside_sim():
    violations = []
    for package in PACKAGES:
        if package == "sim":
            continue
        for path in _files(package):
            for module in _imports(path):
                if any(module == gt or module.startswith(gt + ".") for gt in GROUND_TRUTH_MODULES):
                    violations.append(f"{path.relative_to(PROJECT_ROOT)} imports {module}")
    assert not violations, "\n".join(violations)
