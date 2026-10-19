"""Module-boundary checks (master plan section 2).

Most important: decision-making code (fusion, forecasting, optimization, backend) must never import
the simulator's ground truth. Policies may only see Events and Calls, otherwise results leak.
A declared "oracle" ablation, if ever needed, must live in sim/ and be named as such.
Packages that do not exist yet (later phases) are checked as soon as they are created.
"""

import ast
from pathlib import Path

import pytest

from resqcity.common.config import PROJECT_ROOT

SRC = PROJECT_ROOT / "src" / "resqcity"

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
GROUND_TRUTH_MODULES = {"resqcity.sim.groundtruth", "resqcity.sim.generator"}


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
    return sorted((SRC / package).rglob("*.py"))


def _subpackage(module: str) -> str | None:
    """'resqcity.geo.network' -> 'geo'; None for third-party modules."""
    parts = module.split(".")
    return parts[1] if parts[0] == "resqcity" and len(parts) > 1 else None


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_package_boundaries(package: str):
    violations = []
    for path in _files(package):
        for module in _imports(path):
            if _subpackage(module) in FORBIDDEN[package]:
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


def test_boundary_checks_see_the_code():
    """Guard against a vacuous pass: the scan must find the real modules."""
    assert len(_files("geo")) >= 5 and len(_files("sim")) >= 2


def test_every_existing_package_is_known():
    existing = {p.name for p in SRC.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    unknown = existing - set(PACKAGES)
    assert not unknown, f"add new packages to PACKAGES/FORBIDDEN: {unknown}"
