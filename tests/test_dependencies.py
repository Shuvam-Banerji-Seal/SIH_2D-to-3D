"""What the code imports must be declared in ``pyproject.toml``.

``tifffile`` once went undeclared: the GeoTIFF writer imported it, but it only arrived as a side effect of
``scikit-image`` (the ``depth`` extra), so an install without that extra failed at export time.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

from drone3d import cli

ROOT = Path(__file__).resolve().parents[1]

# import name -> distribution name, where the two differ
DISTRIBUTION = {
    "cv2": "opencv-contrib-python-headless",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "skimage": "scikit-image",
    "pynvml": "nvidia-ml-py",
    "multipart": "python-multipart",
}
# imported from source checked out by tools/setup_third_party.sh or the third_party/ submodules
FROM_SOURCE = {"marigoldv2", "moge", "romav2"}


def _normalise(requirement: str) -> str:
    """``'uvicorn[standard]>=0.29'`` -> ``'uvicorn'``; case and ``-_.`` folded."""
    name = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement.strip())
    assert name, requirement
    return re.sub(r"[-_.]+", "-", name.group()).lower()


def _declared() -> tuple[set[str], set[str]]:
    """(core dependencies, every declared distribution incl. extras and the dev group)."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    core = {_normalise(r) for r in project["project"]["dependencies"]}
    everything = set(core)
    for requirements in project["project"]["optional-dependencies"].values():
        everything |= {_normalise(r) for r in requirements}
    for requirements in project["dependency-groups"].values():
        everything |= {_normalise(r) for r in requirements}
    return core, everything


def _imported_third_party(directory: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(directory.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top not in sys.stdlib_module_names and top != "drone3d":
                    found.setdefault(top, path)
    return found


def test_every_third_party_import_in_src_is_declared() -> None:
    _, declared = _declared()

    undeclared = {
        module: str(path.relative_to(ROOT))
        for module, path in _imported_third_party(ROOT / "src").items()
        if module not in FROM_SOURCE
        and _normalise(DISTRIBUTION.get(module, module)) not in declared
    }

    assert not undeclared, f"imported but not declared in pyproject.toml: {undeclared}"


def test_every_core_dependency_is_checked_by_doctor() -> None:
    core, _ = _declared()
    doctor = {_normalise(DISTRIBUTION.get(module, module)) for module in cli._CORE_MODULES}

    assert core == doctor
