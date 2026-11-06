"""Import stage modules by path. Stage folders aren't packages, and exercises and solutions
share file names, so each gets a unique module name."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def import_stage_module(stage_dir: Path, target: str, name: str):
    path = stage_dir / target / f"{name}.py"
    module_name = f"_planagents_{stage_dir.name}_{target}_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: Pydantic and dataclasses look the module up by name.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def sibling(this_file: str, name: str):
    """Import another module from the same folder, e.g. a lab building on your ex3."""
    path = Path(this_file).resolve()
    return import_stage_module(path.parents[1], path.parent.name, name)
