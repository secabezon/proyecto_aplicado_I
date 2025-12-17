from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def project_root() -> Path:
    """
    Return the absolute project root path (parent of the src directory).

    Returns:
        Absolute path to the project root.
    """
    return Path(__file__).resolve().parents[1]


def load_config() -> ModuleType:
    """
    Load config/config.py as a Python module.

    Returns:
        A module-like object containing configuration attributes.

    Raises:
        FileNotFoundError: If config/config.py is missing.
        ImportError: If the module spec cannot be created or executed.
    """
    root = project_root()
    cfg_path = root / "config" / "config.py"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")

    spec = importlib.util.spec_from_file_location("project_config_module", str(cfg_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from: {cfg_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


cfg = load_config()
