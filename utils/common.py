"""
Common utility functions for path resolution and environment setup.
"""

from pathlib import Path
from typing import Union


def resolve_data_path(path_str: Union[str, Path]) -> str:
    """
    Resolve data path relative to repo or workspace root.
    Searches in:
      1. Absolute path if it exists
      2. Relative to current working directory
      3. Relative to this script's directory / project root
      4. Relative to parent directory (e.g. workspace level)
    """
    path = Path(path_str)
    if path.is_absolute() and path.exists():
        return str(path)

    # Try relative to current working directory
    if path.exists():
        return str(path.resolve())

    # Try relative to project root (parent of utils/)
    project_root = Path(__file__).resolve().parent.parent
    candidate = (project_root / path).resolve()
    if candidate.exists():
        return str(candidate)

    # Try parent directory of project root
    candidate_parent = (project_root.parent / path).resolve()
    if candidate_parent.exists():
        return str(candidate_parent)

    raise FileNotFoundError(f"Could not resolve dataset path: {path_str}")

