"""BaBank Analytics API package."""
import pathlib
import sys

# Enable cross-platform unpickling of WindowsPath objects on Linux/Render
if sys.platform != "win32":
    pathlib.WindowsPath = pathlib.PosixPath

from api.main import app

__all__ = ["app"]
