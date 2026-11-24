"""Compatibility entry point for the current verified scene gallery."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/build-scene-gallery.py"), run_name="__main__")
