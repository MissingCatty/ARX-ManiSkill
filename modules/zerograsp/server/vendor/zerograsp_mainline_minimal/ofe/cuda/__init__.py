"""CUDA extension namespace.

The checked-in extension may have been built for a different Python ABI.  Let
Python also search the active environment's installed ``ofe.cuda`` package so
deployments can use the extension compiled for their own interpreter.
"""

import importlib.machinery
import sys
from pathlib import Path

for entry in sys.path:
    candidate = Path(entry or ".") / "ofe" / "cuda"
    if candidate == Path(__file__).parent or not candidate.is_dir():
        continue
    if any((candidate / f"ofe{suffix}").is_file() for suffix in importlib.machinery.EXTENSION_SUFFIXES):
        __path__.append(str(candidate))
