"""Persistent original simulator with the registered uniform processing fix."""
from pathlib import Path
import runpy
from arx_quiet_diagnostics_20261119 import install as quiet

quiet()
import arx_runtime_imports
arx_runtime_imports.install()
from graspcl_unified_protocol_20261120 import install
install()
runpy.run_path(str(Path(__file__).resolve().with_name('graspcl_unseen_simulator_pinned_20261119.py')), run_name='__main__')
