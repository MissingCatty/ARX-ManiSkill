"""Original persistent simulation with periodic diagnostics disabled."""
from pathlib import Path
import runpy
from arx_quiet_diagnostics_20261119 import install

install()
runpy.run_path(str(Path(__file__).resolve().with_name('graspcl_unseen_simulator_pinned_20261119.py')),run_name='__main__')

