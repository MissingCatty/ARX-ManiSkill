"""Continuous grasp simulator with the matching independent stage contract."""
from pathlib import Path
import runpy
from arx_quiet_diagnostics_20261119 import install as quiet
quiet()
import arx_runtime_imports
arx_runtime_imports.install()
import graspcl_unseen_bench as bench
import arx_direct_verifier_20261120 as verifier

_, baseline = bench.core.batch.original_modules()
verifier.install(baseline)
runpy.run_path(str(Path(__file__).with_name('arx_direct_simulator_20261120.py')), run_name='__main__')
