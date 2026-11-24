"""Persistent original ARX simulator with per-arm continuous MPC execution."""
from pathlib import Path
import runpy

from arx_quiet_diagnostics_20261119 import install as quiet
quiet()
import arx_runtime_imports
arx_runtime_imports.install()
import graspcl_unseen_bench as bench
import graspcl_unified_protocol_20261120 as uniform
import arx_direct_grasp_20261120 as direct
import arx_mpc_guarded_bench as guarded

uniform.install()
guarded.install_variant()
original = bench.execute


def execute(trial, entry):
    registration = bench.common.read(bench.common.read(entry)['campaign_registration'])
    run, baseline = bench.core.batch.original_modules()
    bench.lift.install_execution_policy(run, baseline)
    with direct.execution_hook(run, registration['direct_grasp']):
        original(trial, entry)


bench.execute = execute
runpy.run_path(str(Path(__file__).resolve().with_name('graspcl_unseen_simulator_pinned_20261119.py')), run_name='__main__')
