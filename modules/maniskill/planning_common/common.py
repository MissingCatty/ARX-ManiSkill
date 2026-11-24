"""Pure data and file helpers; no simulator or CUDA initialization."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ARX=ROOT.parent/'arx_core'
ARM=[f'right_joint{i}' for i in range(21,27)]
FINGERS=['right_joint27','right_joint28']


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)








def ordinary_planner_rejection(result):
    if result.get('success') is not False:return None
    reason=result.get('reason')
    if reason in ('no_feasible_network_candidate','no_postprocessed_candidates',
                  'no_executable_postprocessed_candidate','observation_path_failed',
                  'measured_closure_lift_unreachable','measured_pregrasp_approach_unreachable'):return reason
    last=(result.get('error') or '').strip().splitlines()
    known={'AssertionError: Initial ARX state collides in cuRobo model':'initial_state_collision',
           'RuntimeError: Trajectory optimizer rejected all attempts; no PRM-only fallback':'trajectory_optimizer_rejection',
           'AssertionError: Optimized path failed dense scene/limit validation':'trajectory_dense_validation_rejection',
           'AssertionError: Optimized trajectory exceeds joint position limits':'trajectory_joint_limit_rejection',
           'AssertionError: Optimized endpoint mismatch':'trajectory_endpoint_rejection'}
    return known.get(last[-1]) if reason=='planner_error' and last else None
