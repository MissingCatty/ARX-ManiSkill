"""Resolve original native provenance and isolate legacy replay imports."""
import os
from pathlib import Path
import subprocess
import sys

from arx_project.json_io import read_json
from arx_project.layout import ROOT, sha256


def native_trial_origin(root, method, case, *, project_root=ROOT):
    """Locate the original receipt without changing its absolute-path contract."""
    current = Path(root).resolve()
    visited = set()
    while True:
        if current in visited:
            raise RuntimeError('Cycle in accepted evidence lineage')
        visited.add(current)
        reg = read_json(current/'registration.json')
        lineage = reg.get('continuation')
        if not lineage:
            return current/'methods'/method/'trials'/case
        inherited = lineage['method_slots'][method]
        if case not in inherited['complete']+inherited['interrupted']:
            return current/'methods'/method/'trials'/case
        predecessor = (Path(project_root)/lineage['predecessor']).resolve()
        expected = lineage['predecessor_evidence']['registration.json']
        if sha256(predecessor/'registration.json') != expected:
            raise RuntimeError('Original candidate provenance registration changed')
        current = predecessor


def rebuild_generator_candidates(directory):
    """Run the unchanged converter in a fresh process with its own common module.

    The evaluator also imports a module named common. Keeping this rebuild in a
    separate interpreter prevents either module from replacing the other's API.
    """
    program = '''import sys
from pathlib import Path
from modules.pipeline.experiments.rayst3r_graspgenx import common
sys.modules['common'] = common
from modules.pipeline.experiments.repaired_grasp_benchmark.repair_contract import make_candidates
make_candidates(Path(sys.argv[1]))
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    result = subprocess.run([sys.executable, '-c', program, str(Path(directory).resolve())],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError('Independent native candidate rebuild failed: '+result.stderr[-2400:])
