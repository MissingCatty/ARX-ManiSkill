"""Bounded no-grasp base/lift interface check on a development scene."""
import argparse
import importlib.util
from pathlib import Path
import shutil

import numpy as np

from arx_project.layout import ROOT, ensure_run_path, now, write_json
from arx_project.scenes import current_dataset, current_schedule
from modules.pipeline.experiments.dual_arm_grasp.run import imports


def run(root):
    if root.exists():
        raise ValueError('Preflight exists; preserve its evidence')
    root.mkdir(parents=True)
    write_json(root/'registration.json', {'at': now(), 'version': 'V4', 'physical_grasps': 0,
        'scope': 'One 2cm base shift and 1cm lift shift in development scene',
        'schedule': [current_schedule()[0]]})
    shutil.copy2(current_dataset()/'object_catalog.json', root/'object_catalog.json')
    imports()
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('v4_interface_simulation', here.parent/'repaired_grasp_benchmark/simulation.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    executor, _, _ = module.install(root)
    import torch
    from modules.curobo.dual_arm.model import ArmModel
    from modules.mobile_manipulation.placement import SweepChecker
    from modules.pipeline.experiments.mobile_dual_arm.runtime import Stationing
    from arx_project.layout import read_json
    settings = read_json(ROOT/'configs/experiments/sam3_graspgenx_v4.json')['placement']
    model = ArmModel()
    entry = current_schedule()[0]
    env = executor.make(entry['model_id'], scene_spec=entry['scene'])
    try:
        env.reset(seed=entry['seed'])
        for _ in range(80):
            env._step_action(torch.tensor(env.initial_q[None], device=env.device))
        state = executor.snapshot(env, env.get_obs())
        base = np.array(state['links']['base_link']); base[0, 3] -= .02; base[1, 3] += .02
        goal = {'base_world': base.tolist(), 'lift_m': state['qpos'][0]-.01}
        gate, path = SweepChecker(model, state, settings['transit_clearance_m']).path(goal, settings)
        write_json(root/'transit_check.json', gate)
        if not gate['passed']:
            raise RuntimeError('Development transit blocked: '+str(gate))
        write_json(root/'command.json', {'goal': goal, 'path': path})
        receipt = Stationing(None, executor, model, settings, 0).execute(env, root, state, goal, path)
        write_json(root/'receipt.json', receipt)
        write_json(root/'completion.json', {'passed': receipt['success'], 'physical_grasps': 0, 'at': now()})
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    run(ensure_run_path(args.root))
