"""Validate a short release/lift connection using measured closed finger locks."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

from audit import WORKSPACE, ASSET, read, save
from geometry import JOINTS
from runtime_adapter import LocalContactValidator
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg


def pose7(t):
    q = Rotation.from_matrix(t[:3, :3]).as_quat()
    return [*t[:3, 3], q[3], *q[:3]]


def main(out):
    state = read(out/'closed_state.json'); request = read(out/'lift_request.json')
    original = WORKSPACE/'arx_core/scripts/curobo_arx/static_pick/reference/model_closed'
    cfg = yaml.safe_load((original/'arx.yml').read_text())
    cfg['robot_cfg']['kinematics']['urdf_path'] = str(original/'arx.urdf')
    measured = dict(zip(state['joint_names'], state['qpos']))
    for name in ['right_joint27', 'right_joint28']: cfg['robot_cfg']['kinematics']['lock_joints'][name] = measured[name]
    folder = out/'measured_lift'; folder.mkdir(exist_ok=False)
    model = folder/'arx.yml'; model.write_text(yaml.safe_dump(cfg,sort_keys=False))
    inverse = np.linalg.inv(state['links']['body4'])
    scene = {'cuboid':{}}
    for name, box in state['scene_cuboids'].items():
        t = np.eye(4); t[:3,3] = box['center']
        scene['cuboid'][name] = {'pose':pose7(inverse@t),'dims':box['dims']}
    meshes = {n:g for n,g in state['scene_meshes'].items() if not g['target']}
    if meshes: scene['mesh'] = {n:{'pose':pose7(inverse@np.asarray(g['world'])), 'file_path':g['file_path']} for n,g in meshes.items()}
    save(folder/'scene.json', scene)
    def checker(world):
        c = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=str(model),
             scene_model=world,n_meshes=8,n_cuboids=8,collision_activation_distance=0.))
        c.self_collision_cost = None
        return c
    full = checker(scene)
    other_scene = json.loads(json.dumps(scene)); other_scene['cuboid'].pop('table')
    other = checker(other_scene)
    context = SimpleNamespace(model=model,checker=full,
        limits=torch.tensor([state['joint_limits'][state['joint_names'].index(n)] for n in JOINTS],device='cuda'))
    validator = LocalContactValidator(context,state,other,ASSET,request['tolerance_m'])
    initial = np.array([measured[n] for n in JOINTS])
    path = validator.straight_approach(initial,np.array(request['q_lift']))
    # The contact allowance during release can only remove an existing contact.
    ceiling = float(request['tolerance_m']); release_ok = True; seen = 0
    for batch in validator.receipts:
        if path is not None and seen == path['validation_states']:
            ceiling = float(request['tolerance_m'])
        by_index = {r['index']: r for r in batch['narrow_phase']}
        for index in range(batch['states']):
            item = by_index.get(index)
            if item is not None and not item['valid']: continue
            depth = item['maximum_real_finger_depth_m'] if item is not None else 0.
            if depth > ceiling+.0001: release_ok = False
            ceiling = min(ceiling,depth)
        seen += batch['states']
    if path is not None:
        # The exact full dense path will also be checked by measured contacts
        # during physical release; native arm/neighbor/ground checks stay intact.
        path['stage']='lift_release'; path['measured_finger_locks']=cfg['robot_cfg']['kinematics']['lock_joints']
    save(out/'lift_contact_receipts.json', validator.receipts)
    save(out/'lift_plan_result.json', {'success':path is not None and release_ok,
        'reason':'success' if path is not None and release_ok else 'measured_closed_lift_path_collision',
        'paths':{'lift':path},'measured_finger_locks':cfg['robot_cfg']['kinematics']['lock_joints'],
        'release_nonincreasing_depth':release_ok,'physical_attempts':0})


if __name__ == '__main__': main(Path(sys.argv[1]).resolve())
