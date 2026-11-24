"""Launch the recorded ARX bottle fixture without changing mechanical settings."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
for relative in ('shared', 'arx-ops', 'arx-sim'):
    sys.path.insert(0, str(ROOT / relative))

from arx_sim import inprocess_app

original_load = inprocess_app.load_sim_config


def load_recorded_config(**kwargs):
    config = original_load(active_gpu=int(os.environ.get('ARX_SIM_GPU', '3')), **kwargs)
    reference = Path(os.environ['ARX_REPRO_REFERENCE']) / 'setup.json'
    historical = json.loads(reference.read_text())['fingerprint']['invariant']['configured']
    config['simulation']['robot_pose'] = historical['simulation']['robot_pose']
    config['task_layout']['initial_joint_positions'] = historical['task_layout']['initial_joint_positions']
    config['ros2']['domain_id'] = historical['ros2']['domain_id']
    config['services']['arx_ops_port'] = historical['services']['arx_ops_port']
    config['simulation']['camera_roles'] = os.environ.get('ARX_SIM_CAMERA_ROLES','head').split(',')
    return config


inprocess_app.load_sim_config = load_recorded_config
sys.path.insert(0, str(ROOT / 'scripts/curobo_arx/dynamic_pick'))
from sim_runner import main

if __name__ == '__main__':
    raise SystemExit(main())
