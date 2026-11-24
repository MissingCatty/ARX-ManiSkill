"""Load the recorded EGAD49 configuration, calibrated cameras and original USDs."""
import copy
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
os.chdir(ROOT)
for relative in ('shared','arx-ops','arx-sim'):
    sys.path.insert(0,str(ROOT/relative))
from arx_sim import inprocess_app

original=inprocess_app.load_sim_config


def recorded(**kwargs):
    current=original(active_gpu=int(os.environ.get('ARX_SIM_GPU','3')),**kwargs)
    historical=json.loads(Path(os.environ['ARX_REPRO_REFERENCE']).joinpath('setup.json').read_text())
    config=copy.deepcopy(historical['fingerprint']['invariant']['configured'])
    # Only transport/output locations, compute placement and activation of the
    # existing calibrated wrist render sensor differ from the recorded config.
    config['runtime']['run_dir']=current['runtime']['run_dir']
    config['webrtc']['port']=current['webrtc']['port']
    config['simulation']['active_gpu']=current['simulation']['active_gpu']
    config['simulation']['physics_gpu']=current['simulation']['physics_gpu']
    config['simulation']['camera_roles']=['head','right_wrist']
    # Optional transport isolation for simultaneous historical fixtures. These
    # do not change robot, camera, rendering or scene/physics configuration.
    if os.environ.get('ARX_REPRO_OPS_PORT'):
        config['services']['arx_ops_port']=int(os.environ['ARX_REPRO_OPS_PORT'])
    if os.environ.get('ARX_REPRO_ROS_DOMAIN'):
        config['ros2']['domain_id']=int(os.environ['ARX_REPRO_ROS_DOMAIN'])
    Path(os.environ['ARX_EGAD_EFFECTIVE_CONFIG']).write_text(json.dumps(config,indent=2)+'\n')
    return config


inprocess_app.load_sim_config=recorded
sys.path.insert(0,str(ROOT/'scripts/curobo_arx/dynamic_pick'))
from sim_runner import main

if __name__=='__main__':raise SystemExit(main())
