"""Compare side-task robot/physics properties with the preserved original runtime."""
import hashlib
import json
import argparse
from pathlib import Path

root=Path(__file__).resolve().parents[3];out=root/'runs/arx_dynamic_pick'
parser=argparse.ArgumentParser();parser.add_argument('--input',default='initial_setup.json');parser.add_argument('--report',default='runtime_configuration_verification.json');args=parser.parse_args()
before=json.loads((root/'runs/arx_recovery_20261021_2359/restored_fingerprint.json').read_text())['invariant']
now=json.loads((out/args.input).read_text())['fingerprint']['invariant']
def subset(d):
    return {'usd_robot':{k:v for k,v in d['usd'].items() if k.startswith('/World/Robot')},
        'runtime_robot':d['runtime_robot'],'configured':d['configured'],
        'physics_dt':d['physics_dt'],'rendering_dt':d['rendering_dt']}
old,new=subset(before),subset(now);differences=[]
def diff(a,b,path=''):
    if type(a)!=type(b):differences.append(path);return
    if isinstance(a,dict):
        for k in set(a)|set(b):
            if k not in a or k not in b:differences.append(path+'/'+k)
            else:diff(a[k],b[k],path+'/'+k)
    elif isinstance(a,list):
        if a!=b:differences.append(path)
    elif a!=b:differences.append(path)
diff(old,new)
raw_differences=list(differences)
# This new side-task writes its logs into a separate directory. It is an output
# location, not a robot, physics, collision, calibration, or controller property.
differences=[p for p in differences if p!='/configured/runtime/run_dir']
g=json.loads((root/'runs/autoresearch_arx_20261022_approach_regression/configuration_guard.json').read_text())
files=[k for k,v in g['file_sha256'].items() if not Path(k).exists() or hashlib.sha256(Path(k).read_bytes()).hexdigest()!=v]
report={'robot_and_config_differences':differences,'protected_source_changes':files,
    'raw_differences':raw_differences,'allowed_output_location_difference':'/configured/runtime/run_dir',
    'protected_files_checked':len(g['file_sha256']),'reference':'20261021_2359 restored original ARX runtime',
    'scope':'Robot composed USD, live DOF/rigidbody properties, configured controllers/physics and time steps. New authorized obstacle is separate.',
    'main_research_status':json.loads((root/'runs/autoresearch_arx_20261022_approach_regression/state.json').read_text())['status']}
(out/args.report).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
assert not differences and not files
