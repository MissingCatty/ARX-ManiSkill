"""Return the held ARX cube along the recorded paths, release, and retreat."""
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
from client import Client

parser=argparse.ArgumentParser();parser.add_argument('--retreat-only',action='store_true');args=parser.parse_args()
root=Path(__file__).resolve().parents[2];out=root/'runs/arx_grasp';client=Client(root/'runs/arx_curobo/mailbox')
archive=out/'history'/time.strftime('%Y%m%d_%H%M%S');archive.mkdir(parents=True)
for item in out.iterdir():
    if item.is_file() and item.suffix in ('.json','.log','.png'):shutil.copy2(item,archive/item.name)
for name in ('camera','zerograsp','final_camera'):
    if (out/name).exists():shutil.copytree(out/name,archive/name)
(archive/'reset').mkdir();print('Archived',archive,flush=True)

def execute(label,positions,names,gripper,duration):
    snap=client.request('snapshot');q=np.array([snap['joint_positions'][snap['joint_names'].index(n)] for n in names])
    positions=np.asarray(positions)
    assert np.max(abs(q-positions[0]))<.04,'Current state differs from recorded return path'
    positions=np.vstack([q,positions]);times=np.r_[0,np.linspace(.3,duration,len(positions)-1)]
    r=client.request('trajectory',timeout=300,joint_names=names,position=positions.tolist(),time_from_start=times.tolist(),gripper_target=gripper)
    (archive/'reset'/f'{label}.json').write_text(json.dumps(r))
    unwanted=[c for c in r['contacts'] if c['num_contacts']>0 and not('/World/Robot/' in c['a'] and '/World/Robot/' in c['b']) and not('/World/Task/red_cube' in c['a']+c['b'] and any(n in c['a']+c['b'] for n in ('right_link27','right_link28')))]
    print(label,'settled',r['settled'],'unexpected contacts',len(unwanted),flush=True)
    if args.retreat_only and label=='retreat':
        start_time=r['final']['sim_time']-r['records'][-1]['time']
        assert not [c for c in unwanted if c['sim_time']>start_time+1.0],'Retraction did not clear the tabletop contact'
    else:
        assert not unwanted,'Return path made an unexpected contact'
    return r

lift=json.loads((out/'lift_command.json').read_text());names=lift['joint_names']
if not args.retreat_only:
    r=execute('lower',lift['position'][::-1],names,.1,6)
    q=lift['position'][0]
    r=execute('release',[q,q],names,-3.2,3)
approach=json.loads((out/'approach_command.json').read_text())
r=execute('retreat',approach['position'][::-1],names,-3.2,5)
pre=json.loads((out/'pregrasp_command.json').read_text())
r=execute('return_start',pre['position'][::-1],names,-3.2,6)
snap=client.request('snapshot');(out/'reset_snapshot.json').write_text(json.dumps(snap,indent=2))
z=np.array(snap['objects']['red_cube']['world'])[2,3]
assert .68<z<.73,f'Cube is not on the tabletop: z={z}'
print('READY: cube returned to table, arm retreated',flush=True)
