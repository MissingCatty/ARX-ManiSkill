"""Verify actual execution and preserve the unmodified viewport recording."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];TASK=ROOT/'runs/arx_dynamic_pick';out=TASK/sys.argv[1]
sys.path.insert(0,str(ROOT/'scripts/curobo_arx'));sys.path.insert(0,str(ROOT/'scripts/curobo_arx/dynamic_pick'))
from client import Client
from run_demo import check_sources,frozen_properties

def read(p):return json.loads(Path(p).read_text())
def save(n,v):(out/n).write_text(json.dumps(v,indent=2))
c=Client(TASK/'mailbox');save('recording_stop.json',c.request('grasp_video_stop'))
final=c.request('grasp_research_fingerprint');save('fingerprint_after.json',final);save('final_snapshot.json',c.request('snapshot'))
unchanged=frozen_properties(read(out/'fingerprint_before.json'))==frozen_properties(final)
snap=read(out/'capture_snapshot.json');target=snap['objects']['test_object']['path'];contacts=[];commands=[]
for stage in ('pregrasp','approach','close','lift','hold'):
 p=out/(stage+'_execution.json')
 if p.exists():contacts+=read(p)['contacts']
 p=out/(stage+'_command.json')
 if p.exists():commands.append(read(p))
bad=[]
for e in contacts:
 if e['num_contacts']<=0:continue
 actors=e['a']+e['b']
 if '/World/Robot/' in e['a'] and '/World/Robot/' in e['b']:continue
 if target in actors and any(f in actors for f in ('right_link27','right_link28')):continue
 bad.append(e)
violation=0.
for cmd in commands:
 bounds=np.array([snap['joint_limits'][snap['joint_names'].index(n)] for n in cmd['joint_names']]);q=np.array(cmd['position']);violation=max(violation,float(np.maximum(bounds[:,0]-q,q-bounds[:,1]).clip(0).max()))
r=read(out/'grasp_report.json');clear=not any(p.startswith(('/World/StaticPickPole','/World/StaticPickRack','/World/DynamicPickObstacle')) for p in final['invariant']['usd'])
v={'physical_grasp_success':r.get('success',False),'original_configuration_unchanged':unchanged,'protected_sources_count':check_sources(),'no_added_obstacles':clear,'forbidden_contact_events':len(bad),'command_max_joint_limit_violation_rad':violation,'fresh_zerograsp_inference':(out/'zerograsp/object_1.grasp.npy').is_file(),'executed_trajectory_count':len(commands),'held_min_rise_m':r.get('held_object_min_z_m',0)-r.get('initial_object_z_m',0)}
v['pass']=bool(v['physical_grasp_success'] and unchanged and clear and not bad and violation<=1e-6)
save('verification.json',v);save('all_execution_contacts.json',contacts);print(json.dumps(v),flush=True)
