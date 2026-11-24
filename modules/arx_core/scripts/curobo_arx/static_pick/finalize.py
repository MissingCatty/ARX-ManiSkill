import sys,json,copy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];TASK=ROOT/'runs/arx_dynamic_pick';out=TASK/sys.argv[1]
sys.path.insert(0,str(ROOT/'scripts/curobo_arx'));sys.path.insert(0,str(ROOT/'scripts/curobo_arx/dynamic_pick'))
from client import Client
from run_demo import check_sources,frozen_properties
c=Client(TASK/'mailbox')
def read(p):return json.loads(p.read_text())
def save(n,v):(out/n).write_text(json.dumps(v,indent=2))
save('recording_stop.json',c.request('grasp_video_stop'))
final=c.request('grasp_research_fingerprint');save('fingerprint_after.json',final);save('final_snapshot.json',c.request('snapshot'))
before=read(out/'fingerprint_before.json')
unchanged=frozen_properties(before)==frozen_properties(final)
contacts=[];commands=[]
for name in ('pregrasp','approach','close','lift','hold'):
 p=out/(name+'_execution.json')
 if p.exists():contacts+=read(p)['contacts']
 p=out/(name+'_command.json')
 if p.exists():commands.append(read(p))
snap=read(out/'capture_snapshot.json');target=snap['objects']['test_object']['path']
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
report=read(out/'grasp_report.json') if (out/'grasp_report.json').exists() else {'success':False}
result={'physical_grasp_success':report.get('success',False),'original_and_rack_configuration_unchanged':unchanged,'protected_sources_count':check_sources(),'rack_contact_events':sum('/World/StaticPickRack/' in e['a']+e['b'] for e in bad),'forbidden_contact_events':len(bad),'command_max_joint_limit_violation_rad':violation,'baseline_path_blocked':read(out/'baseline_path_check.json')['original_success_path_now_blocked'],'fresh_zerograsp_inference':(out/'zerograsp/object_1.grasp.npy').is_file(),'native_MPC_claim':False}
result['original_and_fixture_configuration_unchanged']=unchanged
result['pole_contact_events']=sum('/World/StaticPickPole/' in e['a']+e['b'] for e in bad)
result['executed_trajectory_count']=len(commands)
result['pass']=bool(result['physical_grasp_success'] and unchanged and not bad and violation<=1e-6 and result['baseline_path_blocked'] and result['fresh_zerograsp_inference'])
save('verification.json',result);save('all_execution_contacts.json',contacts);print(json.dumps(result),flush=True)
