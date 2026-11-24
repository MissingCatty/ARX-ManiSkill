"""Restore the user-requested original bottle scene for a new recording."""
import json,sys,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];TASK=ROOT/'runs/arx_dynamic_pick';out=TASK/'bottle_clear_00'
sys.path.insert(0,str(ROOT/'scripts/curobo_arx'));sys.path.insert(0,str(ROOT/'scripts/curobo_arx/dynamic_pick'))
from client import Client
from run_demo import check_sources,frozen_properties

def read(p):return json.loads(Path(p).read_text())
def save(n,v):(out/n).write_text(json.dumps(v,indent=2))
def original_properties(f):
 v=frozen_properties(f)
 # Transport port changed because the previous port is now occupied by another session.
 port=v.get('configured',{}).get('webrtc',{}).pop('port',None)
 assert port in (49200,49210), 'Unexpected WebRTC transport port'
 for section in ('usd','runtime_objects'):
  for p in list(v.get(section,{})):
   if p.startswith(('/World/StaticPickPole','/World/StaticPickRack')):v[section].pop(p)
 return v
assert read(ROOT/'runs/autoresearch_arx_20261022_approach_regression/state.json')['status']=='paused_by_user'
source=ROOT/'runs/weight_comparison_20261015/06_bbq_sauce_bottle_w100/01_bbq_sauce_bottle';asset=read(source/'active_asset.json')
scenario={'target':'test_object','asset':'bbq_sauce_bottle','placement':{'position':[4.048,2.08,.679+asset['size_m'][2]/2+.012]}}
save('registration.json',{'task':'User-requested fresh recording of historical bottle grasp without added obstacles','registered_unix':time.time(),'scenario':scenario,'historical_source':str(source),'main_research':'paused','original_configuration':'frozen; only remove side-task pole/rack as explicitly requested','success_gate':'original five-stage convergence, bilateral closure and >6cm hold for >=3s','weight_multiplier':100,'fresh_zerograsp':True,'webrtc_signal_port':49210,'webrtc_stream_port':49211,'transport_change_reason':'49200 is occupied by another session; no physics change'})
save('protected_sources.json',{'count':check_sources()})
c=Client(TASK/'mailbox')
# A fresh simulator loads the original room without any added side-task fixtures.
setup=c.request('dynamic_setup',scenario=scenario);save('setup.json',setup)
assert original_properties(setup['fingerprint'])==original_properties(read(out/'previous_fingerprint.json')), 'Original physics/configuration changed'
assert not any(p.startswith(('/World/StaticPickPole','/World/StaticPickRack','/World/DynamicPickObstacle')) for p in setup['fingerprint']['invariant']['usd'])
c.request('grasp_viewport_state',restore=read(out/'viewport_before.json'));save('viewport_initial.json',c.request('grasp_viewport_state'))
save('fingerprint_before.json',setup['fingerprint']);shutil.copy2(source/'active_asset.json',out/'active_asset.json')
c.request('static_view_capture',path='/workspace/COOLAS-ARX/runs/arx_dynamic_pick/bottle_clear_00/scene_initial.png')
save('setup_verification.json',{'original_configuration_unchanged':True,'no_added_obstacles':True,'protected_sources_count':check_sources()})
print('SETUP COMPLETE',flush=True)
