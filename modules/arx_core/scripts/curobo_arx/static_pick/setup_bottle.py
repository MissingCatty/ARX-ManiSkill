import sys,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];TASK=ROOT/'runs/arx_dynamic_pick';out=TASK/'bottle_00';out.mkdir(exist_ok=False)
sys.path.insert(0,str(ROOT/'scripts/curobo_arx'));sys.path.insert(0,str(ROOT/'scripts/curobo_arx/dynamic_pick'))
from client import Client
from run_demo import check_sources,frozen_properties
def read(p):return json.loads(Path(p).read_text())
def save(n,v):(out/n).write_text(json.dumps(v,indent=2))
assert read(ROOT/'runs/autoresearch_arx_20261022_approach_regression/state.json')['status']=='paused_by_user'
source=ROOT/'runs/weight_comparison_20261015/06_bbq_sauce_bottle_w100/01_bbq_sauce_bottle';asset=read(source/'active_asset.json')
scenario={'target':'test_object','asset':'bbq_sauce_bottle','placement':{'position':[4.048,2.08,.679+asset['size_m'][2]/2+.012]}}
save('registration.json',{'task':'User requested bottle instead of cube; unchanged existing pole','registered_unix':time.time(),'scenario':scenario,'historical_source':str(source),'main_research':'paused','original_configuration':'frozen','success_gate':'original five-stage convergence, bilateral closure, hold rise >6cm, no forbidden measured contacts','weight_multiplier':100,'fresh_zerograsp':True,'compute_GPU':0})
save('protected_sources.json',{'count':check_sources()})
c=Client(TASK/'mailbox');setup=c.request('dynamic_setup',scenario=scenario);save('setup.json',setup)
assert frozen_properties(setup['fingerprint'])==frozen_properties(read(TASK/'pole_00/fingerprint_after.json')), 'Original physical configuration or pole changed'
c.request('grasp_viewport_state',restore=read(TASK/'viewport.json'));save('viewport_initial.json',c.request('grasp_viewport_state'))
save('fingerprint_before.json',setup['fingerprint']);shutil.copy2(TASK/'pole_00/pole.json',out/'pole.json');shutil.copy2(source/'active_asset.json',out/'active_asset.json')
c.request('static_view_capture',path='/workspace/COOLAS-ARX/runs/arx_dynamic_pick/bottle_00/scene_initial.png')
print(out,flush=True)
