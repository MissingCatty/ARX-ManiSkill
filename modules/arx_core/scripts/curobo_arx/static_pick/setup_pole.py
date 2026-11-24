import sys,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];TASK=ROOT/'runs/arx_dynamic_pick'
sys.path.insert(0,str(ROOT/'scripts/curobo_arx'));sys.path.insert(0,str(ROOT/'scripts/curobo_arx/dynamic_pick'))
from client import Client
from run_demo import check_sources,frozen_properties
out=TASK/sys.argv[1];out.mkdir(exist_ok=False)
def read(p):return json.loads(Path(p).read_text())
def save(n,v):(out/n).write_text(json.dumps(v,indent=2))
assert read(ROOT/'runs/autoresearch_arx_20261022_approach_regression/state.json')['status']=='paused_by_user'
save('protected_sources.json',{'count':check_sources()})
spec=read(TASK/'pole_layout_design/chosen.json');contract=read(TASK/'task_contract.json')
save('registration.json',{'task':'User-authorized new static pole case replacing failed rack case, same Rubik cube and original robot/scene','registered_unix':time.time(),'pole':spec,'scenario':contract['scenario'],'original_configuration':'frozen','main_research':'paused','selection_rule':'Model-only old path blocked and all old endpoints clear; no planning success used for layout selection','success_gate':'unchanged bilateral closure, five-stage convergence, hold rise >6cm, no forbidden measured contacts','weight_multiplier':100,'fresh_zerograsp':True})
c=Client(TASK/'mailbox');setup=c.request('dynamic_setup',scenario=contract['scenario']);save('setup.json',setup)
assert frozen_properties(setup['fingerprint'])==frozen_properties(read(TASK/'initial_setup.json')['fingerprint'])
c.request('grasp_viewport_state',restore=read(TASK/'viewport.json'));save('viewport_initial.json',c.request('grasp_viewport_state'))
save('pole.json',c.request('static_pole_create',center=spec['center'],height=spec['height'],radius=spec['radius']));save('fingerprint_before.json',c.request('grasp_research_fingerprint'))
shutil.copy2(Path(contract['source'])/'active_asset.json',out/'active_asset.json')
c.request('static_view_capture',path='/workspace/COOLAS-ARX/runs/arx_dynamic_pick/'+out.name+'/scene_initial.png')
print(out,flush=True)
