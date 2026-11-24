import sys,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];TASK=ROOT/'runs/arx_dynamic_pick'
sys.path.insert(0,str(ROOT/'scripts/curobo_arx'));sys.path.insert(0,str(ROOT/'scripts/curobo_arx/dynamic_pick'))
from client import Client
from run_demo import check_sources,frozen_properties
name=sys.argv[1];out=TASK/name;out.mkdir(exist_ok=False)
def read(p):return json.loads(Path(p).read_text())
def save(n,v):(out/n).write_text(json.dumps(v,indent=2))
assert read(ROOT/'runs/autoresearch_arx_20261022_approach_regression/state.json')['status']=='paused_by_user'
save('protected_sources.json',{'count':check_sources()})
parts=[{'name':'crossbar','center':[3.98,2.24,.80],'dims':[.64,.045,.045]}, {'name':'left_post','center':[3.68,2.24,.4],'dims':[.045,.045,.8]}, {'name':'right_post','center':[4.28,2.24,.4],'dims':[.045,.045,.8]}, {'name':'left_foot','center':[3.68,2.26,.015],'dims':[.12,.13,.03]}, {'name':'right_foot','center':[4.28,2.26,.015],'dims':[.12,.13,.03]}]
contract=read(TASK/'task_contract.json')
save('registration.json',{'task':'Static rack avoidance with fresh ZeroGrasp inference + baseline cuRobo PRM and trajectory optimization','registered_unix':time.time(),'parts':parts,'scenario':contract['scenario'],'original_configuration':'frozen','main_research':'paused','original_user_view':str(TASK/'viewport.json'),'success_gate':'unchanged bilateral closure, five-stage convergence, hold rise > 6cm, no forbidden measured contacts','weight_multiplier':100,'no_dynamic_avoidance_claim':True})
c=Client(TASK/'mailbox');setup=c.request('dynamic_setup',scenario=contract['scenario']);save('setup.json',setup)
assert frozen_properties(setup['fingerprint'])==frozen_properties(read(TASK/'initial_setup.json')['fingerprint'])
c.request('grasp_viewport_state',restore=read(TASK/'viewport.json'));save('viewport_initial.json',c.request('grasp_viewport_state'))
save('rack.json',c.request('static_rack_create',parts=parts));save('fingerprint_before.json',c.request('grasp_research_fingerprint'))
shutil.copy2(Path(contract['source'])/'active_asset.json',out/'active_asset.json')
print(out,flush=True)
