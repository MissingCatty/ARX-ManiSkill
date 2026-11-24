"""Verify and summarize the matched offline H-entry version comparison."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation
from urdf_reference import URDFChain,NAMES

HERE=Path(__file__).resolve().parent
ROOT=HERE/'research/20261117_08_H_version_compare'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    reg=read(ROOT/'registration.json');assert read(ROOT/'progress.json')['phase']=='completed'
    for p,h in reg['source_hashes'].items():assert sha(p)==h,p
    assert hashlib.sha256(subprocess.check_output(['git','-C',reg['version_sources']['old'],'diff','--binary'])).hexdigest()==reg['legacy_uncommitted_diff_sha256']
    assert not subprocess.check_output(['git','-C',reg['version_sources']['new'],'diff','--name-only'],text=True).strip()
    chain=URDFChain(HERE.parent/'arx_core/scripts/curobo_arx/static_pick/reference/model/arx.urdf')
    configs={v:read(ROOT/'resident'/v/'configuration.json') for v in reg['versions']}
    for v,c in configs.items():
        assert c['version']==reg['version_expected'][v]
        assert c['num_seeds']==32 and c['use_lm_seed'] is True and c['seed_orientation_weight']==1.
        assert c['position_tolerance']==.002 and c['orientation_tolerance']==.03
        receipt=read(ROOT/'resident'/(v+'_exit.json'));assert receipt['exit_confirmed'] and receipt['returncode']==0
        assert not Path('/proc',str(receipt['pid'])).exists()
        assert read(ROOT/'resident'/(v+'_ready.json'))['details']['model_load_count']=={'open':1}
    queries={};receipts={};timings={};total_queries=0
    for v in reg['versions']:
        base=ROOT/'resident'/v
        q=list(map(json.loads,(base/'requests.jsonl').read_text().splitlines()))
        a=list(map(json.loads,(base/'calls.jsonl').read_text().splitlines()))
        assert len(q)==len(a)==259
        assert [r['request_id'] for r in q]==list(range(259))==[r['request_id'] for r in a]
        for request,answer in zip(q,a):
            for key in ['request_id','case','index','H_world','current_q']:assert request[key]==answer[key]
        total_queries+=len(q)
        queries[v]={(x['case'],x['index']):x for x in q if x['case']!='__warmup__'}
        receipts[v]={(x['case'],x['index']):x for x in a if x['case']!='__warmup__'}
        durations=np.array([x['wall_seconds'] for x in a if x['case']!='__warmup__'])
        timings[v]={'queries':len(durations),'median_IK_ms':float(np.median(durations)*1000),'p90_IK_ms':float(np.percentile(durations,90)*1000),'sum_IK_seconds':float(durations.sum())}
    assert queries['old'].keys()==queries['new'].keys() and len(queries['old'])==258
    for key in queries['old']:
        for field in ['H_world','current_q']:assert queries['old'][key][field]==queries['new'][key][field]
    rows=[];max_sphere_error=0.;dense_samples=0;max_position_error=max_orientation_error=0.
    names={'003_cracker_box':'饼干盒（正例）','004_sugar_box':'糖盒（正例）','037_scissors':'剪刀','007_tuna_fish_can':'金枪鱼罐头','026_sponge':'海绵','048_hammer':'锤子','030_fork':'叉子','012_strawberry':'草莓','017_orange':'橙子','022_windex_bottle':'清洁剂瓶','058_golf_ball':'高尔夫球'}
    for case in reg['cases']:
        identifier=case['id'];trial=Path(reg['source_campaign'])/'trials'/identifier;state=read(trial/'state.json')
        A={v:read(ROOT/'results'/v/identifier/'result.json') for v in reg['versions']}
        keys={(identifier,i) for i in case['indices']}
        for v in reg['versions']:
            assert {key for key in queries[v] if key[0]==identifier}==keys
            accepted={index for c,index in keys if receipts[v][(c,index)]['accepted']}
            assert accepted==set(A[v]['accepted_H_indices'])
            for filename in ['arx.yml','scene.json','nominal_closed.yml']:
                assert (trial/'plan'/filename).read_bytes()==(ROOT/'results'/v/identifier/'model'/filename).read_bytes()
            def check_endpoint(q,world):
                nonlocal max_position_error,max_orientation_error
                T,_=chain.fk(np.asarray(q));goal=np.linalg.inv(np.asarray(state['links']['body4']))@np.asarray(world)
                pe=float(np.linalg.norm(T[:3,3]-goal[:3,3]));re=float(Rotation.from_matrix(T[:3,:3]@goal[:3,:3].T).magnitude())
                assert pe<.002 and re<.03
                max_position_error=max(max_position_error,pe);max_orientation_error=max(max_orientation_error,re)
            for key in keys:
                result=receipts[v][key]
                if result['accepted']:check_endpoint(result['q'],queries[v][key]['H_world'])
            if A[v]['entry_path_passed']:
                folder=ROOT/'results'/v/identifier;plan=read(folder/'entry_plan.json');opt=read(folder/'model/pregrasp_optimization.json')
                assert plan['success'] and set(plan['paths'])=={'pregrasp'} and plan['full_grasp_validated'] is False
                assert opt['success'] and opt['invalid_samples']==0 and opt['max_limit_exceed_rad']<=1e-6
                assert opt['start_error_rad']<=.005 and opt['goal_error_rad']<=.005
                check_endpoint(plan['paths']['pregrasp']['q'][-1],plan['selected']['pregrasp_world'])
                dense_samples+=opt['dense_samples']
        old=np.load(ROOT/'results/old'/identifier/'model/effective_model.npz');new=np.load(ROOT/'results/new'/identifier/'model/effective_model.npz')
        assert np.array_equal(old['q'],new['q']) and np.array_equal(old['bounds'],new['bounds'])
        err=float(np.max(np.abs(old['spheres']-new['spheres'])));max_sphere_error=max(max_sphere_error,err);assert err<2e-6
        oa=set(A['old']['accepted_H_indices']);na=set(A['new']['accepted_H_indices'])
        row={'case':identifier,'object':next(label for model,label in names.items() if identifier.endswith(model)),'control':case['control'],
             'H_goals':len(case['indices']),'old_IK_passes':len(oa),'new_IK_passes':len(na),
             'old_path':A['old']['entry_path_passed'],'new_path':A['new']['entry_path_passed'],
             'old_selected':A['old']['selected_index'],'new_selected':A['new']['selected_index'],
             'newly_accepted_indices':sorted(na-oa),'newly_rejected_indices':sorted(oa-na)}
        rows.append(row)
    assert total_queries==518<=reg['maximum_queries_total']
    failures=[x for x in rows if not x['control']];controls=[x for x in rows if x['control']]
    result={'failed_H_targets_per_version':sum(x['H_goals'] for x in failures),'control_H_targets_per_version':len(controls),
        'failed_scene_H_IK_passes':{v:sum(x[v+'_IK_passes'] for x in failures) for v in reg['versions']},
        'failed_scenes_with_valid_H_path':{v:sum(x[v+'_path'] for x in failures) for v in reg['versions']},
        'control_H_IK_passes':{v:sum(x[v+'_IK_passes'] for x in controls) for v in reg['versions']},
        'control_H_path_passes':{v:sum(x[v+'_path'] for x in controls) for v in reg['versions']},
        'candidate_acceptance_differences':sum(len(x['newly_accepted_indices'])+len(x['newly_rejected_indices']) for x in rows),
        'all_selected_indices_match':all(x['old_selected']==x['new_selected'] for x in rows),
        'timings':timings,'rows':rows,'physical_trials':0,'cloud_requests':0}
    (ROOT/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (ROOT/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    verification={'passed':True,'protected_hashes_checked':len(reg['source_hashes']),'main_H_goals_per_version':258,
        'native_requests_including_warmup':total_queries,'native_receipts_including_warmup':total_queries,
        'unmatched_queries':0,'complete_pairs':11,'candidate_acceptance_differences':result['candidate_acceptance_differences'],
        'same_input_H_goals_and_start_states':True,'model_load_count_per_version':1,'runtime_robot_rebuilds':0,
        'maximum_old_new_sphere_difference_m':max_sphere_error,'independent_FK_max_position_error_m':max_position_error,
        'independent_FK_max_orientation_error_rad':max_orientation_error,'dense_validation_samples_in_accepted_paths':dense_samples,
        'legacy_uncommitted_diff_preserved':True,'official_new_source_unmodified':True,'both_owned_workers_exited':True,
        'physical_trials':0,'simulator_instances':0,'cloud_requests':0}
    (ROOT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','timings']},indent=2));print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
