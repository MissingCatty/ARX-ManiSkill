"""Read-only archived ARX geometry and simulation-clock regressions."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha, verify_files
from training.nearfield_arx import native_mask
from .advisor34_geometry import observe, first_plane
from .advisor34_policy import decision
from .advisor34_rules import StopSample, settle
from .advisor32_rules import settle as old_settle

PROJECT=Path(os.environ.get('ARX_TEST_PROJECT_ROOT',str(ROOT)))
PREVIOUS=PROJECT/'runs/nearfield_advisor33_native_20261111_07'


def sample(row):
    s=row['snapshot'];idx=[s['joint_names'].index(n) for n in s['right_arm']]
    return StopSample(float(s['sim_time']),tuple(np.array(s['link_world']['right_eef_link'])[:3,3]),
                      tuple(np.array(s['joint_positions'])[idx]),tuple(np.array(s['joint_velocities'])[idx]))


def run(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    reg=read(PREVIOUS/'registration.json');projector=native_mask.projector_for(reg)
    previous=read(PREVIOUS/'geometry_preflight.json');review=read(PROJECT/'runs/nearfield_advisor32_20261111_01/diagnosis_complete/review.json')
    refs={r['observation']:r for r in review['camera_views']};rows=[];pins={}
    for entry in previous['rows']:
        ref=refs[entry['source_observation']];rgb=Path(ref['depth']).with_suffix('.png')
        row=native_mask.bind(read(ref['observation']),rgb)
        asset=next(a['usd_path'] for a in reg['catalog']['objects'] if a['object_id']==row['object_id'])
        gate,masks,points=observe(projector,asset,row,np.load(ref['depth'],allow_pickle=False))
        first=first_plane(row['snapshot']['link_world']['right_eef_link'],'REFINE 0 0 0 0 0 0',points,gate['box'])
        req={'method':'D','vision_arm':'fresh','mode':'final_gate','motions_completed':2,
             'axial_corrections_completed':0,'inputs':{'eef_world':row['snapshot']['link_world']['right_eef_link']}}
        action=decision(req,'GRASP',gate,points,row,reg['learned_policy']['visual_contract']['depth_scale'])
        rows.append({'trial_id':entry['trial_id'],'stage':entry['stage'],
            'required_closure_positive':entry['required_closure_positive'],'historical_success':entry['historical_success'],
            'old_gate':entry['gate'],'gate':gate,'first_plane':first,'closure_action':action})
        pins.update(row['native_mask_binding']['files'])
        for p in (ref['observation'],ref['depth'],str(rgb)):pins[p]=sha(p)
    ep=next(PREVIOUS.glob('groups/*/trials/a33m_spam_canon_normal_g0/episodes/*'))
    ledger=read(ep/'policy_requests.json');path=Path(ledger[-1]['request']);req=read(path)
    row=native_mask.request_row(req);depth=np.load(req['depth'],allow_pickle=False)
    gate,masks,points=observe(projector,req['asset'],row,depth)
    action=decision(req,'GRASP',gate,points,row,reg['learned_policy']['visual_contract']['depth_scale'])
    spam={'trial_id':req['trial_id'],'old_gate':read(path.with_name(req['request_id']+'_reply.json'))['finger_box'],
          'gate':gate,'closure_action':action,'source_request':str(path)}
    pins.update(req['files']);pins[str(path)]=sha(path)
    positive=[r for r in rows if r['required_closure_positive']]
    geometry_pass=(len(rows)==20 and len(positive)==4 and all(r['gate']['pass'] and r['closure_action']['kind']=='close' for r in positive)
                   and gate['target_points']==124240 and gate['pass'] and action['kind']=='close'
                   and all(r['first_plane']['pass'] for r in rows if r['stage']=='H' and r['historical_success']))
    write(out/'geometry.json',{'pass':geometry_pass,'historical_views':rows,'spam_final':spam,
          'successful_preclose_states':4,'false_rejections':sum(not r['gate']['pass'] for r in positive),
          'physical_trials':0,'neural_inferences':0,'surface_and_inferred_envelope_reported_separately':True})
    context=read(PREVIOUS/'timeout_window_timing_context.json');rawpath=Path(context['raw'])
    verify_files({str(rawpath):context['sha256']})
    raw=[json.loads(l) for l in rawpath.read_text().splitlines()];samples=[sample(r) for r in raw][-30:]
    original=old_settle(samples,active=False,fault=None,elapsed_wall_s=10.)
    new=settle(samples,active=False,fault=None,elapsed_wall_s=10.,elapsed_sim_s=1.)
    # The archive ends after a stable one-second tail. Extending a constant
    # measured final state tests the clock logic, not an unobserved outcome.
    extension=[replace(samples[-1],sim_time=samples[-1].sim_time+.2*i) for i in range(1,11)]
    simulated=settle(samples+extension,active=False,fault=None,elapsed_wall_s=30.,elapsed_sim_s=3.)
    clock={'pass':original['timeout'] and not new['timeout'] and not new['pass'] and simulated['pass'],
        'archived_original_clock':original,'archived_new_clock':new,'controlled_stationary_extension':simulated,
        'extension_is_synthetic_clock_regression_not_physical_evidence':True,
        'archived_tail_samples':[asdict(s) for s in samples],
        'archived_2s_stability_or_recovered_grasp_proven':False,'source':str(rawpath),'sha256':sha(rawpath)}
    write(out/'clock.json',clock);pins[str(rawpath)]=sha(rawpath)
    for p in (PREVIOUS/'geometry_preflight.json',PREVIOUS/'timeout_window_timing_context.json',out/'geometry.json',out/'clock.json'):
        pins[str(p)]=sha(p)
    verify_files(pins)
    result={'pass':bool(geometry_pass and clock['pass']),'physical_trials':0,'neural_inferences':0,
            'gate_positives':5,'gate_false_rejections':sum(not r['gate']['pass'] for r in positive)+int(not gate['pass']),
            'files':pins}
    write(out/'verification.json',result)
    if not result['pass']:raise RuntimeError('Advisor34 archived ARX acceptance failed')
    return {k:v for k,v in result.items() if k!='files'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');print(run(p.parse_args().out))
