"""Experimental H022 runner; separate registration required before use."""
"""Run a bounded, preregistered batch for an active ARX research agent.

It executes a declared scenario list, not an embedded autonomous LLM. Further
hypotheses require a new preregistration by the active agent within the budget.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from autoresearch import ROOT, PROTECTED, read, write, evaluate
from client import Client


def run(campaign, names):
    os.chdir(ROOT)
    campaign=Path(campaign).resolve();campaign.relative_to(ROOT/'runs')
    config=read(campaign/'campaign.json')
    declared=read(campaign/'scenarios.json')
    scenarios=[s for s in declared if not names or s['name'] in names]
    if not scenarios:raise ValueError('No declared scenarios')
    if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='zhaoyechong':raise RuntimeError('Wrong branch')
    lock=(ROOT/'runs/arx_curobo/autoresearch.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read(campaign/'state.json')
    if state.get('status')=='running':raise RuntimeError('Inspect interrupted running state before resume')
    trials=campaign/'trials';trials.mkdir(exist_ok=True)
    declared_names={s['name'] for s in declared}
    attempted=[p for p in trials.iterdir() if p.is_dir() and p.name in declared_names]
    if len(attempted)+len(scenarios)>config['max_physical_trials']:raise RuntimeError('Trial budget exceeded')
    if any((trials/s['name']).exists() for s in scenarios):raise RuntimeError('Never overwrite a trial')
    hypotheses=sorted(set(state.get('screened_hypotheses',[]))|{s['hypothesis_id'] for s in scenarios if s.get('hypothesis_id')})
    if len(hypotheses)>config['max_hypotheses']:raise RuntimeError('Hypothesis budget exceeded')
    baseline={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in PROTECTED}
    frozen=campaign/'frozen_baseline_hashes.json'
    if frozen.exists() and read(frozen)!=baseline:raise RuntimeError('Baseline/evaluator changed across hypotheses')
    if not frozen.exists():write(frozen,baseline)
    protected=PROTECTED+['scripts/curobo_arx/failure_scenarios.py','scripts/curobo_arx/grasp_depth_execute.py','scripts/curobo_arx/run_depth_grasp.sh',
        'scripts/curobo_arx/run_research_campaign.py','scripts/curobo_arx/grasp_sphere_execute.py','scripts/curobo_arx/run_sphere_grasp.sh','scripts/curobo_arx/sphere_refinement.py',
        'scripts/curobo_arx/grasp_shape_execute.py','scripts/curobo_arx/run_shape_grasp.sh','scripts/curobo_arx/shape_policy.py',
        'scripts/curobo_arx/client.py','scripts/curobo_arx/research_guard.py','scripts/curobo_arx/research_layout.py',
        'arx-sim/arx_sim/diagnostics/research_fingerprint.py','arx-sim/arx_sim/diagnostics/tabletop_actions.py',
        'scripts/curobo_arx/failure_geometry_scenarios.py','scripts/curobo_arx/run_geometry_campaign.py',
        'scripts/curobo_arx/grasp_geometry_execute.py','scripts/curobo_arx/run_geometry_grasp.sh',
        'scripts/curobo_arx/grasp_geometry_candidates.py','scripts/curobo_arx/original_candidate_fallback.py',
        'scripts/curobo_arx/tracking_robustness.py','scripts/curobo_arx/grasp_robust_execute.py',
        'scripts/curobo_arx/run_robust_grasp.sh','scripts/curobo_arx/failure_robust_scenarios.py','scripts/curobo_arx/run_robust_campaign.py',
        'scripts/curobo_arx/tracking_feedback.py','scripts/curobo_arx/audit_feedback_actions.py','scripts/curobo_arx/check_replayed_observation.py','scripts/curobo_arx/grasp_ring_tilt_execute.py',
        'scripts/curobo_arx/run_ring_tilt_grasp.sh','scripts/curobo_arx/failure_body_waypoint_scenarios.py','scripts/curobo_arx/run_ring_tilt_campaign.py',
        'scripts/curobo_arx/ring_grasp_candidates.py','scripts/curobo_arx/audit_ring_actions.py','scripts/curobo_arx/elliptic_torus_prototype.py',
        'scripts/curobo_arx/archive_recorded_frames.py','scripts/curobo_arx/ring_tilt_candidates.py','scripts/curobo_arx/audit_ring_tilt_actions.py',
        'runs/research_contract_20261019/fixed_gripper_quality_proxy_2mm.npz','runs/research_contract_20261019/fixed_gripper_quality_proxy_2mm.json']
    protected += ['scripts/curobo_arx/thin_cylinder_fit_prototype.py','scripts/curobo_arx/thin_boundary_candidates.py','scripts/curobo_arx/grasp_body_waypoint_execute.py','scripts/curobo_arx/run_body_waypoint_grasp.sh','scripts/curobo_arx/failure_body_waypoint_scenarios.py','scripts/curobo_arx/run_body_waypoint_campaign.py','scripts/curobo_arx/summarize_body_waypoint_campaign.py']
    protected += ['scripts/curobo_arx/thin_neural_candidates.py']
    protected += ['scripts/curobo_arx/long_axis_misalignment.py','scripts/curobo_arx/axis_adaptive_lift.py','scripts/curobo_arx/axis_palm_quality.py','scripts/curobo_arx/palm_target_quality.py']
    protected += ['scripts/curobo_arx/axis_body_center.py','scripts/curobo_arx/contact_normal_quality.py']
    protected += ['scripts/curobo_arx/body_waypoint_adaptive.py']
    protected += ['scripts/curobo_arx/orientation_ablation.py',
                  'scripts/curobo_arx/grasp_orientation_ablation_execute.py',
                  'scripts/curobo_arx/run_orientation_ablation_grasp.sh',
                  'scripts/curobo_arx/failure_orientation_ablation_scenarios.py',
                  'scripts/curobo_arx/run_orientation_ablation_campaign.py']
    source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in protected}
    batch=campaign/'batches'/str(time.time_ns());batch.mkdir(parents=True)
    write(batch/'protected_hashes.json',source_hashes);write(batch/'scenarios.json',scenarios)
    client=Client(ROOT/'runs/arx_curobo/mailbox')
    view=client.request('grasp_viewport_state');write(batch/'before_viewport.json',view)
    write(batch/'before_control.json',client.request('grasp_control_probe'))
    snapshot=client.request('snapshot');write(batch/'before_snapshot.json',snapshot)
    asset=read(ROOT/'runs/arx_grasp/active_asset.json')
    saved=next((obj for obj in snapshot['objects'].values() if obj['path']==snapshot['objects']['test_object']['path']),snapshot['objects']['test_object'])
    transform=saved['world']
    # Assets are restored with their current full pose, including any settled
    # orientation, so later captures use matching size and pose metadata.
    from scipy.spatial.transform import Rotation
    quat=Rotation.from_matrix([row[:3] for row in transform[:3]]).as_quat()
    restored={'asset_path':asset['container_path'],'size_m':asset['size_m'],
              'position':[row[3] for row in transform[:3]],'orientation':[float(quat[3]),*map(float,quat[:3])]}
    state.update(status='running',active_batch=str(batch),physical_trials=len(attempted),screened_hypotheses=hypotheses)
    write(campaign/'state.json',state)
    try:
        for scenario in scenarios:
            if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h for p,h in source_hashes.items()):raise RuntimeError('Protected source changed')
            state['active_trial']=scenario['name'];state['physical_trials']+=1;write(campaign/'state.json',state)
            command=[sys.executable,'scripts/curobo_arx/failure_orientation_ablation_scenarios.py','--scenarios',str(campaign/'scenarios.json'),
                     '--output',str(trials),'--names',scenario['name'],'--policy',scenario['research_policy'],
                     '--gpu','1','--timeout',str(config['trial_timeout_s'])]
            with (batch/(scenario['name']+'.log')).open('w') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            folder=trials/scenario['name']
            if not (folder/'result.json').is_file():raise RuntimeError('Runner failed before trial archive')
            score=evaluate(folder)
            from audit_ring_tilt_actions import audit
            extra=audit(folder)
            write(folder/'core_evaluation.json',score)
            score.update(pass_before_extra_action_checks=score['pass'],extra_action_checks=extra)
            score['pass']=bool(score['pass'] and extra['pass'])
            score['guard_pass']=bool(score['guard_pass'] and extra['guard_pass'])
            score['guard_violations'].extend(extra['guard_violations'])
            score['score']=score['metric_value']=float(score['pass'])
            write(folder/'independent_evaluation.json',score)
            with (campaign/'events.jsonl').open('a') as stream:stream.write(json.dumps({'trial':scenario['name'],'score':score,'time_unix':time.time()})+'\n')
            print(json.dumps({'trial':scenario['name'],'score':score}),flush=True)
            if result.returncode or not score['guard_pass']:raise RuntimeError('Infrastructure or guard failure; inspect before further motion')
        state.update(status='awaiting_agent_analysis',default_promoted=False)
    except BaseException as exc:
        state.update(status='stopped_error',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        integrity=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in source_hashes.items())
        state['last_batch_sources_unchanged']=integrity
        for source in protected:
            destination=batch/'source_at_end'/source;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/source,destination)
        write(batch/'integrity.json',{'protected_sources_unchanged':integrity,'scope':'At completed batch boundary'})
        try:
            if (client.root/'request.json').exists():raise RuntimeError('Mailbox busy; restoration deferred')
            shutil.copy2(ROOT/'runs/tabletop_trials/initial_setup.json',ROOT/'runs/arx_grasp/capture_snapshot.json')
            client.request('grasp_reset_setup')
            for name,pos in [('blue_cylinder',[4.33,1.79,.76]),('green_sphere',[4.46,1.96,.76]),('red_cube',[4.22,1.75,.73])]:client.request('place_object',name=name,position=pos)
            client.request('grasp_asset_load',**restored)
            write(ROOT/'runs/arx_grasp/active_asset.json',asset)
            client.request('grasp_viewport_state',restore=view)
            write(batch/'after_viewport.json',client.request('grasp_viewport_state'))
            write(batch/'after_control.json',client.request('grasp_control_probe'))
            write(batch/'after_snapshot.json',client.request('snapshot'))
            state['scene_restored']=True
        except Exception as exc:state['restore_error']=str(exc)
        state['last_batch_finished_unix']=time.time();write(campaign/'state.json',state)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);p.add_argument('--names',nargs='*')
    a=p.parse_args();run(a.campaign,a.names)
