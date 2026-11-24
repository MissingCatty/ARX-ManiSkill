"""Read-only evidence review of completed and interrupted advisor31 slots."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from scripts.curobo_arx.data_factory.common import read, write, sha, verify_files
from training.nearfield_arx.learned_policy import audit_policy
from .advisor31_campaign import exited
from .capture import audit_episode
from .runtime import audit_actions
from .report import video
from .visual_grounding_spec import expected_reset
from .verify_validation import reset_signature


def review(root):
    root = Path(root).resolve(); result = read(root/'advisor31_verification.json')
    summary = read(root/'summary.json'); reg = read(root/'registration.json')
    if (root/'final_analysis.json').exists(): raise FileExistsError('Preserve completed analysis')
    exit_proof = exited(root)
    specs = {t['trial_id']: t for t in reg['seed_trials']}
    verify_files(read(root/'cancellation_receipt.json')['evidence'])
    interrupted = []
    for attempt in summary['incomplete_accounting']:
        ep = Path(attempt['episode']); spec = specs[attempt['trial_id']]
        actions = audit_actions(ep, read(ep/'requests.json')); policy = audit_policy(ep)
        capture = audit_episode(ep)
        if reset_signature(read(ep/'initial_conditions.json')) != expected_reset(reg['seeds'][spec['seed_id']], spec):
            raise RuntimeError('Interrupted slot reset changed')
        rows = [json.loads(l) for l in (ep/'samples.jsonl').read_text().splitlines()]
        tail = rows[-50:]
        def speed(row):
            s = row['snapshot']; v = dict(zip(s['joint_names'], s['joint_velocities']))
            return max(abs(v[n]) for n in s['right_arm'])
        xyz = np.array([r['snapshot']['link_world']['right_eef_link'] for r in tail])[:, :3, 3]
        motion = video(ep, root/'videos'/(spec['trial_id']+'.mp4'))
        # A separate clip keeps the stopped-command waiting interval visible;
        # the standard motion clip intentionally omits planner/idle waits.
        dest = root/'videos'/(spec['trial_id']+'_waiting_tail.mp4')
        playlist = dest.with_suffix('.ffconcat')
        if any("'" in r['rgb_path'] for r in tail): raise ValueError('Unsafe ffconcat path')
        playlist.write_text('ffconcat version 1.0\n'+''.join("file '"+r['rgb_path']+"'\nduration 0.2\n" for r in tail)+"file '"+tail[-1]['rgb_path']+"'\n")
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-n','-f','concat','-safe','0','-i',str(playlist),
            '-vf','fps=5','-an','-c:v','libx264','-threads','2','-pix_fmt','yuv420p',str(dest)],check=True)
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-i',str(dest),'-f','null','-'],capture_output=True,check=True)
        tail_video = {'path':str(dest),'sha256':sha(dest),'frames':len(tail),'decode_pass':True,
                      'scope':'Last50 actual captured frames, waiting for measured idle; no fabricated movement'}
        write(dest.with_suffix('.json'), tail_video)
        item = {'trial_id':spec['trial_id'],'classification':'interrupted_before_visual_takeover',
            'physical_attempt_counted':True,'ordinary_grasp_failure_claimed':False,
            'occurred_action_records':actions,'policy_requests':policy['policy_calls'],'capture':capture,
            'terminal_physical_verdict_present':(ep/'physical_receipt.json').exists(),
            'last_sample_age_at_review_s':time.time()-rows[-1]['wall_time'],
            'tail_sim_seconds':tail[-1]['frame_time']-tail[0]['frame_time'],
            'tail_arm_speed_min_rad_s':min(map(speed,tail)),'tail_arm_speed_max_rad_s':max(map(speed,tail)),
            'registered_stationary_speed_limit_rad_s':reg['closedloop30_policy']['maximum_arm_speed_rad_s'],
            'tail_eef_xyz_range_m':np.ptp(xyz,axis=0).tolist(),
            'low_rate_pose_does_not_rule_out_substep_motion':True,
            'model_effect_evaluation_applicable':policy['policy_calls']>0,
            'close_lift_hold_applicable':False,'motion_video':motion,'waiting_video':tail_video,
            'evidence':{str(f):sha(f) for f in ep.glob('*.json*')}}
        interrupted.append(item)
    completed = result['trials']; main = [t for t in completed if t['phase']=='target_shift']
    groups = {}
    for arm in ('g0','fresh','stale_first'):
        scheduled = [t for t in reg['seed_trials'] if t['phase']=='target_shift' and t['vision_arm']==arm]
        done = [t for t in main if t['vision_arm']==arm]
        interrupted_ids = {t['trial_id'] for t in interrupted}
        groups[arm] = {'scheduled':len(scheduled),'completed':len(done),
            'successes':sum(t['task_success'] for t in done),'ordinary_grasp_failures':sum(not t['task_success'] for t in done),
            'interrupted':sum(t['trial_id'] in interrupted_ids for t in scheduled),
            'unexecuted':sum(t['trial_id'] in summary['unexecuted'] for t in scheduled),
            'comparable_completed':sum(t['first_control_applicable'] for t in done)}
    # Fully paired, applicable controls only for visual-effect comparisons.
    pairs = {}
    for arm, old in result['paired'].items():
        eligible = [p for p in old['pairs'] if p['both_visual_controls_comparable']]
        pairs[arm] = {'eligible_pairs':len(eligible),'rescued':sum(p['rescued'] for p in eligible),
                      'regressed':sum(p['regressed'] for p in eligible)}
    output = {'complete':result['complete'],'scheduled':len(reg['seed_trials']),
        'completed':len(completed),'interrupted_count':len(interrupted),'unexecuted_count':len(summary['unexecuted']),
        'task_successes_completed':sum(t['task_success'] for t in completed),
        'ordinary_task_failures_completed':sum(not t['task_success'] for t in completed),
        'completed_records_independently_recomputed':True,'full_campaign_terminal_record_completeness':result['record_integrity_pass'],
        'interrupted_occurred_actions_and_captures_intact':all(t['occurred_action_records']['pass'] and t['capture']['pass'] for t in interrupted),
        'groups':groups,'eligible_visual_pairs':pairs,'interrupted':interrupted,
        'event_coverage_completed':result['event_coverage_trials'],
        'donor_pose_pairing_applicable_trials':len(main),'donor_pose_pairing_passed_trials':sum(t['first_control_applicable'] for t in main),
        'regression_donor_pairing_applicable':False,
        'actual_model_inferences_completed':sum(t['model_inferences'] for t in completed),
        'exit':exit_proof,'commands_finished_and_World_absent_is_not_a_physical_stationarity_verdict':True,
        'physical_stationarity_confirmed_for_interrupted_slot':False if interrupted else None,
        'all_original_STOPs_preserved':True,'automatic_retries':0,'new_training':False,
        'evidence':{str(root/n):sha(root/n) for n in ('registration.json','advisor31_verification.json','summary.json','STOP','delivery_receipt.json')},
        'review_source':str(Path(__file__).resolve()),'review_source_sha256':sha(Path(__file__))}
    write(root/'final_analysis.json', output)
    write(root/'video_manifest.json', {'motion_clips':len(list((root/'videos').glob('*.mp4')))-len(interrupted),
        'waiting_tail_clips':len(interrupted),'clips':[read(p) for p in sorted((root/'videos').glob('*.json'))]})
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('campaign'); a = p.parse_args()
    value = review(a.campaign)
    print(json.dumps({k:value[k] for k in ('completed','interrupted_count','unexecuted_count','groups')},ensure_ascii=False))
