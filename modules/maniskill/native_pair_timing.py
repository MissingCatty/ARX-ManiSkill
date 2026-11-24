"""Timing-only instrumentation and native ZeroGrasp-only candidate adaptation."""
from datetime import datetime
from pathlib import Path
import json
import shutil
import time

import numpy as np

import graspcl_pipeline as common
import postprocess_bench as post
from postprocess_pipeline import processed_rows, canonical_candidates
from resident_backend import rpc as real_rpc


def native_candidates(directory):
    state = common.read(directory / 'state.json')
    camera = np.asarray(state['camera_world_optical'])
    rows, files = processed_rows(directory)
    candidates = canonical_candidates(rows, camera)
    maximum = sum(state['joint_limits'][state['joint_names'].index(n)][1] for n in ['right_joint27','right_joint28'])
    accepted, rejected = [], []
    for candidate in candidates:
        if candidate['width'] <= 0 or not np.isfinite(candidate['width']):
            rejected.append({'reason':'invalid_candidate_values','grasp':candidate})
        elif candidate['width'] > maximum+1e-6:
            rejected.append({'reason':'required_opening_exceeds_unchanged_arx_joint_limits','grasp':candidate})
        else:
            candidate['score'] = candidate['metadata']['source_network_score']
            accepted.append(candidate)
    destination = directory / 'candidates'
    destination.mkdir(exist_ok=False)
    (destination / 'zg_outputs').mkdir()
    # Compatibility with the validated canonical-pose planner, without GraspCL.
    (directory / 'graspcl').symlink_to(destination, target_is_directory=True)
    pool = {'candidates':accepted,'rejected':rejected,'recalled_before_filter':[],
        'counts':{'native_postprocessed_zero':len(rows),'zero_scored_input':len(rows),
                  'scored':len(accepted),'rejected':len(rejected),'recalled':{'object':0,'patch':0}},
        'camera':camera,'custom_geometry_filters':[],'zero_top_k_cap':None,
        'additional_finger_symmetry_candidates':False,'candidate_protocol':post.PROTOCOL,
        'pipeline_profile':'zerograsp_200','graspcl_invoked':False,
        'native_postprocessed_source_sha256':{str(p):common.sha(p) for p in files}}
    common.save(destination / 'candidate_pool.json',pool)
    for name in ['state.json','camera/params.json','camera/color.png','camera/depth.png','camera/mask.png']:
        target = destination / name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(directory / name,target)
    display = np.stack([rows[c['metadata']['postprocessed_row_index']] for c in accepted]) if accepted else np.empty((0,17))
    np.save(destination / 'raw_grasps.npy',display,allow_pickle=False)
    common.save(destination / 'network_candidates.json',{'model_name':'ZeroGrasp native post',
        'count':len(accepted),'candidate_stage':'Native ZeroGrasp corrected, collision-filtered and NMS candidates; before ARX planning',
        'score_label':'Network score','physical_attempt':False,'graspcl_invoked':False})
    return {'success':True,'counts':pool['counts'],'graspcl_invoked':False}


def execute(trial, entry_path):
    entry = common.read(entry_path)
    profile = entry['pipeline_profile']
    run, _ = post.batch.original_modules()
    events, stages = [], []
    totals = {'simulation_and_contact_trace_seconds':0.,'recording_capture_encode_seconds':0.,
              'recording_finish_seconds':0.,'execution_setup_seconds':0.,'visualization_seconds':0.}
    original_execution = run.Execution

    class TimedExecution(original_execution):
        def __init__(self,*args,**kwargs):
            start = time.perf_counter()
            super().__init__(*args,**kwargs)
            totals['execution_setup_seconds'] += time.perf_counter()-start

        def step(self):
            # Same commands, stepping, recording cadence and failure gate as run.Execution.step.
            start = time.perf_counter()
            self.env._step_action(run.torch.tensor(self.command[None],device=self.env.device))
            self.control_ticks += 1
            totals['simulation_and_contact_trace_seconds'] += time.perf_counter()-start
            if self.control_ticks % 2 == 0:
                start = time.perf_counter()
                self.writer.append_data(self.env.render().cpu().numpy()[0])
                if self.sensor_writer is not None:
                    obs = self.env.get_obs()
                    self.sensor_writer.append_data(obs['sensor_data'][self.env.perception_camera]['rgb'].cpu().numpy()[0])
                totals['recording_capture_encode_seconds'] += time.perf_counter()-start
            if self.forbidden:
                raise RuntimeError('forbidden_robot_contact')

        def segment(self,label,*args,**kwargs):
            start = time.perf_counter()
            try:
                return super().segment(label,*args,**kwargs)
            finally:
                stages.append({'stage':label,'wall_seconds':time.perf_counter()-start})

        def finish(self):
            start = time.perf_counter()
            try:
                return super().finish()
            finally:
                totals['recording_finish_seconds'] += time.perf_counter()-start

    original_render = post.post.render
    post.install(render=True)

    def timed_render(*args,**kwargs):
        start = time.perf_counter()
        try:
            return original_render(*args,**kwargs)
        finally:
            totals['visualization_seconds'] += time.perf_counter()-start

    def timed_rpc(path,message,timeout=1800):
        kind = Path(path).stem
        start = time.perf_counter()
        response = None
        event = {'component':kind,'started_at':datetime.now().astimezone().isoformat(),
                 'mode':next((k for k in ['observe','approach','lift'] if message.get(k)),'select')}
        try:
            if kind == 'graspcl' and profile == 'zerograsp_200':
                event['component'] = 'native_candidate_adapter'
                result = native_candidates(Path(message['directory']))
                response = {'status':'ok','returncode':0,'result':result,
                            'wall_seconds':time.perf_counter()-start,'graspcl_invoked':False}
            else:
                response = real_rpc(path,message,timeout=timeout)
            return response
        finally:
            event['client_wall_seconds'] = time.perf_counter()-start
            event['service_seconds'] = response.get('wall_seconds') if response else None
            event['status'] = response.get('status') if response else 'exception'
            event['service_queue_and_transport_seconds'] = max(0.,event['client_wall_seconds']-(event['service_seconds'] or 0.))
            events.append(event)

    old_rpc = post.rpc
    run.Execution = TimedExecution
    post.post.render = timed_render
    post.rpc = timed_rpc
    started = time.perf_counter()
    try:
        post.execute(trial,entry_path)
    finally:
        wrapper_wall = time.perf_counter()-started
        campaign = Path(entry['campaign_registration']).parent
        requests = []
        for path in (campaign / 'backend_requests').glob('*.json'):
            receipt = common.read(path)
            if receipt['slot'] != entry['id']:
                continue
            parse = datetime.fromisoformat
            queue = (parse(receipt['started_at'])-parse(receipt['queued_at'])).total_seconds() if 'started_at' in receipt else None
            requests.append({'kind':receipt['kind'],'flags':receipt['flags'],
                'queue_seconds':queue,'state':receipt['state']})
        components = {kind:sum(e['client_wall_seconds'] for e in events if e['component']==kind)
                      for kind in ['zerograsp','graspcl','curobo','native_candidate_adapter']}
        queue = sum(r['queue_seconds'] or 0. for r in requests)
        result = common.read(trial / 'result.json') if (trial / 'result.json').exists() else {}
        trial_wall = result.get('wall_seconds')
        accounted = sum(totals.values())+sum(components.values())+queue
        common.save(trial / 'timing.json',{'pipeline_profile':profile,
            'trial_wall_seconds':trial_wall,'wrapper_wall_seconds':wrapper_wall,
            **totals,'component_client_seconds':components,'gpu_backend_queue_seconds':queue,
            'other_trial_seconds':max(0.,trial_wall-accounted) if trial_wall is not None else None,
            'stage_wall_seconds':stages,'requests':events,'gpu_queue_requests':requests,
            'scope':'Component, queue, simulation, recording, visualization and other_trial times partition trial wall time. Stage totals overlap these categories and must not be added again. Controller slot time additionally includes process imports and independent verification.'})
        run.Execution = original_execution
        post.post.render = original_render
        post.rpc = old_rpc


def timing_summary(arm):
    clock = common.read(arm / 'round_timing.json')
    completed, measured, slot_times = [], [], []
    for path in (arm / 'claims').glob('*.json'):
        claim = common.read(path)
        if claim['state'] != 'complete':
            continue
        completed.append(claim)
        file = arm / 'trials' / path.stem / 'timing.json'
        if file.exists(): measured.append(common.read(file))
        slot_times.append((datetime.fromisoformat(claim['finished_at'])-datetime.fromisoformat(claim['claimed_at'])).total_seconds())
    trial_times = [t['trial_wall_seconds'] for t in measured if t['trial_wall_seconds'] is not None]
    elapsed = clock.get('round_wall_seconds',time.perf_counter()-clock['started_monotonic'])
    def stats(values):
        return {'mean':float(np.mean(values)),'median':float(np.median(values)),'p95':float(np.percentile(values,95))} if values else None
    means = {key:float(np.mean([t[key] for t in measured])) for key in [
        'simulation_and_contact_trace_seconds','recording_capture_encode_seconds','recording_finish_seconds',
        'execution_setup_seconds','visualization_seconds','gpu_backend_queue_seconds','other_trial_seconds'] if measured}
    for kind in ['zerograsp','graspcl','curobo','native_candidate_adapter']:
        if measured: means[kind+'_client_seconds'] = float(np.mean([t['component_client_seconds'][kind] for t in measured]))
    resource_wait = []
    for path in (arm / 'resource_wait').glob('*.jsonl'):
        resource_wait.extend(json.loads(line)['seconds'] for line in path.read_text().splitlines())
    output = {'phase':clock['phase'],'registered':200,'completed':len(completed),
        'timed_trials':len(measured),'admission_failures_without_execution':len(completed)-len(measured),
        'round_elapsed_seconds_excluding_shared_startup':elapsed,
        'shared_suite_startup_seconds':clock['shared_suite_startup_seconds'],
        'round_plus_shared_startup_seconds':elapsed+clock['shared_suite_startup_seconds'],
        'controller_slot_seconds':stats(slot_times),'trial_seconds':stats(trial_times),
        'mean_trial_breakdown_seconds':means,'scene_resource_wait_worker_seconds':sum(resource_wait),
        'estimated_total_round_seconds':elapsed*200/len(completed) if len(completed)>=5 and clock['phase']=='running' else None,
        'scope':'One round is 200 fresh slots with two parallel scene workers. Round wall time is not the sum of per-case durations. Both rounds share one startup; adding round_plus_shared_startup values would count startup twice.'}
    common.save(arm / 'timing_summary.json',output)
    return output
