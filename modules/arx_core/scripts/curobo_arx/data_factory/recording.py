"""Bounded asynchronous raw writer and independent v1 episode validation."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import queue
import threading
import numpy as np
from PIL import Image
from .common import ROOT, VERSION, STAGES, canonical, digest, read, sha, space, write


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Writer:
    """Never drop frames. Backpressure stops dispatch, overflow is a group fault."""
    def __init__(self, capacity=128):
        self.jobs = queue.Queue(capacity)
        self.error = None
        self.thread = threading.Thread(target=self._loop, name='arx-raw-writer', daemon=True)
        self.thread.start()

    def _loop(self):
        while True:
            job = self.jobs.get()
            try:
                if job is None:
                    return
                if self.error is None:
                    job()
            except BaseException as exc:
                self.error = exc
            finally:
                self.jobs.task_done()

    def check(self):
        if self.error:
            raise RuntimeError('Shared raw writer failed') from self.error

    def submit(self, job):
        self.check()
        try:
            self.jobs.put_nowait(job)
        except queue.Full as exc:
            raise RuntimeError('Raw writer backlog full; stop group without dropping training frames') from exc

    def drain(self):
        self.jobs.join()
        self.check()

    def close(self):
        self.jobs.put(None)
        self.jobs.join()
        self.thread.join()
        self.check()

    @property
    def dispatch_ready(self):
        self.check()
        return self.jobs.qsize() < self.jobs.maxsize//2


class Recording:
    def __init__(self, episode, task, writer, *, existing=False):
        self.directory = Path(episode)
        self.task, self.writer, self.rows, self.last_time = task, writer, 0, None
        self.directory.mkdir(exist_ok=existing, parents=True)
        (self.directory/'raw').mkdir()
        self.content = []

    def add(self, row, rgb, depth, *, extras=None):
        self.writer.check()
        t = row['frame_time']
        if self.last_time is not None and t <= self.last_time:
            raise ValueError('Nonmonotonic/reused camera frame in episode')
        if self.last_time is not None and not .19<=t-self.last_time<=.21:
            raise ValueError('Wrist recording must follow consecutive 0.2-second rendered ticks')
        if row['env_id'] != self.task['env'] or row['attempt_id'] != self.task['id']:
            raise ValueError('Cross-environment recording identity')
        if abs(t-row['capture_sim_time']) > 1e-4:
            raise ValueError('Image/state times differ')
        index = self.rows
        image = f'raw/{index:06d}.png'
        depth_path = f'raw/{index:06d}.npy'
        row = {**row, 'rgb_path': str(self.directory/image), 'depth_path': str(self.directory/depth_path),
               'rgb_relative': image, 'depth_relative': depth_path}
        rgb, depth = np.array(rgb, dtype=np.uint8, copy=True), np.array(depth, dtype=np.float32, copy=True)
        extra_arrays = {k: np.array(v, copy=True) for k, v in (extras or {}).items()}
        if 'capture' in row and not extra_arrays:
            raise ValueError('Capture metadata without raw instance/mask channels')
        if extra_arrays:
            if set(extra_arrays) != {'instance_ids','depth_valid','target','right_link27','right_link28'}:
                raise ValueError('Unexpected capture channels')
            row['capture_files'] = {k: f'raw/{index:06d}_{k}.npy' for k in extra_arrays}
        self.content.append(digest([sha_pixels(rgb), row['snapshot']['link_world']['right_eef_link'], row['stage'],
                                    (row.get('teacher_command') or {}).get('gripper_target')]))
        def persist():
            space(self.directory, rgb.nbytes + depth.nbytes + sum(a.nbytes for a in extra_arrays.values()) + 65536)
            Image.fromarray(rgb).save(self.directory/image)
            np.save(self.directory/depth_path, depth)
            for key, array in extra_arrays.items():np.save(self.directory/row['capture_files'][key], array)
            with (self.directory/'samples.jsonl').open('ab') as stream:
                stream.write(canonical(row)+b'\n')
                stream.flush()
        self.writer.submit(persist)
        self.rows += 1
        self.last_time = t


def sha_pixels(value):
    import hashlib
    return hashlib.sha256(value.tobytes()).hexdigest()


def physical(episode):
    """The original unchanged evaluator sees a verified, lossless role view."""
    episode = Path(episode)
    requests = read(episode/'requests.json')
    actual = {p.name.removesuffix('_execution.json') for p in episode.glob('*_execution.json')}
    listed = {r['label'] for r in requests}
    if actual - listed:
        raise ValueError('Execution receipt without registered command request')
    for label in listed | actual:
        for suffix in ('_command.json', '_execution.json', '_settling.json'):
            if not (episode/(label+suffix)).is_file():
                raise ValueError('Occurred/requested execution has missing evidence: '+label+suffix)
    evaluator = module('data_factory_frozen_egad_evaluator', 'scripts/curobo_arx/lilo_vla/egad_evaluate.py')
    roles=episode/'execution_roles.json'
    sequence=None
    if roles.exists():
        sequence=read(roles)['sequence']
        completed=list(read(episode/'grasp_report.json')['stage_timing'])
        expected=[r['label'] for r in requests if r['label'] in completed]
        if [item[1] for item in sequence]!=expected:
            raise ValueError('Role view must preserve all occurred actions in request order')
        for semantic,label in sequence:
            allowed=label if label in ('pregrasp','approach','close','lift','hold') else 'approach'
            if semantic!=allowed:raise ValueError('Unexpected physical role mapping')
    result = evaluator.evaluate(episode,sequence=sequence)
    write(episode/'physical_receipt.json', result)
    return result


def quality(episode):
    """Recompute provenance, synchrony, actual actions, onsets and chunks."""
    episode = Path(episode)
    meta = read(episode/'episode.json')
    if meta['version'] != VERSION or not meta['write_complete']:
        raise ValueError('Unsupported or incomplete new production protocol')
    if meta.get('visual_geometry_required'):
        visual=read(episode/'visual_receipt.json')
        if not visual.get('checked') or not visual.get('pass'):
            raise ValueError('Visual admission is absent or failed')
    rows = [json.loads(line) for line in (episode/'samples.jsonl').read_text().splitlines()]
    if len(rows) != meta['samples'] or len(rows) < 9:
        raise ValueError('Raw frame count/receipt incomplete')
    # observations imports its old contract constants, but not episode validation.
    import sys
    sys.path.insert(0, str(ROOT/'scripts/curobo_arx/lilo_vla'))
    from observations import encode_state, teacher_action, policy_observation
    times, items, files, content = [], [], {}, []
    request_map = {r['id']: r for r in read(episode/'requests.json')}
    for index, row in enumerate(rows):
        if row['env_id'] != meta['env_id'] or row['attempt_id'] != meta['attempt_id'] or row['object_id'] != meta['object_id']:
            raise ValueError('Cross-environment/attempt/object row')
        if row['camera_owner_root'] != meta['environment_root'] or not row['camera_source_path'].startswith(meta['environment_root']+'/Robot/'):
            raise ValueError('Camera source belongs to a different environment')
        if row['camera_role'] != 'right_wrist' or abs(row['frame_time']-row['capture_sim_time']) > 1e-4:
            raise ValueError('Wrong camera or stale frame/state association')
        if meta.get('visual_geometry_required') and not row.get('visual_geometry',{}).get('pass'):
            raise ValueError('Unchecked or invalid target/camera geometry in training frame')
        t = float(row['frame_time'])
        if not np.isfinite(t) or times and t <= times[-1]:
            raise ValueError('Invalid/nonmonotonic frame time')
        times.append(t)
        state = encode_state(row['snapshot'], 'grasp_goal_world_error_v1', meta['goal']['grasp_world'])
        if not np.allclose(state, row['state'], atol=1e-7, rtol=0):
            raise ValueError('Goal-state codec mismatch')
        for key in ('rgb_relative','depth_relative'):
            path = episode/row[key]
            if not path.is_file():
                raise ValueError('Missing raw RGB/depth')
            files[row[key]] = sha(path)
        with Image.open(episode/row['rgb_relative']) as image:
            image.verify()
        if meta.get('synchronized_capture_required') and 'capture' not in row:
            raise ValueError('Required synchronized capture is missing')
        if 'capture' in row:
            from scripts.curobo_arx.nearfield.capture import validate_saved_capture
            files.update(validate_saved_capture(episode, row))
        cmd = row.get('teacher_command') or {}
        if cmd and cmd['id'] not in request_map:
            raise ValueError('Unregistered executed teacher command')
        content.append([files[row['rgb_relative']], row['snapshot']['link_world']['right_eef_link'],
                        row.get('stage'), cmd.get('gripper_target')])
        if index == len(rows)-1:
            continue
        following = rows[index+1]
        if not .19 <= following['frame_time']-t <= .21:
            raise ValueError('Missing consecutive 0.2-second wrist frame')
        next_cmd = following.get('teacher_command') or {}
        onset = bool(next_cmd.get('id') and next_cmd.get('id') != cmd.get('id'))
        command = next_cmd if onset else cmd
        stage = following.get('stage') if onset else row.get('stage')
        if stage not in ('approach', 'close', 'lift', 'hold'):
            continue
        if not command or command.get('gripper_target') is None:
            raise ValueError('Local label lacks actual gripper command')
        action = teacher_action({**row, 'teacher_command': command}, following)
        if not np.isfinite(action).all() or (np.abs(action[:6])>1+1e-6).any():
            raise ValueError('Teacher action exceeds existing ARX action representation')
        items.append({'raw_index':index, 'frame_time':t, 'stage':stage, 'command_onset':onset,
                      'command_id':command['id'], 'state':state.tolist(), 'action':action.tolist()})
    # Entry is recorded immediately before the first local command, not synthesized later.
    entry = meta['actual_entry']
    if not any(abs(r['frame_time']-entry['frame_time'])<1e-6 and r['env_id']==entry['env_id'] for r in rows):
        raise ValueError('Actual handoff first frame absent from raw recording')
    onsets = {stage: sum(i['stage']==stage and i['command_onset'] for i in items) for stage in ('approach','close','lift','hold')}
    if any(value < 1 for value in onsets.values()):
        raise ValueError('Missing synchronized pre-command transition: '+str(onsets))
    starts = []
    for i in range(len(items)-7):
        chunk = items[i:i+8]
        if not all(b['raw_index']==a['raw_index']+1 for a,b in zip(chunk,chunk[1:])):
            continue
        if chunk[0]['stage'] in ('approach','lift') or chunk[0]['stage']=='close' and chunk[0]['command_onset']:
            starts.append(i)
    if not starts or not any(items[i]['stage']=='close' and items[i]['command_onset'] for i in starts):
        raise ValueError('No contiguous eight-step chunk including the pre-close observation')
    for name in ('samples.jsonl','episode.json','requests.json','capture_snapshot.json','selected_grasp.json', 'grasp_report.json'):
        files[name] = sha(episode/name)
    if meta.get('visual_geometry_required'):files['visual_receipt.json']=sha(episode/'visual_receipt.json')
    for name in ('execution_roles.json','decision_traces.json','interpreter_events.json',
                 'closure_request.json','closure_model_validation.json'):
        if (episode/name).exists():files[name]=sha(episode/name)
    for path in episode.glob('*_*.json'):
        if path.name.endswith(('_command.json','_execution.json','_settling.json','_optimization.json','_visual_validation.json')):
            files[path.name] = sha(path)
    from interaction_contract import DISPLAY_NAMES
    labels = {'version':VERSION, 'items':items, 'chunk_starts':starts, 'chunk_size':8,
              'state_convention':'grasp_goal_world_error_v1', 'action_convention':'world_delta_position_0.05m_rotation_0.5rad_gripper_1_open',
              'instruction':'pick up the '+DISPLAY_NAMES.get(meta['object_id'],meta['object_id'].replace('_',' '))}
    write(episode/'labels.json', labels)
    files['labels.json'] = sha(episode/'labels.json')
    result = {'pass':True, 'samples':len(rows), 'chunks':len(starts), 'onsets':onsets,
              'content_id':digest(content), 'files':files, 'physical_receipt_sha256':sha(episode/'physical_receipt.json'),
              'scope':'v1 native protocol; unchanged pure state/action/image codecs, no old protocol spoofing'}
    write(episode/'quality_receipt.json', result)
    return result
