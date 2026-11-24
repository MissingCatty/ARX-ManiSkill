"""Pure wrist-image D policy. Geometry is never used to synthesize an action."""
import argparse
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image

from scripts.curobo_arx.data_factory.common import read, write, sha, verify_files
from scripts.curobo_arx.data_factory.watchdog import exclusive_stop
from training.nearfield_arx import native_mask
from training.nearfield_arx.advisor40_train import install_parser, load_frozen, generate
from .advisor36_contract import example, interpret


def decision(req, prediction):
    if req['method'] != 'D' or req['mode'] != 'decision' or req['vision_arm'] != 'fresh':
        raise ValueError('Only a fresh pure-D inference is registered')
    if set(req['inputs']) != {'eef_world', 'stopped'} or req['inputs']['stopped'] is not True:
        raise ValueError('Target, history, or unconfirmed motion state crossed the policy boundary')
    return interpret(req['inputs']['eef_world'], prediction)


def serve(root):
    root = Path(root).resolve(); reg = read(root / 'registration.json'); policy = reg['learned_policy']
    directory = root / 'workers/policy'
    os.environ.update(CUDA_VISIBLE_DEVICES=policy['gpu_uuid'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    try:
        verify_files({**reg['execution_source_files'], **reg['dependencies']})
        install_parser()
        model, processor = load_frozen({'model': policy['base_model']}, Path(policy['checkpoints']['D']))
        warm = [generate(model, processor, item) for item in policy['warmups']['D']]
        write(directory / 'warmup.json', {'calls': warm, 'scored': False})
        write(directory / 'ready.json', {'registration_sha256': sha(root / 'registration.json'), 'wall': time.time()})
        while not (directory / 'STOP').exists() and not (root / 'STOP').exists():
            for queued in sorted((directory / 'queue').glob('*.json')):
                done = directory / 'done' / queued.name
                if done.exists():
                    continue
                path = Path(read(queued)['request']); req = read(path)
                verify_files(req['files'])
                if req['registration_sha256'] != sha(root / 'registration.json'):
                    raise RuntimeError('Policy request registration mismatch')
                trial = next(t for t in reg['seed_trials'] if t['trial_id'] == req['trial_id'])
                if trial['method'] != 'D' or trial['object_id'] != req['object_id']:
                    raise RuntimeError('Policy request slot mismatch')
                row = native_mask.request_row(req)
                depth = np.load(req['depth'], allow_pickle=False)
                masks, truth = native_mask.load(row, depth)
                views = native_mask.model_views(Image.open(req['rgb']), depth, masks, policy['visual_contract']['depth_scale'])
                images = [path.with_name(req['request_id'] + '_' + x + '.png') for x in ('rgb', 'depth')]
                for im, p in zip(views, images):
                    im.save(p)
                item = example(images, policy['visual_contract']['depth_scale'])
                prompt = path.with_name(req['request_id'] + '_input.json')
                write(prompt, {**item, 'model_consumed_input': True})
                generated = generate(model, processor, item)
                try:
                    action = decision(req, generated['prediction'])
                except ValueError as exc:
                    action = {'kind': 'reject', 'reason': 'invalid_prediction: ' + str(exc), 'goal_world': None}
                files = [*images, prompt]
                reply = {**generated, 'ok': True, 'request_id': req['request_id'], 'method': 'D',
                    'request_sha256': sha(path), 'registration_sha256': req['registration_sha256'],
                    'model_inference': True, 'checkpoint': policy['checkpoints']['D'],
                    'visual_control': {'images': list(map(str, images)), 'replaced': False},
                    'mask_counts': {k: int(m.sum()) for k, m in masks.items()},
                    'geometry_action_guidance': False, 'action': action,
                    'files': {str(p): sha(p) for p in files}}
                verify_files(req['files'])
                dest = path.with_name(req['request_id'] + '_reply.json')
                if dest.exists():
                    raise RuntimeError('Never repeat or overwrite inference')
                write(dest, reply); write(done, {'reply': str(dest), 'sha256': sha(dest)})
            time.sleep(.05)
        write(directory / 'exit.json', {'pass': True, 'wall': time.time()})
    except BaseException:
        error = traceback.format_exc()
        write(directory / 'FAULT.json', {'error': error})
        exclusive_stop(root / 'STOP', {'reason': 'policy_fault', 'error': error})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('campaign'); serve(p.parse_args().campaign)
