"""Frozen C/D inference plus explicitly attributed contact/controller requests."""
import argparse
import os
from pathlib import Path
import time
import traceback
import numpy as np
from PIL import Image
from scripts.curobo_arx.data_factory.common import read, write, sha, digest, verify_files
from scripts.curobo_arx.data_factory.watchdog import exclusive_stop
from training.nearfield_arx import lateral_recovery, closure_recovery, entry_guard
from training.nearfield_arx.projection import Projector, model_views
from training.nearfield_arx.receipt_numbers import same_pose_diagnostics
from .closedloop30_policy import validate_checkpoints
from .attribution_spec import geometry_prediction
from .visual_grounding_spec import alignment, first_decision
from . import advisor31_rules as rules


class Models:
    """Keep both immutable checkpoints in RAM, with only one resident on CUDA."""
    def __init__(self, reg):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import PeftModel
        from safetensors.torch import load_file
        from training.nearfield_arx.visual_fit import projector_module
        from training.nearfield_arx.learned_policy import Models as Original
        validate_checkpoints(reg)
        self.reg = reg; self.models = {}; self.processors = {}; self.active = None
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.35)
        for method in ('C', 'D'):
            folder = Path(reg['learned_policy']['checkpoints'][method])
            contract = read(folder/'arx_policy_contract.json')
            model = AutoModelForImageTextToText.from_pretrained(reg['learned_policy']['base_model'],
                       local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa')
            model = PeftModel.from_pretrained(model, folder, is_trainable=False)
            name, merger = projector_module(model)
            if name != contract['projector_module']:
                raise ValueError('Frozen projector changed')
            merger.float(); merger.load_state_dict(load_file(folder/'projector.safetensors'), strict=True)
            model.requires_grad_(False); model.eval(); model.config.use_cache = True
            self.models[method] = model
            self.processors[method] = AutoProcessor.from_pretrained(folder, local_files_only=True,
                                         min_pixels=65536, max_pixels=65536)
        self.original_generate = Original.generate

    def generate(self, method, item):
        import torch
        started = time.perf_counter()
        if self.active != method:
            if self.active is not None:
                self.models[self.active].to('cpu')
                torch.cuda.empty_cache()
            self.models[method].to('cuda'); self.active = method
        result = self.original_generate(self, method, item)
        result['device_switch_and_generation_seconds'] = time.perf_counter()-started
        return result


def trial_for(reg, request):
    trial = next(t for t in reg['seed_trials'] if t['trial_id'] == request['trial_id'])
    if trial['method'] != request['method'] or trial['G0_world'] != request['inputs']['G0_world']:
        raise ValueError('Registered method or G0 changed')
    return trial


def visual_control(reg, request, actual):
    trial = trial_for(reg, request)
    match = None
    if trial['phase'] == 'target_shift' and first_decision(request):
        match = alignment(request, reg['visual_donors'][trial['object_id']])
    applicable = match is None or match['pass']
    swapped = applicable and trial['vision_arm'] == 'stale_first' and first_decision(request)
    images = reg['visual_donors'][trial['object_id']]['images'] if swapped else list(map(str, actual))
    return {'applicable': bool(applicable), 'replaced': bool(swapped), 'alignment': match,
            'model_images': images, 'actual_observation': request['observation'],
            'replacement_scope': 'First decision only; all safety observations and later model images fresh'}


def verify_reply(reg, request, reply):
    trial = trial_for(reg, request); method = trial['method']
    directory = Path(request['observation']).parent
    actual = [directory/(request['request_id']+'_'+name+'.png') for name in ('rgb', 'depth')]
    expected = visual_control(reg, request, actual)
    if not same_pose_diagnostics(expected, reply['visual_control']) or reply['vision_arm'] != trial['vision_arm']:
        raise ValueError('Visual intervention changed')
    contact = rules.contact_from_request(request)
    if not same_pose_diagnostics(contact, reply['contact_control']):
        raise ValueError('C contact decision changed')
    generated_contact = expected['applicable'] and contact['trigger']
    model_used = expected['applicable'] and trial['vision_arm'] != 'g0' and not generated_contact
    if reply['model_inference'] != model_used or reply['controller_contact_closure'] != generated_contact:
        raise ValueError('Model/controller identity is wrong')
    if model_used and reply['checkpoint'] != reg['learned_policy']['checkpoints'][method]:
        raise ValueError('Wrong frozen checkpoint')
    if generated_contact and reply['prediction'] != 'GRASP':
        raise ValueError('C contact controller failed to issue its registered closure')
    if expected['applicable'] and trial['vision_arm'] == 'g0' and reply['prediction'] != geometry_prediction(request['inputs']):
        raise ValueError('Fixed G0 control target changed')
    depth_proof = reply['advisor31_depth']
    try:
        parts = closure_recovery.parse_action(method, reply['prediction'].strip())
    except ValueError:
        parts = None
    if method == 'D' and parts is not None:
        mask_path = directory/(request['request_id']+'_target_mask.npy')
        if reply['advisor31_depth_mask'] != str(mask_path) or str(mask_path) not in reply['files']:
            raise ValueError('D depth guard lacks its bound actual-view mask')
        verify_files({str(mask_path): reply['files'][str(mask_path)]})
        goal = entry_guard.residual_goal(request['inputs']['eef_world'], reply['prediction'].strip())
        recomputed = entry_guard.depth_admission(read(request['observation']),
                      np.load(request['depth'], allow_pickle=False), np.load(mask_path, allow_pickle=False),
                      goal, reg['learned_policy']['visual_contract']['depth_scale']['near_m'])
        if not same_pose_diagnostics(recomputed, depth_proof):
            raise ValueError('D depth guard disagrees with its recorded actual geometry')
    elif depth_proof is not None or reply['advisor31_depth_mask'] is not None:
        raise ValueError('Depth guard attached to an inapplicable action')
    return trial


def serve(campaign):
    root = Path(campaign).resolve(); reg = read(root/'registration.json'); p = reg['learned_policy']
    directory = root/'workers/policy'
    os.environ.update(CUDA_VISIBLE_DEVICES=p['gpu_uuid'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    try:
        verify_files({**reg['execution_source_files'], **reg['dependencies']})
        model = Models(reg); projector = Projector(p['robot_usd'])
        warmup = {m: [model.generate(m, item) for item in p['warmups'][m]] for m in ('D', 'C')}
        if any(len(v) != 2 for v in warmup.values()):
            raise ValueError('Exactly two uniform warmups per model required')
        write(directory/'warmup.json', {'calls': warmup, 'scored': False, 'only_one_model_on_GPU': True})
        write(directory/'ready.json', {'gpu_uuid': p['gpu_uuid'], 'registration_sha256': sha(root/'registration.json'), 'wall': time.time()})
        while not (root/'STOP').exists() and not (directory/'STOP').exists() and time.time() < reg['deadline_unix']:
            for queued in sorted((directory/'queue').glob('*.json')):
                done = directory/'done'/queued.name
                if done.exists():
                    continue
                start = time.perf_counter(); path = Path(read(queued)['request']); req = read(path)
                trial = trial_for(reg, req); method = req['method']
                if req['registration_sha256'] != sha(root/'registration.json'):
                    raise ValueError('Request registration changed')
                verify_files(req['files']); row = read(req['observation']); depth = np.load(req['depth'], allow_pickle=False)
                masks, projection, _ = projector.project(req['asset'], row, depth)
                images = model_views(Image.open(req['rgb']), depth, masks, p['visual_contract']['depth_scale'])
                actual = [path.with_name(req['request_id']+'_'+name+'.png') for name in ('rgb', 'depth')]
                for im, file in zip(images, actual):
                    im.save(file)
                visual = visual_control(reg, req, actual)
                if visual['replaced']:
                    verify_files({x: reg['dependencies'][x] for x in visual['model_images']})
                contact = rules.contact_from_request(req)
                contact_closure = visual['applicable'] and contact['trigger']
                used = bool(visual['applicable'] and trial['vision_arm'] != 'g0' and not contact_closure)
                item = lateral_recovery.inference_example(req['inputs'], method, visual['model_images'], p['visual_contract'])
                prompt = path.with_name(req['request_id']+'_input.json')
                write(prompt, {'instruction': item['instruction'], 'images': item['images'],
                              'model_consumed_input': used, 'visual_control': visual, 'contact_control': contact})
                generated = {'token_ids': [], 'policy_seconds': 0., 'prompt_tokens': 0,
                             'prompt_sha256': None, 'input_token_sha256': None}
                if not visual['applicable']:
                    generated['prediction'] = ''
                elif contact_closure:
                    generated['prediction'] = 'GRASP'
                elif trial['vision_arm'] == 'g0':
                    generated['prediction'] = geometry_prediction(req['inputs'])
                else:
                    generated = model.generate(method, item)
                token = generated['prediction'].strip(); depth_proof = None; depth_mask = None
                try:
                    parts = closure_recovery.parse_action(method, token)
                except ValueError as exc:
                    guard = {'version': entry_guard.VERSION, 'pass': False, 'reason': 'invalid_model_action', 'error': str(exc)}
                else:
                    previous = entry_guard.previous_motion(path) if method == 'C' and parts is not None else None
                    guard = entry_guard.action_admission(req, token, projector, p['visual_contract']['depth_scale'],
                                 lambda f: np.load(f, allow_pickle=False), read, previous)
                    if method == 'D' and parts is not None:
                        goal = entry_guard.residual_goal(req['inputs']['eef_world'], token)
                        depth_proof = entry_guard.depth_admission(row, depth, masks['target'], goal,
                                          p['visual_contract']['depth_scale']['near_m'])
                        depth_mask = path.with_name(req['request_id']+'_target_mask.npy')
                        np.save(depth_mask, masks['target'], allow_pickle=False)
                guard.update(request_id=req['request_id'], request_sha256=sha(path), prediction=token)
                verify_files(req['files'])
                reply = {**generated, 'ok': True, 'method': method, 'request_id': req['request_id'],
                         'request_sha256': sha(path), 'registration_sha256': req['registration_sha256'],
                         'vision_arm': trial['vision_arm'], 'visual_control': visual, 'contact_control': contact,
                         'controller_contact_closure': bool(contact_closure), 'model_inference': used,
                         'checkpoint': p['checkpoints'][method] if used else None, 'projection': projection,
                         'action_admission': guard, 'advisor31_depth': depth_proof,
                         'advisor31_depth_mask': str(depth_mask) if depth_mask is not None else None,
                         'total_service_seconds': time.perf_counter()-start,
                         'files': {str(f): sha(f) for f in set(actual+[prompt]+list(map(Path, visual['model_images'])))} }
                reply['files'].update(contact['files'])
                if depth_mask is not None: reply['files'][str(depth_mask)] = sha(depth_mask)
                verify_reply(reg, req, reply); dest = path.with_name(req['request_id']+'_reply.json')
                if dest.exists():
                    raise RuntimeError('No overwritten reply or automatic inference retry')
                write(dest, reply); write(done, {'request': str(path), 'reply_sha256': sha(dest)})
            time.sleep(.05)
        write(directory/'exit.json', {'pass': True, 'wall': time.time()})
    except BaseException:
        error = traceback.format_exc(); write(directory/'FAULT.json', {'error': error})
        exclusive_stop(root/'STOP', {'reason': 'advisor31_policy_failure', 'error': error}); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('campaign'); serve(parser.parse_args().campaign)
