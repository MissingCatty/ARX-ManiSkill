"""S02 evidence only: content identities and clocks, with no motion/model setters."""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np
from PIL import Image

VERSION = 'arx_policy_trace_s02_v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def stamp():
    # Cross-process subtraction is valid only when clock_domain matches.
    return dict(monotonic_ns=time.monotonic_ns(), wall_time_ns=time.time_ns(),
                clock_domain=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                + ':' + os.readlink('/proc/self/ns/time'))


def identity():
    return uuid.uuid4().hex


def array_identity(value):
    a = np.ascontiguousarray(value)
    return dict(sha256=sha(a.tobytes()), shape=list(a.shape), dtype=str(a.dtype))


def rgb_identity(image):
    return array_identity(np.asarray(image.convert('RGB'), dtype=np.uint8))


def tensor_identity(value):
    # Hash the actual forward argument, including bf16 bits, not a float32 surrogate.
    a = value.detach().contiguous().cpu()
    return dict(sha256=sha(a.view(__import__('torch').uint8).numpy().tobytes()),
                shape=list(a.shape), dtype=str(a.dtype))


def request_identity(png, state, instruction, convention):
    value = dict(png_sha256=sha(png), state=state, instruction=instruction,
                 state_convention=convention)
    return sha(canonical(value))


def make_request(observation, png, state, instruction, convention):
    capture = observation.get('trace')
    if not capture or capture.get('version') != VERSION:
        raise ValueError('S02 needs a simulator capture trace; historical paths are insufficient')
    original = Path(observation['rgb_path']).read_bytes()
    if sha(original) != capture['png_sha256']:
        raise ValueError('Captured observation PNG changed')
    with Image.open(io.BytesIO(original)) as image:
        if rgb_identity(image) != capture['rgb']:
            raise ValueError('Captured observation pixels changed')
    if sha(canonical(observation['snapshot'])) != capture['snapshot_sha256']:
        raise ValueError('Captured observation state changed')
    velocities = observation['snapshot'].get('joint_velocities')
    if not velocities or len(velocities) != len(observation['snapshot']['joint_names']) or not all(map(math.isfinite, velocities)):
        raise ValueError('S02 requires measured finite joint velocities')
    with Image.open(io.BytesIO(png)) as image:
        rgb = rgb_identity(image)
    return dict(version=VERSION, observation_id=capture['observation_id'], request_id=identity(),
                capture=capture, request_png_sha256=sha(png), request_rgb=rgb,
                request_sha256=request_identity(png, state, instruction, convention),
                client_send=stamp())


def verify_request(trace, png, image, state, instruction, convention):
    if trace.get('version') != VERSION or not trace.get('observation_id') or not trace.get('request_id'):
        raise ValueError('Missing S02 trace identity')
    if trace['observation_id'] != trace['capture']['observation_id']:
        raise ValueError('Observation ID mismatch')
    if trace['request_png_sha256'] != sha(png) or trace['request_rgb'] != rgb_identity(image):
        raise ValueError('Request image checksum mismatch')
    if trace['request_sha256'] != request_identity(png, state, instruction, convention):
        raise ValueError('Request state/instruction/content checksum mismatch')


def verify_response(request, response):
    for key in ('version', 'observation_id', 'request_id', 'request_sha256', 'request_png_sha256', 'request_rgb', 'capture'):
        if response.get(key) != request[key]:
            raise ValueError('Policy response trace mismatch: '+key)
    for key in ('server_received', 'preprocess_start', 'forward_start', 'forward_end', 'server_finished',
                'preprocessed_rgb', 'model_inputs', 'trace_overhead_seconds'):
        if key not in response:
            raise ValueError('Incomplete model trace: '+key)
    if 'pixel_values' not in response['model_inputs']:
        raise ValueError('Actual forward pixel tensor was not recorded')


def elapsed_seconds(start, end):
    if start['clock_domain'] != end['clock_domain']:
        return None
    delta = (end['monotonic_ns']-start['monotonic_ns']) / 1e9
    if delta < 0:
        raise ValueError('Trace clocks are out of order')
    return delta
