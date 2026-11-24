"""Explicit current-host GPU placement for ARX jobs; importing this module uses no CUDA."""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import struct
import time
from arx_project.layout import ROOT as PROJECT_ROOT

ROOT = Path(__file__).resolve().parents[2]
# Defaults for coolas-8; all inventoried devices are authorized by the user.
POLICY_GPU = 'GPU-3164d5f3-6a9a-17b4-53f9-a70e3a82b834'
COMPUTE_GPU = 'GPU-d9e4f8a1-c4f3-232a-4ceb-a9dde26aa45b'


def inventory():
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,name,compute_cap', '--format=csv,noheader'],
        text=True, timeout=15)
    return [{'index': int(row[0].strip()), 'uuid': row[1].strip(),
             'name': row[2].strip(), 'compute_cap': row[3].strip()}
            for row in csv.reader(output.splitlines())]


def resolve(value, devices=None):
    """Resolve one current nvidia-smi index or UUID; never silently substitute."""
    token = str(value).strip()
    if not token or ',' in token:
        raise ValueError('Choose exactly one current GPU index or full UUID')
    for device in inventory() if devices is None else devices:
        if (token.isdigit() and int(token) == device['index']) or token == device['uuid']:
            return {**device, 'visible_index': device['index']}
    raise ValueError('Requested GPU is not visible: ' + token)


def compute_env(environment, value):
    """Normalize visibility before Torch imports and separate the sm89 cache."""
    result = dict(environment)
    gpu = resolve(value)
    arch = gpu['compute_cap']
    result.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu['uuid'],
                  TORCH_CUDA_ARCH_LIST=arch,
                  TORCH_EXTENSIONS_DIR=str(PROJECT_ROOT / 'tmp/curobo' / ('torch_extensions_sm' + arch.replace('.', ''))))
    return result


def checkpoint_minimum_bytes(checkpoint):
    """Read safetensors headers on CPU; ignore head/activations for a lower bound."""
    checkpoint = Path(checkpoint)
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())
    total = 0
    for name in sorted(set(index['weight_map'].values())):
        with (checkpoint / name).open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            if length > 100_000_000:
                raise ValueError('Invalid safetensors header length')
            header = json.loads(stream.read(length))
        for key, tensor in header.items():
            if key == '__metadata__':continue
            if tensor['dtype'] in ('BF16', 'F16', 'F32', 'F64'):
                total += math.prod(tensor['shape']) * 2
    if total <= 0:raise ValueError('No floating model weights found')
    return total


def memory_preflight(checkpoint, value):
    gpu = resolve(value)
    free = int(subprocess.check_output([
        'nvidia-smi', '--id=' + gpu['uuid'], '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=15).strip()) * 1024 ** 2
    minimum = checkpoint_minimum_bytes(checkpoint)
    return {'gpu_uuid': gpu['uuid'], 'physical_gpu_index': gpu['index'],
            'visible_gpu_index': gpu['visible_index'], 'free_bytes': free,
            'bf16_model_minimum_bytes': minimum, 'weights_fit_lower_bound': free >= minimum,
            'additional_head_and_activation_memory_required': True,
            'scope': 'Necessary lower bound, not a guarantee that inference fits'}


def verify_process(pid, expected):
    """Query once at startup/boundary, never for every predicted action."""
    gpu = resolve(expected)
    for attempt in range(5):
        output = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
            text=True, timeout=15)
        actual = {row[0].strip() for row in csv.reader(output.splitlines())
                  if len(row) == 2 and row[1].strip() == str(pid)}
        if actual:
            if actual != {gpu['uuid']}:
                raise ValueError(f'PID {pid} GPU placement differs: {sorted(actual)}')
            return {'gpu_uuid': gpu['uuid'], 'physical_gpu_index': gpu['index'],
                    'visible_gpu_index': gpu['visible_index'],
                    'gpu_process_pid': int(pid), 'gpu_binding_verified': True}
        if attempt < 4:
            time.sleep(.2)
    raise ValueError(f'PID {pid} has no verifiable CUDA allocation on the selected GPU')


def verify_health(health, expected=None):
    actual = resolve(health.get('gpu_uuid', ''))
    if expected is not None and actual['uuid'] != resolve(expected)['uuid']:
        raise ValueError('Policy health GPU differs from the requested physical GPU')
    if health.get('gpu_binding_verified') is not True:
        raise ValueError('Policy service did not verify its CUDA allocation')
    return verify_process(health['gpu_process_pid'], actual['uuid'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('resolve', 'shell'))
    parser.add_argument('values', nargs='+')
    args = parser.parse_args()
    devices = inventory()
    selected = [resolve(value, devices) for value in args.values]
    if args.mode == 'resolve':
        if len(selected) != 1:
            parser.error('resolve expects one physical GPU')
        print(selected[0]['uuid'])
    else:
        if len(selected) != 4:
            parser.error('shell expects policy, perception, planner and simulator GPUs')
        print(' '.join([d['uuid'] for d in selected[:3]] + [str(selected[3]['index'])]))


if __name__ == '__main__':
    main()
