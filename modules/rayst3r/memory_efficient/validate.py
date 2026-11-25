"""Compare all 50 frozen observations and replay the OOM-causing input."""
import argparse
import os
from pathlib import Path
import random
import shutil
import time
import traceback

from arx_project.layout import ROOT, now, read_json, sha256, write_json


CONFIG = ROOT/'configs/experiments/rayst3r_memory_validation.json'


def prepare(root):
    if root.exists():
        raise ValueError('Use a fresh memory-validation registration')
    settings = read_json(CONFIG)
    previous = read_json(ROOT/settings['reference_validation']/'registration.json')
    inputs = dict(previous['input_hashes'])
    stress = ROOT/settings['stress_source']
    inputs.update({str(stress/name): sha256(stress/name) for name in
                   ['camera/color.png', 'camera/depth_m.npy', 'camera/mask.png', 'state.json']})
    sources = {str(CONFIG): sha256(CONFIG), **previous['source_hashes']}
    for p in (ROOT/'modules/rayst3r/memory_efficient').glob('*.py'):
        sources[str(p)] = sha256(p)
    for name, digest in {**inputs, **sources}.items():
        if sha256(name) != digest:
            raise RuntimeError('Validation source or input changed: '+name)
    assert len(previous['cases']) == 50
    root.mkdir(parents=True)
    for folder in ['source_registered', 'trials', 'preflight']:
        (root/folder).mkdir()
    for name in sources:
        dest = root/'source_registered'/Path(name).relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, dest)
    write_json(root/'registration.json', {'at': now(), 'settings': settings,
        'cases': previous['cases'], 'reference': previous['reference'],
        'segmentation': previous['segmentation'], 'stress_source': str(stress),
        'input_hashes': inputs, 'source_hashes': sources, 'weights': previous['weights'],
        'measurement_budget': 101, 'common_warmup_calls': 2,
        'physical_grasps': 0, 'planning_calls': 0, 'automatic_retries': False})
    write_json(root/'progress.json', {'phase': 'registered', 'completed': 0, 'registered': 50})


def guard(root):
    reg = read_json(root/'registration.json')
    if (root/'STOP').exists():
        raise RuntimeError('STOP exists')
    for name, digest in {**reg['source_hashes'], **reg['input_hashes']}.items():
        if sha256(name) != digest:
            raise RuntimeError('Validation input or source changed: '+name)
    for name, entry in reg['weights'].items():
        if sha256(name) != entry['sha256']:
            raise RuntimeError('Validation weight changed: '+name)
    return reg


def run(root):
    import numpy as np
    import torch
    from PIL import Image
    from modules.rayst3r.optimized.worker import load_input, compare_points
    from modules.rayst3r.arx_reconstruction import RaySt3RAdapter
    from .adapter import MemoryEfficientAdapter
    reg = guard(root)
    if (root/'started.json').exists():
        raise RuntimeError('Validation already started')
    if torch.cuda.mem_get_info()[0]/2**20 < reg['settings']['minimum_gpu_free_mib']:
        raise RuntimeError('Insufficient free GPU memory for the registered comparison')
    torch.set_num_threads(6)
    write_json(root/'started.json', {'at': now(), 'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'pid': os.getpid()})
    native = RaySt3RAdapter()
    efficient = MemoryEfficientAdapter(baseline=native)
    adapters = {'baseline': native, 'inplace': efficient}
    def predict(adapter, values, directory):
        torch.manual_seed(reg['settings']['seed']); np.random.seed(reg['settings']['seed'])
        random.seed(reg['settings']['seed'])
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        points = adapter.predict(*values, directory)['points']
        torch.cuda.synchronize()
        stats = {'seconds': time.perf_counter()-started,
                 'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                 'remaining_mib': torch.cuda.mem_get_info()[0]/2**20}
        torch.cuda.empty_cache()
        return np.asarray(points).copy(), stats
    warm = load_input(reg, reg['cases'][0]['id'])
    for name, adapter in adapters.items():
        predict(adapter, warm, root/'preflight'/name)
    for index, case in enumerate(reg['cases']):
        values = load_input(reg, case['id'])
        work = root/'trials'/case['id']
        outputs, measurements = {}, {}
        for name in (['baseline', 'inplace'] if index%2 == 0 else ['inplace', 'baseline']):
            outputs[name], measurements[name] = predict(adapters[name], values, work/name)
            np.save(work/name/'points.npy', outputs[name], allow_pickle=False)
        comparison = compare_points(outputs['baseline'], outputs['inplace'], reg['settings']['coordinate_atol_m'])
        write_json(work/'result.json', {'id': case['id'], **comparison, 'measurements': measurements})
        if not comparison['passed']:
            raise RuntimeError('Point outputs differ: '+case['id']+' '+str(comparison))
        write_json(root/'progress.json', {'phase': 'running', 'completed': index+1,
                                        'registered': 50, 'physical_grasps': 0})
        print('MEMORY_CASE', index+1, case['id'], comparison['bitwise_equal'], measurements, flush=True)
    source = Path(reg['stress_source'])
    stress_input = (np.asarray(Image.open(source/'camera/color.png').convert('RGB')),
        np.load(source/'camera/depth_m.npy', allow_pickle=False),
        np.asarray(Image.open(source/'camera/mask.png')) > 0,
        np.asarray(read_json(source/'state.json')['intrinsics']))
    points, stress = predict(efficient, stress_input, root/'stress_replay')
    np.save(root/'stress_replay/points.npy', points, allow_pickle=False)
    valid = len(points) >= 20 and np.isfinite(points).all()
    if not valid or stress['peak_allocated_mib'] > reg['settings']['maximum_stress_allocated_mib']:
        raise RuntimeError('OOM-input replay failed the registered output/memory threshold: '+str(stress))
    write_json(root/'stress_replay/result.json', {'passed': True, 'point_count': len(points),
        **stress, 'baseline_replay': 'not_executed; original OOM retained in source preflight STOP'})
    issues, peak, bitwise = [], {'baseline': 0., 'inplace': 0.}, 0
    for case in reg['cases']:
        folder = root/'trials'/case['id']; row = read_json(folder/'result.json')
        a, b = [np.load(folder/name/'points.npy', allow_pickle=False) for name in ['baseline', 'inplace']]
        if not compare_points(a, b, reg['settings']['coordinate_atol_m'])['passed']:
            issues.append(case['id'])
        bitwise += int(np.array_equal(a, b))
        for name in peak:
            peak[name] = max(peak[name], row['measurements'][name]['peak_allocated_mib'])
    guard(root)
    report = {'at': now(), 'passed': not issues, 'completed': 50,
        'record_integrity': not issues, 'outputs_equivalent': not issues, 'bitwise_equal_cases': bitwise,
        'coordinate_atol_m': reg['settings']['coordinate_atol_m'], 'peak_allocated_mib': peak,
        'stress_replay': stress, 'stress_output_points': len(points),
        'same_gpu_and_common_warmup': True, 'source_input_weight_integrity': True,
        'physical_grasps': 0, 'planning_calls': 0, 'issues': issues,
        'eligible_for_next_registration': not issues}
    write_json(root/'acceptance.json', report)
    write_json(root/'progress.json', {'phase': 'complete', 'completed': 50, 'registered': 50})
    print('MEMORY_VALIDATION_PASSED', report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve()
    if args.stage == 'prepare':
        prepare(root)
    else:
        try:
            run(root)
        except BaseException:
            write_json(root/'STOP', {'at': now(), 'error': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
