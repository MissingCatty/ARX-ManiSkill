"""Exercise the registry worker and retained inference memory twice."""
import argparse
import os
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

from arx_project.layout import ROOT, now, read_json, sha256, write_json


def run(root, competitor_mib):
    import numpy as np
    import torch
    from arx_project.registry import create_worker
    if root.exists():
        raise RuntimeError('Use a fresh workspace interface registration')
    prior = ROOT/'tmp/rayst3r/memory_validation_20261125_01'
    registered = read_json(prior/'registration.json')
    source = Path(registered['stress_source'])
    reference = np.load(prior/'stress_replay/points.npy', allow_pickle=False)
    paths = [ROOT/'modules/rayst3r/memory_efficient/adapter.py',
             ROOT/'modules/rayst3r/memory_efficient/attention.py', Path(__file__),
             ROOT/'modules/rayst3r/arx_adapter.py', ROOT/'configs/modules.json']
    paths += [source/n for n in ['state.json', 'camera/color.png', 'camera/mask.png', 'camera/depth_m.npy']]
    hashes = {str(p): sha256(p) for p in paths}
    root.mkdir(parents=True)
    write_json(root/'registration.json', {'at': now(), 'source': str(source), 'files': hashes,
        'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'workspace_mib': 14336,
        'competitor_mib': competitor_mib, 'inference_calls': 2, 'physical_grasps': 0,
        'planning_calls': 0, 'model_weights': registered['weights']})
    for name, entry in registered['weights'].items():
        if sha256(name) != entry['sha256']:
            raise RuntimeError('Model weight changed')
    if torch.cuda.mem_get_info()[0]/2**20 < 23000:
        raise RuntimeError('Need an empty GPU for the bounded contention check')
    worker = create_worker('rayst3r', SimpleNamespace())
    torch.cuda.empty_cache()
    held = lambda: sum(t.numel()*t.element_size() for t in worker._workspace)/2**20
    if held() != 14336:
        raise RuntimeError('Warmup cache cleanup dropped the live workspace')
    competitor = torch.empty(competitor_mib*1024**2, dtype=torch.uint8, device='cuda')
    records = []
    for index in range(2):
        work = root/f'call_{index:02d}'; work.mkdir()
        shutil.copytree(source/'camera', work/'camera'); shutil.copy2(source/'state.json', work/'state.json')
        seed = registered['settings']['seed']
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        result = worker.request(work)
        points = np.load(work/'completed_points_camera.npy', allow_pickle=False)
        if not result['success'] or not np.array_equal(reference, points):
            raise RuntimeError('Registry worker output differs from validated stress replay')
        torch.cuda.empty_cache()
        record = {'call': index, 'point_count': len(points), 'bitwise_equal': True,
                  'workspace_still_held_mib': held(), 'allocated_mib': torch.cuda.memory_allocated()/2**20,
                  'remaining_mib': torch.cuda.mem_get_info()[0]/2**20}
        if record['workspace_still_held_mib'] != 14336 or record['remaining_mib'] < 1536:
            raise RuntimeError('Workspace ownership or headroom check failed: '+str(record))
        records.append(record)
        print('WORKSPACE_CALL_PASSED', record, flush=True)
    for name, expected in hashes.items():
        if sha256(name) != expected:
            raise RuntimeError('Interface source/input changed during verification')
    write_json(root/'acceptance.json', {'at': now(), 'passed': True, 'physical_grasps': 0,
        'planning_calls': 0, 'registry_worker_calls': 2, 'competitor_mib': competitor.numel()/2**20,
        'cache_cleanup_preserves_live_workspace': True, 'outputs_bitwise_equal': True,
        'source_input_weight_integrity': True, 'records': records})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--competitor-mib', type=int, default=0, choices=[0, 8192])
    args = parser.parse_args(); directory = args.root.resolve()
    try:
        run(directory, args.competitor_mib)
    except BaseException:
        if directory.exists():
            write_json(directory/'STOP', {'at': now(), 'error': __import__('traceback').format_exc()})
        raise
