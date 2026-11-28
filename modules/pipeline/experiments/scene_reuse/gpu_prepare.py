"""Register the second-stage pilot after the first stage has fully stopped."""
import argparse
from pathlib import Path
import shutil

from arx_project.layout import ROOT, ensure_run_path, now, read_json, sha256, write_json
from .prepare import prepare as base_prepare

CONFIG = ROOT/'configs/experiments/scene_reuse_gpu.json'


def prepare(root):
    cfg = read_json(CONFIG); previous = ROOT/'tmp/state_updates/scene_reuse_cpu5_20261128_01'
    prior = read_json(previous/'acceptance.json')
    if (not prior['record_integrity'] or not prior['exact_selection_and_paths']
            or not read_json(previous/'services_stopped.json')['all_owned_exited']):
        raise RuntimeError('Stage one is not independently complete and stopped')
    if prior['optional_implementation_accepted']:
        raise RuntimeError('This pilot was specified against the non-adopted CPU-cache baseline')
    base_prepare(root); reg = read_json(root/'registration.json')
    reg.update(config=cfg, experiment='V8_shared_native_GPU_geometry',
               preceding_CPU_cache_adopted=False)
    for name in ['registration.json', 'acceptance.json', 'analysis.json', 'independent_review.json', 'services_stopped.json']:
        p = previous/name; reg['input_sha256'][str(p)] = sha256(p)
    p = CONFIG; dest = root/'source_registered'/p.relative_to(ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p,dest)
    reg['source_sha256'][str(p)] = sha256(p)
    write_json(root/'registration.json',reg)
    write_json(ROOT/'tmp/state_updates/scene_reuse_active.json', {'at': now(),
        'run': str(root.relative_to(ROOT)), 'stage': 'shared_native_GPU_geometry', 'phase': 'registered'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    prepare(ensure_run_path(parser.parse_args().root))
