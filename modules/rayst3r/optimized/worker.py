"""Same-GPU, interleaved four-way comparison of 50 saved observations."""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import json
import os
from pathlib import Path
import random
import statistics
import time
import traceback
from unittest.mock import patch

from arx_project.layout import ROOT, now, read_json, sha256, write_json
from .benchmark import dependency_ready, gpu_inventory, guard


def load_input(registration, case_id):
    import numpy as np
    from PIL import Image
    source = Path(registration['reference'])/'observations'/case_id
    camera = read_json(source/'camera.json')
    return (np.asarray(Image.open(source/'rgb.png').convert('RGB')),
            np.load(source/'depth_m.npy', allow_pickle=False),
            np.asarray(Image.open(Path(registration['reference'])/registration['segmentation']/case_id/'mask.png')) > 0,
            np.asarray(camera['intrinsics']))


def compare_points(reference, candidate, tolerance):
    import numpy as np
    same_shape = reference.shape == candidate.shape
    finite = bool(np.isfinite(candidate).all())
    difference = float(np.max(np.abs(reference-candidate))) if same_shape and reference.size else 0. if same_shape else None
    return {'passed': same_shape and finite and difference <= tolerance,
            'same_point_count': same_shape, 'finite': finite,
            'bitwise_equal': same_shape and bool(np.array_equal(reference, candidate)),
            'maximum_coordinate_difference_m': difference,
            'reference_points': len(reference), 'candidate_points': len(candidate)}


def summarize(root):
    reg = read_json(root/'registration.json')
    rows = [read_json(p) for p in sorted((root/'trials').glob('*/result.json'))]
    variants = reg['settings']['variants']
    timing, peak, paired = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        baseline = statistics.median(row['timings_seconds']['baseline'])
        for name in variants:
            timing[name].extend(row['timings_seconds'][name])
            peak[name].extend(row['peak_allocated_mib'][name])
            paired[name].append(baseline/statistics.median(row['timings_seconds'][name]))
    report = {'at': now(), 'completed_cases': len(rows), 'registered_cases': 50,
              'comparisons_passed': sum(r['passed'] for r in rows), 'physical_trials': 0,
              'automatic_activation': False, 'variants': {}}
    for name, values in timing.items():
        report['variants'][name] = {'measurements': len(values), 'median_seconds': statistics.median(values),
            'p95_seconds': sorted(values)[max(0, int(len(values)*.95)-1)],
            'median_paired_speedup': statistics.median(paired[name]),
            'peak_allocated_mib': max(peak[name])}
    write_json(root/'summary.json', report)
    write_json(root/'progress.json', {'at': now(), 'phase': 'running',
        'completed_cases': len(rows), 'registered_cases': 50,
        'comparisons_passed': report['comparisons_passed'], 'physical_trials': 0})
    return report


def verify_records(root, reg):
    """Re-read saved arrays and require every registered measurement."""
    import numpy as np
    issues, record_hashes = [], {}
    for case in reg['cases']:
        directory = root/'trials'/case['id']
        result = directory/'result.json'
        if not result.is_file():
            issues.append(case['id']+': missing result')
            continue
        row = read_json(result)
        record_hashes[str(result.relative_to(root))] = sha256(result)
        baseline_path = directory/'baseline/points.npy'
        if not baseline_path.is_file():
            issues.append(case['id']+': missing baseline points')
            continue
        reference = np.load(baseline_path, allow_pickle=False)
        expected = len(reg['settings']['variants'])*reg['settings']['repetitions']
        if len(row.get('comparisons', [])) != expected or len(row.get('traces', [])) != expected:
            issues.append(case['id']+': missing comparison or compute trace')
        for name in reg['settings']['variants']:
            times = row.get('timings_seconds', {}).get(name, [])
            if len(times) != reg['settings']['repetitions'] or not all(np.isfinite(t) and t > 0 for t in times):
                issues.append(case['id']+': missing or invalid timing '+name)
            path = directory/name/'points.npy'
            if not path.is_file():
                issues.append(case['id']+': missing saved point output '+name)
                continue
            points = np.load(path, allow_pickle=False)
            if not compare_points(reference, points, reg['settings']['coordinate_atol_m'])['passed']:
                issues.append(case['id']+': saved point mismatch '+name)
            record_hashes[str(path.relative_to(root))] = sha256(path)
        if not all(c.get('passed', False) for c in row.get('comparisons', [])):
            issues.append(case['id']+': repeat output mismatch')
    return {'passed': not issues, 'issues': issues, 'record_hashes': record_hashes}


def write_report(root, acceptance, summary):
    labels = {'baseline': '原实现', 'cache_only': '只复用DINO特征',
              'prediction_only': '只去掉损失和指标', 'combined': '两项一起'}
    lines = ['# RaySt3R计算优化结果', '',
        f"完成{acceptance['completed_cases']}/50组已有观测的四路对照，物理抓取0次。", '',
        '| 方法 | 中位耗时 | P95耗时 | 配对加速比 | GPU峰值分配 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name, row in summary['variants'].items():
        lines.append(f"| {labels[name]} | {row['median_seconds']:.3f}秒 | {row['p95_seconds']:.3f}秒 | "
                     f"{row['median_paired_speedup']:.2f}倍 | {row['peak_allocated_mib']/1024:.2f}GiB |")
    lines += ['', f"全部点云对齐检查{'通过' if acceptance['outputs_equivalent'] else '未通过'}；"
        f"同点数、逐点坐标容差为{acceptance['coordinate_atol_m']*1e6:.1f}微米。",
        '', f"下一次登记可用：{'是' if acceptance['eligible_for_next_registration'] else '否'}。"
        "当前主线配置保持原值。", '', f"原始记录：{root.relative_to(ROOT)}。", '']
    (ROOT/'modules/pipeline/docs/pointcloud_benchmark/RaySt3R计算优化结果.md').write_text('\n'.join(lines))


def run(root):
    import numpy as np
    import torch
    reg = guard(root, include_payload=True)
    ready, reason = dependency_ready(Path(reg['protected_campaign']))
    if not ready:
        raise RuntimeError('Protected campaign is still using the GPUs: '+reason)
    if (root/'started.json').exists() or list((root/'trials').iterdir()):
        raise RuntimeError('Do not repeat an already started comparison')
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    if gpu not in reg['allowed_gpu_uuids'] or gpu_inventory().get(gpu, 0) < reg['settings']['minimum_gpu_free_mib']:
        raise RuntimeError('Selected GPU no longer meets registered resource admission')
    torch.set_num_threads(6)
    write_json(root/'started.json', {'at': now(), 'pid': os.getpid(), 'gpu': gpu})
    from modules.rayst3r.arx_reconstruction import RaySt3RAdapter
    from modules.rayst3r.optimized.adapter import OptimizedRaySt3RAdapter
    started = time.perf_counter()
    baseline = RaySt3RAdapter()
    adapters = {'baseline': baseline, **{v: OptimizedRaySt3RAdapter(baseline=baseline, variant=v)
                for v in reg['settings']['variants'] if v != 'baseline'}}
    import engine
    torch.cuda.synchronize()
    load_seconds = time.perf_counter()-started
    warmup = load_input(reg, 'warmup')
    for name, adapter in adapters.items():
        torch.manual_seed(reg['settings']['seed']); np.random.seed(reg['settings']['seed'])
        adapter.predict(*warmup, root/'preflight/warmup'/name)
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0]/1024**2 < reg['settings']['minimum_post_load_free_mib']:
        raise RuntimeError('Insufficient GPU memory after common warmup')
    write_json(root/'preflight/all_ready.json', {'at': now(), 'passed': True, 'gpu': gpu,
        'model_load_count': 1, 'shared_weights': True, 'load_seconds': load_seconds,
        'torch': torch.__version__, 'free_mib': torch.cuda.mem_get_info()[0]/1024**2})
    variants = reg['settings']['variants']
    expected_trace = {'baseline': (2, 2, 2), 'cache_only': (1, 2, 2),
                      'prediction_only': (2, 0, 0), 'combined': (1, 0, 0)}
    for number, case in enumerate(reg['cases']):
        guard(root)
        directory = root/'trials'/case['id']; directory.mkdir()
        write_json(directory/'started.json', {'at': now(), 'case': case['id'], 'number': number+1})
        inputs = load_input(reg, case['id'])
        timings, peaks, comparisons, traces = defaultdict(list), defaultdict(list), [], []
        reference = None
        for repetition in range(reg['settings']['repetitions']):
            offset = (number+repetition) % len(variants)
            order = variants[offset:]+variants[:offset]
            outputs = {}
            for name in order:
                adapter = adapters[name]
                random.seed(reg['settings']['seed']); np.random.seed(reg['settings']['seed'])
                torch.manual_seed(reg['settings']['seed'])
                counts = {'dino_forwards': 0, 'loss_calls': 0, 'metric_calls': 0}
                def dino_hook(*_):
                    counts['dino_forwards'] += 1
                def loss_hook(*_):
                    counts['loss_calls'] += 1
                native_metric = engine.eval_pred
                def counted_metric(*args, **kw):
                    counts['metric_calls'] += 1
                    return native_metric(*args, **kw)
                with ExitStack() as stack:
                    for handle in [baseline.dino.patch_embed.register_forward_pre_hook(dino_hook),
                                   baseline.model.model.criterion.register_forward_pre_hook(loss_hook)]:
                        stack.callback(handle.remove)
                    if name == 'baseline':
                        stack.enter_context(patch.object(engine, 'eval_pred', counted_metric))
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    began = time.perf_counter()
                    output = adapter.predict(*inputs, directory/name/'native_input')
                    torch.cuda.synchronize()
                    seconds = time.perf_counter()-began
                points = np.asarray(output['points'], dtype=np.float32)
                if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                    raise RuntimeError('Invalid reconstruction output')
                timings[name].append(seconds)
                peaks[name].append(torch.cuda.max_memory_allocated()/1024**2)
                if name != 'baseline':
                    counts['metric_calls'] = adapter.last_trace['metric_calls']
                    counts.update(feature_cache_hits=adapter.last_trace['feature_cache_hits'],
                                  network_forwards=adapter.last_trace['network_forwards'])
                    wanted_hits = 1 if name in ['cache_only', 'combined'] else 0
                    if counts['network_forwards'] != 2 or counts['feature_cache_hits'] != wanted_hits:
                        raise RuntimeError('Two-stage inference or feature reuse differs from registration')
                observed = (counts['dino_forwards'], counts['loss_calls'], counts['metric_calls'])
                if observed != expected_trace[name]:
                    raise RuntimeError(f'Unexpected actual compute calls for {name}: {counts}')
                traces.append({'variant': name, 'repetition': repetition, 'seconds': seconds, **counts})
                outputs[name] = points
                if repetition == 0:
                    np.save(directory/name/'points.npy', points, allow_pickle=False)
            if reference is None:
                reference = outputs['baseline']
            for name, points in outputs.items():
                check = compare_points(reference, points, reg['settings']['coordinate_atol_m'])
                comparisons.append({'variant': name, 'repetition': repetition, **check})
                if not check['passed']:
                    write_json(directory/'mismatch.json', comparisons[-1])
                    raise RuntimeError('Point output changed beyond registered tolerance')
        result = {'id': case['id'], 'passed': all(c['passed'] for c in comparisons),
            'timings_seconds': dict(timings), 'peak_allocated_mib': dict(peaks),
            'comparisons': comparisons, 'traces': traces}
        write_json(directory/'result.json', result)
        summarize(root)
        print('COMPUTE_COMPARISON', number+1, case['id'],
              {v: round(statistics.median(t), 4) for v, t in timings.items()}, flush=True)
    guard(root, include_payload=True)
    summary = summarize(root)
    records = verify_records(root, reg)
    write_json(root/'record_verification.json', records)
    equivalent = summary['completed_cases'] == 50 and summary['comparisons_passed'] == 50 and records['passed']
    speedup = summary['variants']['combined']['median_paired_speedup']
    acceptance = {'at': now(), 'completed_cases': summary['completed_cases'],
        'outputs_equivalent': equivalent, 'coordinate_atol_m': reg['settings']['coordinate_atol_m'],
        'combined_median_paired_speedup': speedup, 'eligible_for_next_registration': equivalent and speedup > 1,
        'same_gpu_and_common_warmup': True, 'physical_trials': 0, 'automatic_activation': False,
        'source_input_weight_integrity': True, 'record_integrity': records['passed'],
        'issues': records['issues']}
    write_json(root/'acceptance.json', acceptance)
    write_report(root, acceptance, summary)
    write_json(root/'progress.json', {'at': now(), 'phase': 'complete',
        'completed_cases': 50, 'comparisons_passed': summary['comparisons_passed'], 'physical_trials': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        run(root)
    except BaseException:
        write_json(root/'STOP', {'at': now(), 'error': traceback.format_exc(), 'automatic_retry': False})
        previous = read_json(root/'progress.json')
        write_json(root/'progress.json', {**previous, 'at': now(), 'phase': 'stopped'})
        raise


if __name__ == '__main__':
    main()
