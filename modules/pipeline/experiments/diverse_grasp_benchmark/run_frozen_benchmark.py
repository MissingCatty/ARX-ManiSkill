"""Execute all frozen geometry-only scenes without consulting post-hoc IK results."""
import argparse
import fcntl
from pathlib import Path
import traceback

import run_benchmark as baseline
from common import PROJECT, read, save, sha, now

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / 'runs/geometry200_fixed452_20261122_01'
original_report = baseline.report


def verify_dataset(dataset):
    frozen = read(dataset / 'frozen_dataset.json')
    verification = read(dataset / 'verification.json')
    manifest = read(dataset / 'manifest.json')
    assert frozen['construction_independent_of_ik'] and frozen['IK_evaluations_at_freeze'] == 0
    assert verification['passed'] and verification['construction_independent_of_ik']
    assert len(manifest['schedule']) == 200
    for name, digest in frozen['files'].items():
        assert sha(dataset / name) == digest, 'Frozen scene changed: ' + name
    return {'passed': True, 'at': now(), 'dataset': str(dataset), 'frozen_at': frozen['at'],
            'frozen_file_count': len(frozen['files']), 'freeze_sha256': sha(dataset / 'frozen_dataset.json'),
            'scene_ids': [entry['id'] for entry in manifest['schedule']],
            'construction_IK_queries': 0, 'diagnostic_results_used_for_selection': False,
            'physical_grasps': 0}


def prepare(root):
    config = read(PROJECT / 'configs/scene_dataset.json')
    assert config['grasp_tests_authorized'] and config['grasp_test_budget'] == 200
    assert config['construction_independent_of_ik']
    dataset = Path(config['dataset'])
    evidence = verify_dataset(dataset)
    baseline.prepare(root)
    reg = read(root / 'registration.json')
    assert [entry['id'] for entry in reg['schedule']] == evidence['scene_ids']
    reg.update(protocol='independent_geometry200_head_sam3_rayst3r_graspgenx_fixed452_v1',
               construction_independent_of_ik=True, dataset_freeze_sha256=evidence['freeze_sha256'],
               posthoc_IK_used_for_selection=False, registered_scene_count=200,
               all_frozen_scenes_retained=True,
               user_authorization='2026-11-22: scenes ready; start physical grasp testing of the complete frozen 200-scene dataset')
    for filename in ['frozen_dataset.json', 'sampling_region.json']:
        reg['source_hashes'][str(dataset / filename)] = sha(dataset / filename)
    save(root / 'registration.json', reg)
    save(root / 'preflight/frozen_dataset.json', evidence)
    baseline.aggregate(root, 'registered')
    render_report(root, read(root / 'progress.json'))


def render_report(root, summary):
    original_report(root, summary)
    doc = PROJECT / 'docs/diverse_grasp_benchmark/抓取测试结果.md'
    text = doc.read_text().replace('# 多类别200场景抓取测试', '# 独立采样200场景抓取测试', 1)
    text = text.replace('40类、80型号，五种布局各40例。',
        '使用生成后先冻结、构造阶段未调用IK的完整200个场景。40类、80型号，五种布局各40例。', 1)
    text += '\n所有200例均按登记顺序调度，事后静态IK诊断不参与试验选取。\n'
    doc.write_text(text)
    (root / 'report.md').write_text(text)


def run(root):
    reg = read(root / 'registration.json')
    assert reg['construction_independent_of_ik'] and reg['all_frozen_scenes_retained']
    assert not reg['posthoc_IK_used_for_selection']
    dataset = Path(reg['scene_dataset'])
    evidence = verify_dataset(dataset)
    assert evidence['freeze_sha256'] == reg['dataset_freeze_sha256']
    assert [entry['id'] for entry in reg['schedule']] == evidence['scene_ids']
    save(root / 'preflight/frozen_dataset_before_run.json', evidence)
    baseline.report = render_report
    try:
        baseline.run(root)
        save(root / 'frozen_dataset_after_run.json', verify_dataset(dataset))
    except BaseException:
        error = traceback.format_exc()
        marker = root / 'STOP'
        if not marker.exists():
            marker.write_text(error)
        if not (root / 'halt.json').exists():
            save(root / 'halt.json', {'at': now(), 'error': error})
        summary = baseline.aggregate(root, 'halted')
        render_report(root, summary)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, default=DEFAULT)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == 'prepare':
        prepare(root)
    else:
        with (root / 'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(root)
