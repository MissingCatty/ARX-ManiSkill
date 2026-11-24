"""Run uniform static IK after scene freezing, retaining every success and failure."""
from arx_project.assets import load_object_catalog
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import queue
import subprocess
import threading
import traceback

import numpy as np

from asset_catalog import HERE, PROJECT, MS, read, save, sha
from workspace_geometry import world_corners


def assert_frozen(root):
    record = read(root / 'frozen_dataset.json')
    assert record['scenes'] == 200 and record['construction_independent_of_ik']
    assert record['IK_evaluations_at_freeze'] == 0
    for name, digest in record['files'].items():
        assert sha(root / name) == digest, name
    return record


def main(root):
    frozen = assert_frozen(root)
    verification = read(root / 'verification.json')
    assert verification['passed'] and verification['construction_independent_of_ik']
    rows = read(root / 'manifest.json')['schedule']
    catalog = load_object_catalog(root / 'object_catalog.json')['models']
    out = root / 'diagnostics/preparation_position_ik_01'
    if out.exists():
        raise RuntimeError('Diagnostic output already exists; preserve it, do not overwrite or retry silently')
    out.mkdir(parents=True)
    (out / 'records').mkdir()
    helper = HERE / 'workspace_ik_position.py'
    reference = PROJECT.parent / 'arx_core/scripts/curobo_arx/static_pick/reference'
    sources = [Path(__file__), helper, MS / 'arx_pipeline/planner.py']
    sources += [reference / model / filename for model in ['model', 'model_closed']
                for filename in ['arx.yml', 'arx.urdf']]
    registration = {'at': datetime.now().astimezone().isoformat(), 'dataset_frozen_at': frozen['at'],
                    'dataset_freeze_sha256': sha(root / 'frozen_dataset.json'),
                    'source_hashes': {str(p): sha(p) for p in sources},
                    'registered': 200, 'scene_ids': [e['id'] for e in rows],
                    'criterion': 'Uniform position-only IK for EEF top+8cm, top+16cm, top+11cm; joint limits and original table/ground collision; object meshes excluded',
                    'orientation': 'chosen by the solver, same criterion for all 200 scenes',
                    'selection_or_repositioning_allowed': False, 'physical_grasps': 0}
    save(out / 'registration.json', registration)
    responses = queue.Queue()
    stdout = (out / 'stdout.log').open('w')
    stderr = (out / 'stderr.log').open('w')
    command = [str(PROJECT.parent / 'curobo/.venv/bin/python'), str(helper),
               str(root / 'preflight/head_camera.json'), str(out / 'model')]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=stderr, text=True, bufsize=1)
    def receive():
        try:
            for line in process.stdout:
                stdout.write(line)
                stdout.flush()
                if line.startswith('WORKSPACE_IK '):
                    responses.put(json.loads(line[len('WORKSPACE_IK '):]))
        finally:
            responses.put({'error': 'Static IK process exited'})
    reader = threading.Thread(target=receive, daemon=True)
    reader.start()
    records = []
    try:
        ready = responses.get(timeout=240)
        if not ready.get('ready'):
            raise RuntimeError(str(ready))
        save(out / 'ready.json', ready)
        for entry in rows:
            target = next(a for a in entry['scene']['actors'] if a['name'] == 'target')
            points = world_corners(target, catalog)
            request = {'id': entry['id'], 'center_world_m': ((points.min(0)+points.max(0))/2).tolist(),
                       'object_top_m': float(points[:, 2].max())}
            process.stdin.write(json.dumps(request) + '\n')
            process.stdin.flush()
            reply = responses.get(timeout=120)
            if reply.get('id') != entry['id'] or 'error' in reply:
                raise RuntimeError('Static IK response integrity failure: ' + str(reply))
            record = {'id': entry['id'], 'layout': entry['layout'], 'passed': bool(reply['passed']),
                      'result': 'solution_found' if reply['passed'] else 'no_solution_found',
                      'request': request, 'response': reply}
            save(out / 'records' / (entry['id'] + '.json'), record)
            records.append(record)
            counts = Counter(r['result'] for r in records)
            save(out / 'progress.json', {'phase': 'diagnosing_frozen_scenes', 'registered': 200,
                 'completed': len(records), 'solution_found': counts['solution_found'],
                 'no_solution_found': counts['no_solution_found'], 'not_executed': 200-len(records),
                 'scenes_removed': 0, 'scenes_repositioned': 0, 'physical_grasps': 0})
            if len(records) % 20 == 0:
                print('POST_FREEZE_IK', len(records), '/200', dict(counts), flush=True)
        assert_frozen(root)
        for path, digest in registration['source_hashes'].items():
            assert sha(path) == digest, path
        by_layout = {layout: {'registered': 40,
                             'solution_found': sum(r['passed'] for r in records if r['layout'] == layout),
                             'no_solution_found': sum(not r['passed'] for r in records if r['layout'] == layout)}
                     for layout in read(root / 'registration.json')['layouts']}
        summary = {'phase': 'complete', 'registered': 200, 'completed': 200,
                   'solution_found': sum(r['passed'] for r in records),
                   'no_solution_found': sum(not r['passed'] for r in records),
                   'not_executed': 0, 'by_layout': by_layout,
                   'frozen_files_unchanged': True, 'frozen_file_count': len(frozen['files']),
                   'scenes_removed': 0, 'scenes_repositioned': 0, 'physical_grasps': 0}
        save(out / 'summary.json', summary)
        save(out / 'progress.json', summary)
        report = PROJECT / 'docs/diverse_object_scenes/独立采样场景结果.md'
        lines = ['', '## 冻结后的静态IK诊断', '',
                 '全部200例冻结后，统一采用位置IK检查目标上方的三个准备位置，朝向由求解器选择，同时检查关节限位和桌面碰撞。', '',
                 '| 布局 | 场景数 | 解出可行解 | 未解出可行解 |', '| --- | ---: | ---: | ---: |']
        labels = ['开阔桌面', '靠近桌边', '稀疏混放', '中度杂乱', '部分遮挡']
        for label, item in zip(labels, by_layout.values()):
            lines.append(f'| {label} | 40 | {item["solution_found"]} | {item["no_solution_found"]} |')
        lines += ['', f'合计解出可行解{summary["solution_found"]}例，未解出{summary["no_solution_found"]}例，未执行0例。',
                  '所有场景保留在200例清单内。诊断前后冻结文件的散列完全一致，未重新摆放或删除任何场景。',
                  '这里记录准备位置的静态求解结果；实际抓取测试保持停止。']
        report.write_text(report.read_text() + '\n'.join(lines) + '\n')
        print('POST_FREEZE_DIAGNOSIS_COMPLETE', summary, flush=True)
    except BaseException:
        save(out / 'halt.json', {'completed': len(records), 'error': traceback.format_exc(),
                                 'scenes_removed': 0, 'scenes_repositioned': 0})
        raise
    finally:
        if process.poll() is None:
            try:
                process.stdin.write('{"stop": true}\n')
                process.stdin.flush()
                process.wait(timeout=20)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.terminate()
                process.wait(timeout=20)
        reader.join(timeout=5)
        stdout.close()
        stderr.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root.resolve())
