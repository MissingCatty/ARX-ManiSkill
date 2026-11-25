"""Read-only experiment follow-up, independent checks and V4 website publication."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from arx_project.layout import ROOT, now, read_json, sha256, write_json


def audit(root):
    reg = read_json(root/'registration.json')
    rows, issues, videos = [], [], []
    for entry in reg['schedule']:
        directory = root/'trials'/entry['id']
        claim = root/'claims'/(entry['id']+'.json')
        if not claim.exists():
            rows.append({'id': entry['id'], 'state': 'not_started'})
            continue
        receipt = read_json(claim)
        if receipt['state'] != 'complete':
            rows.append({'id': entry['id'], 'state': 'interrupted'})
            continue
        result = read_json(directory/'result.json')
        if sha256(directory/'result.json') != receipt['result_sha256']:
            issues.append(entry['id']+': result hash differs from claim')
        record = {'id': entry['id'], 'state': 'complete', 'success': result['success'],
                  'reason': result['reason'], 'selected_arm': result.get('selected_arm')}
        if result.get('stationing', {}).get('motion_started'):
            station = directory/'stationing'
            required = ['command.json', 'receipt.json', 'settled_state.json', 'execution.jsonl', 'contacts.json', 'motion.mp4', 'head.mp4']
            missing = [name for name in required if not (station/name).is_file() or not (station/name).stat().st_size]
            if missing:
                issues.append(entry['id']+': missing station evidence '+str(missing))
            else:
                command, actual = read_json(station/'command.json'), read_json(station/'receipt.json')
                final = read_json(station/'settled_state.json')
                h = final['qpos'][final['joint_names'].index('joint4')]
                if abs(h-command['goal']['lift_m']) >= .001 or not actual['success'] or actual['idle_samples'] < 60:
                    issues.append(entry['id']+': station not settled')
                if read_json(station/'contacts.json'):
                    issues.append(entry['id']+': unexpected transit contact')
                rows_trace = [json.loads(line) for line in (station/'execution.jsonl').read_text().splitlines()]
                if len(rows_trace) < 60 or max(np.abs(r['qvel']).max() for r in rows_trace[-60:]) >= .01:
                    issues.append(entry['id']+': idle trace does not confirm rest')
                if np.abs(np.asarray(final['links']['base_link'])-np.asarray(command['goal']['base_world'])).max() > .001:
                    issues.append(entry['id']+': base readback differs from goal')
                record['station_trace_samples'] = len(rows_trace)
                coarse = read_json(station/'coarse/state.json')
                initial_base = np.asarray(coarse['links']['base_link'])
                record['base_translation_m'] = float(np.linalg.norm(initial_base[:2, 3]-np.asarray(final['links']['base_link'])[:2, 3]))
                record['lift_change_m'] = float(h-coarse['qpos'][coarse['joint_names'].index('joint4')])
                fresh = read_json(directory/'state.json')
                if np.abs(np.asarray(fresh['camera_world_optical'])-np.asarray(final['camera_world_optical'])).max() > .001:
                    issues.append(entry['id']+': fresh camera is not at settled station')
        if result.get('physical_attempt'):
            sys.path.insert(0, str(ROOT/'modules/maniskill'))
            from planning_common.contact_execution import verify_trial
            from modules.curobo.fingertip_contact import load_profile
            verify_trial(directory, load_profile())
            evaluation = read_json(directory/'independent_evaluation.json')
            if not evaluation['passed'] or evaluation.get('independently_successful', False) != result['success']:
                issues.append(entry['id']+': independent grasp result mismatch')
        for video in directory.glob('*.mp4'):
            videos.append(video)
        for video in (directory/'stationing').glob('*.mp4'):
            videos.append(video)
        rows.append(record)
    import imageio_ffmpeg
    binary = imageio_ffmpeg.get_ffmpeg_exe()
    def decode(path):
        done = subprocess.run([binary, '-v', 'error', '-threads', '1', '-i', str(path),
            '-threads', '1', '-f', 'null', '-'], capture_output=True, text=True, timeout=300)
        return {'file': str(path.relative_to(root)), 'passed': done.returncode == 0,
                'error': done.stderr[-1000:] if done.returncode else None}
    with ThreadPoolExecutor(max_workers=4) as pool:
        decoded = list(pool.map(decode, videos))
    if not all(row['passed'] for row in decoded):
        issues.append('video decoding failed')
    for name, digest in reg['source_hashes'].items():
        if sha256(name) != digest:
            issues.append('source/config changed: '+name)
    frozen = read_json(Path(reg['dataset'])/'frozen_dataset.json')
    for name, digest in frozen['files'].items():
        if sha256(Path(reg['dataset'])/name) != digest:
            issues.append('scene changed: '+name)
    completed = sum(row['state'] == 'complete' for row in rows)
    report = {'at': now(), 'passed': not issues, 'task_complete': completed == 100 and not (root/'STOP').exists(),
              'registered': 100, 'completed': completed,
              'interrupted': sum(row['state'] == 'interrupted' for row in rows),
              'not_started': sum(row['state'] == 'not_started' for row in rows),
              'successes': sum(row.get('success', False) for row in rows),
              'record_integrity': not issues, 'issues': issues, 'video_count': len(decoded),
              'videos': decoded, 'results': rows, 'audit_source_sha256': sha256(__file__)}
    write_json(root/'acceptance.json', report)
    return report


def publish(root):
    node = os.environ.get('ARX_NODE_EXECUTABLE') or shutil.which('node')
    if node is None:
        managed = ROOT/'environments/node/bin/node'
        if managed.is_file():
            node = str(managed)
        else:
            # NVM's executable is normally added by the interactive shell and
            # is absent from systemd's PATH. Discover its installed versions.
            candidates = list((Path.home()/'.nvm/versions/node').glob('v*/bin/node'))
            if candidates:
                def version(path):
                    return tuple(int(part) for part in path.parents[1].name.lstrip('v').split('.'))
                node = str(max(candidates, key=version))
    if node is None:
        # User services have a smaller PATH than an interactive terminal.
        # Resolve the user's configured Node runtime through their login shell.
        resolved = subprocess.run(['/bin/bash', '-lc', 'command -v node'],
                                  capture_output=True, text=True, check=True)
        node = resolved.stdout.strip().splitlines()[-1]
    if not Path(node).is_file():
        raise RuntimeError('Configured Node runtime is unavailable: '+node)
    subprocess.run([sys.executable, str(ROOT/'website/tools/publish_mobile_v4.py'), '--root', str(root)], check=True)
    with (root/'website_build.log').open('a') as stream:
        subprocess.run([node, 'build.mjs'], cwd=ROOT/'website', stdout=stream, stderr=subprocess.STDOUT, check=True)


def write_report(root, report):
    """Render accepted terminal results separately from interrupted claims."""
    summary = read_json(root/'summary.json')
    complete_ids = {row['id'] for row in report['results'] if row['state'] == 'complete'}
    from collections import Counter
    reasons = Counter(row['reason'] for row in summary['results']
                      if row['id'] in complete_ids and not row['success'])
    text = ['# V4：100场景测试结果', '',
        f"登记100例，完成{report['completed']}例，中断{report['interrupted']}例，未开始{report['not_started']}例。", '',
        f"站位到位{summary['station_ready']}例，最终抓取规划通过{summary['planning_passes']}例，实际抓取{summary['physical_attempts']}例，成功{summary['successes']}例。", '',
        '| 布局 | 完成数 | 最终规划通过 | 抓取成功 |', '| --- | ---: | ---: | ---: |']
    for layout, row in summary['by_layout'].items():
        completed = sum(r['layout'] == layout and r['id'] in complete_ids for r in summary['results'])
        text.append(f"| {layout} | {completed}/20 | {row['planning_passes']} | {row['successes']} |")
    text += ['', '## 主要失败原因', '', '| 原因 | 数量 |', '| --- | ---: |']
    sys.path.insert(0, str(ROOT/'website/tools'))
    from publish_mobile_v4 import REASONS
    for reason, count in reasons.items():
        text.append(f'| {REASONS.get(reason, reason)} | {count} |')
    text += ['', '## 记录核验', '',
        f"独立核验{'通过' if report['passed'] else '发现问题'}；{report['video_count']}段视频完成逐帧解码检查。", '',
        f"原始记录：`{root.relative_to(ROOT)}`。", '',
        '入口计算、移动车体与升降、到位后感知和最终抓取分别留有记录；结果以本轮独立100例为分母。', '']
    destination = ROOT/'modules/pipeline/docs/mainline_versions/V4_100场景测试结果.md'
    destination.write_text('\n'.join(text))


def finish(root):
    report = audit(root)
    write_report(root, report)
    publish(root)
    write_json(root/'publication_complete.json', {'at': now(), 'audit_passed': report['passed'],
        'task_complete': report['task_complete'], 'page': '/iterations/sam3-graspgenx/versions/V4/'})


def watch(root):
    last_published = 0
    failures = 0
    while True:
        progress = read_json(root/'progress.json')
        stopped = root/'services_stopped.json'
        try:
            if progress['phase'] in ['complete', 'stopped'] and stopped.exists() \
                    and read_json(stopped)['all_owned_exited']:
                finish(root)
                return
            if progress['completed'] >= last_published+10:
                publish(root)
                last_published = progress['completed']
                failures = 0
                write_json(root/'publication_status.json', {'at': now(), 'state': 'published',
                    'completed': last_published})
        except (OSError, subprocess.SubprocessError, RuntimeError) as error:
            failures += 1
            write_json(root/'publication_status.json', {'at': now(), 'state': 'publication_error',
                'error': str(error), 'consecutive_failures': failures})
            print('PUBLICATION_ERROR', now(), str(error), flush=True)
        # Background watcher only: no model loading, restart or robot action.
        time.sleep(min(300, 30*max(1, failures)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    (watch if args.watch else finish)(args.root.resolve())
