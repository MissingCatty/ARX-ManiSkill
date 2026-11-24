"""Validate and package ARX feedback evidence after all simulation trials stop."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import numpy as np

from summarize_visual_feedback import compare, read


def package(out):
    compare(out)
    report = read(out / 'comparison.json')
    root = Path(__file__).resolve().parents[2]
    camera = read(out / 'viewport_before.json')
    command_checks, stage_checks = [], []
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    def decode(path):
        subprocess.run([ffmpeg, '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def montage(name, rows):
        manifest = out / (name + '_concat.txt')
        manifest.write_text(''.join(f"file '{r['video']}'\n" for r in rows))
        video = out / (name + '.mp4')
        subprocess.run([ffmpeg, '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i',
                        str(manifest), '-c', 'copy', '-an', '-movflags', '+faststart', str(video)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        decode(video)
        return {'video': str(video), 'trial_order': [r['name'] for r in rows], 'decoded': True}

    for row in report['rows']:
        folder = Path(row['folder'])
        result = read(folder / 'result.json')
        assert result['recording']['camera_matrix'] == camera['world_matrix'], row['name']
        assert result['recording']['camera_path'] == camera['camera_path'], row['name']
        assert row['bound_weight'] == 10000.
        decode(row['video'])
        initial = read(folder / 'capture_snapshot.json')
        limits = np.asarray([initial['joint_limits'][initial['joint_names'].index(n)] for n in initial['right_arm']])
        for path in folder.glob('*_command.json'):
            command = read(path)
            q = np.asarray(command['position'])
            violation = float(np.maximum(limits[:, 0] - q, q - limits[:, 1]).clip(0).max())
            assert violation <= 1e-5, (row['name'], path.name, violation)
            command_checks.append({'trial': row['name'], 'stage': path.stem.removesuffix('_command'),
                                   'max_command_limit_exceed_rad': violation})
        for stage, optimized in row['optimization'].items():
            executed = (folder / (stage + '_execution.json')).exists()
            if executed:
                assert optimized['success'] and optimized['invalid_samples'] == 0
                assert optimized['max_limit_exceed_rad'] <= 1e-6
            stage_checks.append({'trial': row['name'], 'stage': stage, 'executed': executed, **optimized})
        if row.get('paired_initial_proposal'):
            assert row['paired_initial_proposal']['identical_initial_proposal']
    old = read(root / 'runs/object_failure_search_20261019/baseline.json')
    protected = read(out / 'baseline_preservation.json')
    preserved = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() == old['source_sha256'][name]
                 for name in protected}
    assert all(preserved.values())
    by_name = {r['name']: r for r in report['rows']}
    paired_rows = []
    bagel = '12_bagel_feedback_bounded_state' if '12_bagel_feedback_bounded_state' in by_name else '09_bagel_feedback_feasible'
    sphere = '13_sphere_feedback_envelope' if '13_sphere_feedback_envelope' in by_name else '11_sphere_feedback_feasible'
    for name, before in [(bagel, '04_bagel_failed_seed_baseline'),
                         ('10_marker_feedback_feasible', '02_marker_baseline'),
                         (sphere, '03_sphere_baseline')]:
        if name in by_name:
            paired_rows.extend([by_name[before], by_name[name]])
    montages = [montage('all_trials', report['rows'])]
    if paired_rows:
        montages.append(montage('paired_feasible', paired_rows))
    verification = {'trials': len(report['rows']), 'all_videos_decoded': True,
                    'all_same_webrtc_camera': True, 'protected_baseline_files_unchanged': preserved,
                    'command_checks': command_checks, 'optimization_checks': stage_checks,
                    'unexpected_contact_events': sum(s['unexpected_contact_events'] for r in report['rows'] for s in r['stages'].values()),
                    'max_measured_joint_limit_exceed_rad': max(r['measured_joint_limit_max_exceed_rad'] for r in report['rows']),
                    'montages': montages, 'video_editing': 'Planning waits removed; original viewport frames; no titles, text or audio'}
    (out / 'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2))
    fields = ['name', 'asset', 'execution_policy', 'success', 'outcome', 'integration_failure',
              'lift_peak_cm', 'hold_min_cm', 'actual_pose_updates', 'video']
    with (out / 'trials.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in report['rows']:
            record = {k: row[k] for k in fields if k in row}
            record['lift_peak_cm'] = row['stages']['lift']['object_max_rise_m'] * 100 if 'lift' in row['stages'] else ''
            record['hold_min_cm'] = row['stages']['hold']['object_min_rise_m'] * 100 if 'hold' in row['stages'] else ''
            record['actual_pose_updates'] = sum(u['pose_changed'] for u in row.get('feedback', {}).get('updates', []))
            writer.writerow(record)
    with (out / 'readme_zh.txt').open('a') as f:
        f.write('\n视频：all_trials.mp4 按逐试次列表顺序；paired_feasible.mp4 顺序为贝果基线/新版、记号笔基线/新版、绿色球基线/新版。均无标题、文字或音轨。\n')
    print(json.dumps({'trials': verification['trials'], 'planned_stages': len(stage_checks),
                      'montages': montages}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    package(parser.parse_args().output.resolve())
