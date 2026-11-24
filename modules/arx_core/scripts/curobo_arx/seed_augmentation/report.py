"""Read-only trial evidence -> dataset views, coverage report and real videos."""
import argparse
import collections
import json
from pathlib import Path
import sqlite3
import subprocess
import time
from scripts.curobo_arx.data_factory.common import read, write, sha


def records(campaign):
    campaign = Path(campaign).resolve(); reg = read(campaign/'registration.json'); rows = []
    for path in (campaign/'groups').glob('*/seed_results.json'):
        rows.extend(read(path)['trials'])
    recovered=campaign/'reconciled_results.json'
    if recovered.exists():
        for row in read(recovered)['trials']:
            prior=next((r for r in rows if r['trial_id']==row['trial_id']),None)
            if prior is None:rows.append(row)
            elif prior!=row:raise ValueError('Reconciled trial conflicts with original result')
    ids = [row['trial_id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate trial result')
    expected = {trial['trial_id'] for trial in reg['seed_trials']}
    if set(ids)-expected:
        raise ValueError('Unregistered result')
    return reg, rows


def export(campaign, videos=False):
    campaign = Path(campaign).resolve(); reg, results = records(campaign)
    selected = []; views = []
    for result in results:
        if not result.get('admitted'):
            continue
        dbfile = campaign/'trial_ledgers'/(result['trial_id']+'.sqlite3')
        connection = sqlite3.connect('file:'+str(dbfile)+'?mode=ro', uri=True); connection.row_factory = sqlite3.Row
        accepted = [dict(row) for row in connection.execute("SELECT * FROM attempts WHERE state='committed'")]
        connection.close()
        if len(accepted) != 1:
            raise ValueError('Admitted trial must have exactly one committed episode')
        row = accepted[0]; episode = Path(row['episode']); receipt = json.loads(row['receipt'])
        for name, key in [('quality_receipt.json', 'quality_sha256'), ('physical_receipt.json', 'physical_sha256')]:
            if sha(episode/name) != receipt[key] or read(episode/name)['pass'] is not True:
                raise ValueError('Successful source admission changed')
        meta = read(episode/'episode.json')
        labels = read(episode/'labels.json')
        command_labels = {command['id']: command['label'] for command in read(episode/'requests.json')}
        prefix_items = sum(command_labels.get(item['command_id']) == 'approach_connect' for item in labels['items'])
        audit = read(episode/'seed_augmentation_optimization.json')
        if audit['pass'] is not True:
            raise ValueError('Seed command/connector audit failed')
        if result['policy'] == 'stitch' and 'approach_connect' not in audit['executed_actions']:
            raise ValueError('Stitched success lacks actual connector execution')
        # These are transparent aliases, not new executions or a fabricated
        # merged ledger. The existing converter can consume each view as one
        # campaign, validating the original scoped ledger and acceptance hashes.
        view = campaign/'dataset_views'/result['trial_id']; view.mkdir(parents=True, exist_ok=True)
        for name, source in [('registration.json', campaign/'registration.json'), ('ledger.sqlite3', dbfile)]:
            target = view/name
            if not target.is_symlink():
                if target.exists():raise FileExistsError('Do not overwrite a dataset view')
                target.symlink_to(source)
            elif target.resolve() != source:
                raise ValueError('Dataset view points to another source')
        proof = {'view_only_no_executions': True, 'source_campaign': str(campaign), 'trial_id': result['trial_id'],
                 'source_ledger': str(dbfile), 'source_ledger_sha256': sha(dbfile),
                 'source_registration_sha256': sha(campaign/'registration.json'), 'episode_id': row['id']}
        prior = view/'source_view.json'
        if prior.exists() and read(prior) != proof:
            raise ValueError('Existing dataset view provenance changed')
        if not prior.exists():write(prior, proof)
        views.append(str(view))
        selected.append({'id': row['id'], 'object_id': row['object_id'], 'episode': str(episode),
            'trial_id': result['trial_id'], 'seed_id': result['seed_id'], 'policy': result['policy'],
            'placement_id': result['placement']['placement_id'], 'entry_world': result['entry_world'],
            'source_split': meta['split'], 'raw_samples': meta['samples'],
            'training_role': 'repeatability_control' if result['policy'] == 'control' else 'entry_augmentation',
            'connector_action_items': prefix_items, 'all_action_items': len(labels['items']),
            'original_physical_pass': True, 'image_and_record_pass': True, 'fresh_execution': True,
            'quality_receipt_sha256': receipt['quality_sha256'], 'seed_audit_sha256': sha(episode/'seed_augmentation_optimization.json')})
    counters = collections.Counter(item['policy'] for item in selected)
    unique_entries = {(item['seed_id'], json.dumps(item['entry_world'], sort_keys=True)) for item in selected if item['policy'] != 'control'}
    manifest = {'source_campaign': str(campaign), 'successful_episodes': selected, 'dataset_campaign_views': views,
        'total_admitted': len(selected), 'admitted_by_policy': dict(counters),
        'successful_perturbed_entries': len(unique_entries),
        'successful_seed_object_placements': len({(item['object_id'], item['placement_id']) for item in selected}),
        'new_object_placements_claimed': 0, 'children_inherit_source_object_split': True,
        'sampling_note': 'Group by seed family; avoid overweighting the shared replayed tail. Controls are identified separately from entry augmentation.',
        'raw_data_or_source_ledgers_modified': False, 'updated_unix': time.time()}
    write(campaign/'accepted_seed_episodes.json', manifest)
    if videos:
        encode(campaign, selected)
    return manifest


def encode(campaign, selected):
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe(); out = campaign/'videos'; out.mkdir(exist_ok=True)
    clips = []
    for item in selected:
        episode = Path(item['episode']); dest = out/(item['id']+'.mp4'); receipt = out/(item['id']+'_receipt.json')
        if receipt.exists():
            prior = read(receipt)
            if not dest.exists() or sha(dest) != prior['sha256']:
                raise ValueError('Existing preview changed')
            clips.append(prior); continue
        if dest.exists():raise FileExistsError('Keep partial exports; do not overwrite without diagnosis')
        samples = [json.loads(line) for line in (episode/'samples.jsonl').read_text().splitlines()]
        intervals = []
        for request in read(episode/'requests.json'):
            if request['label'] not in ('approach_connect', 'approach', 'close', 'lift', 'hold'):
                continue
            execution = read(episode/(request['label']+'_execution.json'))
            intervals.append((request['requested_sim_time']-.2, request['requested_sim_time']+execution['sim_duration_s']+.001))
        frames = [row for row in samples if any(a-.0001 <= row['frame_time'] <= b+.0001 for a, b in intervals)]
        if not frames:raise ValueError('No actual augmentation video frames')
        playlist = out/(item['id']+'.ffconcat')
        paths = [Path(row['rgb_path']) for row in frames]
        if any("'" in str(path) for path in paths):raise ValueError('Unsafe ffconcat path')
        playlist.write_text('ffconcat version 1.0\n'+''.join("file '"+str(path)+"'\nduration 0.2\n" for path in paths)+"file '"+str(paths[-1])+"'\n")
        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-f', 'concat', '-safe', '0',
            '-i', str(playlist), '-vf', 'fps=5', '-an', '-c:v', 'libx264', '-threads', '2', '-pix_fmt', 'yuv420p', str(dest)], check=True)
        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-i', str(dest), '-f', 'null', '-'], check=True, capture_output=True)
        value = {**item, 'video': str(dest), 'sha256': sha(dest), 'decode_pass': True, 'frames': len(frames),
            'actual_connector_retained': item['policy'] == 'stitch',
            'timebase': 'Original 5fps simulation cadence, planning waits removed, all actual connector/tail motions retained'}
        write(receipt, value); clips.append(value)
    lines = ['# 成功种子扩充的实际视频', '', '| 物体 | 方法 | 视频 |', '|---|---|---|']
    for clip in clips:
        lines.append(f'| {clip["object_id"]} | {clip["policy"]} | [播放]({clip["video"]}) |')
    lines += ['', 'stitch视频包含实际H′→S连接和完整后段；没有拼入旧视频、生成帧或旧成功结论。']
    (campaign/'VIDEOS.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('campaign'); parser.add_argument('--videos', action='store_true')
    args = parser.parse_args(); result = export(args.campaign, args.videos)
    print(json.dumps({k: v for k, v in result.items() if k not in ('successful_episodes', 'dataset_campaign_views')}, ensure_ascii=False))
