"""Pack, checksum and copy project resources into the configured Drive folder."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from arx_project.layout import ROOT, now, read_json, sha256, write_json


def groups(root=ROOT):
    policy = root / 'configs/cloud_backup.json'
    excluded = json.loads(policy.read_text()).get('exclude', []) if policy.exists() else []

    def included(path):
        relative = path.relative_to(root)
        return not any(relative == Path(name) or Path(name) in relative.parents for name in excluded)

    result = [{'id': 'metadata', 'members': ['configs', 'docs', 'README.md', 'AGENTS.md',
        'simulation/assets/catalog.json', 'simulation/scenes/latest200',
        'simulation/robots', 'simulation/cameras']}]
    for path in sorted((root / 'weights').iterdir()):
        if not included(path):
            continue
        result.append({'id': 'weights_' + path.name, 'members': [str(path.relative_to(root))]})
    active = []
    for path in sorted((root / 'simulation/assets/models').iterdir()):
        if not included(path):
            continue
        if path.name.startswith('source_'):
            result.append({'id': 'assets_' + path.name, 'members': [str(path.relative_to(root))]})
        else:
            active.append(str(path.relative_to(root)))
    result.insert(1, {'id': 'assets_active', 'members': active})
    for name in json.loads(policy.read_text()).get('archive_runs', []) if policy.exists() else []:
        path = Path(name)
        if path.is_absolute() or len(path.parts) != 3 or path.parts[0] != 'tmp' or '..' in path.parts:
            raise ValueError('Archived runs require tmp/<method>/<run_id>: ' + name)
        if not (root / path).is_dir() or not included(root / path):
            raise ValueError('Missing or excluded archived run: ' + name)
        result.append({'id': 'archive_' + '_'.join(path.parts[1:]), 'members': [path.as_posix()]})
    return result


def fingerprint(root, members):
    digest = hashlib.sha256()
    count = size = 0

    def add(path):
        nonlocal count, size
        stat = path.lstat()
        target = os.readlink(path) if path.is_symlink() else None
        if target is not None and (Path(target).is_absolute() or not path.resolve().is_relative_to(root.resolve())):
            raise ValueError('Asset symlink must remain inside the project: ' + str(path))
        name = str(path.relative_to(root))
        if path.name in {'credentials.json', 'token.json', 'service_account.json'} or path.name.endswith(('_keys.json', '.pem', '.key', '.lic')):
            raise ValueError('Credential or license needs separate storage: ' + name)
        digest.update(json.dumps([name, stat.st_mode, stat.st_size, stat.st_mtime_ns, target]).encode())
        count += 1
        if path.is_file() and target is None:
            size += stat.st_size

    for member in members:
        path = root / member
        if path.is_symlink() or path.is_file():
            add(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)
        for folder, directories, files in os.walk(path):
            directories[:] = sorted(n for n in directories if n not in {'.git', '__pycache__'})
            for name in directories + sorted(files):
                if not name.endswith('.pyc'):
                    add(Path(folder) / name)
    return {'signature': digest.hexdigest(), 'files': count, 'bytes': size}


def archive_group(root, payload, group, chunk_bytes, threads=2):
    before = fingerprint(root, group['members'])
    directory = payload / 'packages' / group['id']
    receipt = directory / 'receipt.json'
    if receipt.exists():
        saved = read_json(receipt)
        if saved['source'] != before:
            raise ValueError('Already packed resources changed; create a new snapshot: ' + group['id'])
        for part in saved['parts']:
            if sha256(payload / part['path']) != part['sha256']:
                raise ValueError('Packed resource part is damaged: ' + part['path'])
        return saved
    pending = directory.with_name(directory.name + '.pending')
    if pending.exists():
        shutil.rmtree(pending)
    pending.mkdir(parents=True)
    parts = []
    with (pending / 'pack.log').open('w') as log:
        tar = subprocess.Popen(['tar', '--sort=name', '--exclude=.git', '--exclude=__pycache__',
            '--exclude=*.pyc', '-cf', '-', '-C', str(root), '--', *group['members']],
            stdout=subprocess.PIPE, stderr=log)
        compressor = subprocess.Popen(['zstd', '-q', '-3', '-T' + str(threads), '-c'],
                                      stdin=tar.stdout, stdout=subprocess.PIPE, stderr=log)
        tar.stdout.close()
        index = 0
        try:
            while first := compressor.stdout.read(min(8 * 1024 * 1024, chunk_bytes)):
                name = f'{group["id"]}.tar.zst.part-{index:05d}'
                digest, md5, size = hashlib.sha256(), hashlib.md5(), 0
                with (pending / name).open('wb') as output:
                    block = first
                    while block:
                        output.write(block)
                        digest.update(block)
                        md5.update(block)
                        size += len(block)
                        if size == chunk_bytes:
                            break
                        block = compressor.stdout.read(min(8 * 1024 * 1024, chunk_bytes - size))
                parts.append({'path': str((directory / name).relative_to(payload)), 'bytes': size,
                              'sha256': digest.hexdigest(), 'md5': md5.hexdigest()})
                index += 1
        except BaseException:
            compressor.terminate()
            tar.terminate()
            raise
        finally:
            compressor.stdout.close()
        compression_code, tar_code = compressor.wait(), tar.wait()
    if compression_code or tar_code:
        raise RuntimeError('Packing failed: ' + group['id'] + '; see ' + str(pending / 'pack.log'))
    if fingerprint(root, group['members']) != before:
        raise RuntimeError('Resources changed during packing: ' + group['id'])
    saved = {'id': group['id'], 'members': group['members'], 'source': before, 'parts': parts}
    write_json(pending / 'receipt.json', saved)
    pending.rename(directory)
    return saved


def reuse_groups(root, previous, payload, planned):
    """Reuse immutable parts only when the source inventory still matches."""
    reused = []
    for group in planned:
        if group['id'] == 'metadata':
            continue
        old = previous / 'payload/packages' / group['id']
        if not (old / 'receipt.json').is_file():
            continue
        saved = json.loads((old / 'receipt.json').read_text())
        if saved['source'] != fingerprint(root, group['members']):
            continue
        destination = payload / 'packages' / group['id']
        if destination.exists():
            continue
        pending = destination.with_name(destination.name + '.reuse.pending')
        if pending.exists():
            shutil.rmtree(pending)
        pending.mkdir(parents=True)
        try:
            for part in saved['parts']:
                source = previous / 'payload' / part['path']
                if source.stat().st_size != part['bytes']:
                    raise ValueError('Reusable resource part has the wrong size: ' + str(source))
                os.link(source, pending / source.name)
            shutil.copy2(old / 'receipt.json', pending / 'receipt.json')
            pending.rename(destination)
        except BaseException:
            shutil.rmtree(pending)
            raise
        reused.append(group['id'])
    return reused


def pack(snapshot, config, reuse=None):
    snapshot.mkdir(parents=True, exist_ok=True)
    payload = snapshot / 'payload'
    payload.mkdir(exist_ok=True)
    planned = groups()
    if reuse is not None:
        reused = reuse_groups(ROOT, reuse, payload, planned)
        write_json(snapshot / 'reuse.json', {'from': str(reuse), 'groups': reused,
                    'parts_verified_during_pack': True, 'at': now()})
    plan_path = snapshot / 'plan.json'
    if plan_path.exists():
        plan = read_json(plan_path)
        if plan['groups'] != planned:
            raise ValueError('Resource groups changed; create a new snapshot')
    else:
        plan = {'at': now(), 'groups': planned,
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'folder_id': config['folder_id'], 'account_email': config['account_email']}
        write_json(plan_path, plan)
    results = []
    for group in planned:
        write_json(snapshot / 'status.json', {'stage': 'packing', 'group': group['id'],
                   'completed_groups': len(results), 'total_groups': len(planned), 'at': now()})
        print('Packing', group['id'], flush=True)
        results.append(archive_group(ROOT, payload, group, config['chunk_bytes'], config['compression_threads']))
    manifest = {'version': 1, 'state': 'ready', 'at': now(), 'snapshot': snapshot.name,
                'source_commit': plan['source_commit'], 'groups': results,
                'original_bytes': sum(g['source']['bytes'] for g in results),
                'compressed_bytes': sum(p['bytes'] for g in results for p in g['parts']),
                'scene_count': 200, 'scene_freeze_sha256': sha256(ROOT / 'simulation/scenes/latest200/frozen_dataset.json'),
                'asset_catalog_sha256': sha256(ROOT / 'simulation/assets/catalog.json')}
    write_json(payload / 'manifest.json', manifest)
    shutil.copy2(ROOT / 'tools/restore_resources.py', payload / 'restore_resources.py')
    sums = [f'{p["sha256"]}  {p["path"]}' for g in results for p in g['parts']]
    sums += [sha256(payload / name) + '  ' + name for name in ['manifest.json', 'restore_resources.py']]
    (payload / 'SHA256SUMS').write_text('\n'.join(sums) + '\n')
    write_json(snapshot / 'status.json', {'stage': 'ready_to_upload', 'at': now(),
                                         'compressed_bytes': manifest['compressed_bytes']})
    return manifest


def upload(snapshot, config, wait_for_auth):
    from tools.restore_resources import verify
    manifest = verify(snapshot / 'payload')
    private = Path(config['credential_file']).expanduser()
    while not private.exists():
        write_json(snapshot / 'status.json', {'stage': 'waiting_for_google_authorization', 'at': now(),
                    'compressed_bytes': manifest['compressed_bytes']})
        if not wait_for_auth:
            raise RuntimeError('Google Drive authorization is required')
        time.sleep(20)
    command = [str(ROOT / config['rclone_binary']), '--config', str(private),
               '--drive-root-folder-id', config['folder_id']]
    quota = json.loads(subprocess.check_output(command + ['about', config['remote_name'] + ':', '--json'], text=True))
    required = sum(p.stat().st_size for p in (snapshot / 'payload').rglob('*') if p.is_file())
    # Count already uploaded bytes so restarting a completed/partial copy does
    # not require an extra full snapshot worth of free space.
    remote = config['remote_name'] + ':' + config['remote_prefix'] + '/' + snapshot.name
    existing = subprocess.run(command + ['size', remote, '--json'], capture_output=True, text=True)
    uploaded = json.loads(existing.stdout)['bytes'] if existing.returncode == 0 else 0
    needed = max(0, required - uploaded)
    if 'free' in quota and quota['free'] < needed:
        write_json(snapshot / 'status.json', {'stage': 'insufficient_drive_space', 'at': now(),
                   'free_bytes': quota['free'], 'required_additional_bytes': needed, 'snapshot_bytes': required})
        raise RuntimeError('Google Drive has insufficient free storage; see snapshot/status.json')
    write_json(snapshot / 'status.json', {'stage': 'uploading', 'at': now(), 'remote': remote,
                                         'snapshot_bytes': required, 'quota': quota})
    with (snapshot / 'upload.log').open('a') as log:
        subprocess.run(command + ['copy', str(snapshot / 'payload'), remote, '--checksum',
            '--transfers', '2', '--checkers', '4', '--drive-chunk-size', '64M',
            '--tpslimit', '8', '--retries', '5', '--stats', '30s',
            '--stats-log-level', 'NOTICE'], stdout=log, stderr=log, check=True)
        subprocess.run(command + ['check', str(snapshot / 'payload'), remote, '--one-way',
            '--checkers', '4', '--combined', str(snapshot / 'remote_check.txt')], stdout=log, stderr=log, check=True)
        completion = snapshot / 'remote_completion.json'
        write_json(completion, {'snapshot': snapshot.name, 'at': now(),
                    'manifest_sha256': sha256(snapshot / 'payload/manifest.json'),
                    'remote_checksum_check_passed': True, 'snapshot_bytes': required})
        subprocess.run(command + ['copyto', str(completion), remote + '/COMPLETE.json',
                       '--checksum'], stdout=log, stderr=log, check=True)
    write_json(snapshot / 'status.json', {'stage': 'complete', 'at': now(), 'remote': remote,
               'snapshot_bytes': required, 'remote_checksum_check_passed': True,
               'folder_url': 'https://drive.google.com/drive/folders/' + config['folder_id']})
    print('Google Drive upload and remote checksum check complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['pack', 'upload', 'run'])
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--wait-for-auth', action='store_true')
    parser.add_argument('--reuse', type=Path, help='Reuse verified immutable parts from an earlier local snapshot')
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    if not snapshot.is_relative_to(ROOT / 'tmp/cloud_backup'):
        parser.error('Snapshots require tmp/cloud_backup/<snapshot_id>')
    snapshot.mkdir(parents=True, exist_ok=True)
    if args.reuse is not None:
        args.reuse = args.reuse.resolve()
        if not args.reuse.is_relative_to(ROOT / 'tmp/cloud_backup') or args.reuse == snapshot:
            parser.error('--reuse requires a different snapshot in tmp/cloud_backup')
    config = read_json(ROOT / 'configs/cloud_backup.json')
    with (snapshot / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.stage in ['pack', 'run']:
                pack(snapshot, config, args.reuse)
            if args.stage in ['upload', 'run']:
                upload(snapshot, config, args.wait_for_auth)
        except Exception as error:
            write_json(snapshot / 'last_error.json', {'at': now(), 'type': type(error).__name__, 'error': str(error)})
            status_path = snapshot / 'status.json'
            previous = read_json(status_path) if status_path.exists() else {}
            if previous.get('stage') != 'insufficient_drive_space':
                write_json(status_path, {'stage': 'failed', 'failed_stage': previous.get('stage'),
                           'at': now(), 'error': str(error)})
            raise
