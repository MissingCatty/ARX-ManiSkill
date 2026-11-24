"""Verify and restore the resource archives downloaded from Google Drive."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def verify(backup):
    backup = backup.resolve()
    manifest = json.loads((backup / 'manifest.json').read_text())
    if manifest['state'] != 'ready':
        raise ValueError('The resource backup is incomplete')
    for group in manifest['groups']:
        for part in group['parts']:
            path = (backup / part['path']).resolve()
            if not path.is_relative_to(backup):
                raise ValueError('Invalid backup part path')
            if path.stat().st_size != part['bytes'] or digest(path) != part['sha256']:
                raise ValueError('Resource checksum mismatch: ' + part['path'])
    return manifest


def restore(backup, destination, verify_only=False):
    # Verify every part before writing any resource to the destination.
    manifest = verify(backup)
    if verify_only:
        return manifest
    destination.mkdir(parents=True, exist_ok=True)
    for group in manifest['groups']:
        decoder = subprocess.Popen(['zstd', '-q', '-d', '-c'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        extractor = subprocess.Popen(['tar', '-xpf', '-', '-C', str(destination)], stdin=decoder.stdout)
        decoder.stdout.close()
        try:
            for part in group['parts']:
                with (backup / part['path']).open('rb') as handle:
                    while block := handle.read(8 * 1024 * 1024):
                        decoder.stdin.write(block)
        finally:
            decoder.stdin.close()
        decode_code = decoder.wait()
        extract_code = extractor.wait()
        if decode_code or extract_code:
            raise RuntimeError('Resource extraction failed: ' + group['id'])
        print('Restored', group['id'], flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup-dir', type=Path, default=Path('.'))
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not args.verify_only and args.destination is None:
        parser.error('--destination is required when restoring resources')
    manifest = restore(args.backup_dir, args.destination, args.verify_only)
    print('Verified', len(manifest['groups']), 'groups; source commit:', manifest['source_commit'])
