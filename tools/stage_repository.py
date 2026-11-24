"""Stage project sources while preserving local upstream Git repositories."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', 'tmp', 'weights', 'environments', 'node_modules', '__pycache__'}


def git(directory: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(directory), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def safe_url(url: str) -> str:
    if url.startswith(('http://', 'https://')):
        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.netloc.rsplit('@', 1)[-1], parts.path, '', ''))
    return url


def main() -> None:
    # Local upstream attribute files still exist in the working tree. Override
    # their LFS filters without editing those repositories or uploading models.
    info = Path(git(ROOT, 'rev-parse', '--absolute-git-dir')) / 'info'
    info.mkdir(exist_ok=True)
    attributes = info / 'attributes'
    previous = attributes.read_text() if attributes.exists() else ''
    if '** -filter' not in previous.splitlines():
        attributes.write_text(previous.rstrip() + '\n** -filter\n')
    repositories = []
    for folder, directories, files in os.walk(ROOT / 'modules'):
        path = Path(folder)
        if '.git' in directories or '.git' in files:
            repositories.append(path)
        directories[:] = [name for name in directories
                          if name not in SKIP and not (path / name).is_symlink()]

    sources = []
    for directory in sorted(repositories):
        try:
            origin = safe_url(git(directory, 'remote', 'get-url', 'origin'))
        except subprocess.CalledProcessError:
            origin = None
        sources.append({'path': directory.relative_to(ROOT).as_posix(),
                        'origin': origin, 'commit': git(directory, 'rev-parse', 'HEAD')})
    (ROOT / 'configs/upstream_sources.json').write_text(
        json.dumps({'sources': sources}, ensure_ascii=False, indent=2) + '\n')

    # Git otherwise stages nested repositories as empty gitlinks. Move only
    # their metadata for the duration of staging, then restore it in all cases.
    work = ROOT / 'tmp/git/staging'
    work.mkdir(parents=True, exist_ok=True)
    moved = []
    with tempfile.TemporaryDirectory(dir=work, prefix='metadata-') as folder:
        try:
            for index, directory in enumerate(repositories):
                source = directory / '.git'
                destination = Path(folder) / str(index)
                shutil.move(source, destination)
                moved.append((source, destination))
            subprocess.run(['git', '-C', str(ROOT), 'add', '--all'], check=True)
        finally:
            for source, destination in reversed(moved):
                shutil.move(destination, source)
    entries = git(ROOT, 'ls-files', '--stage').splitlines()
    if any(line.startswith('160000 ') for line in entries):
        raise RuntimeError('Nested repository was staged as a gitlink')
    print(f'Staged project files; preserved {len(repositories)} upstream repositories.')


if __name__ == '__main__':
    main()
