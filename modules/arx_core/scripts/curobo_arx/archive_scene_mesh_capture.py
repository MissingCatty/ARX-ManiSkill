"""Byte-exact capture of NEW completed-attempt ARX mesh evidence, directly gzip.

The runtime source is read only. No USD, collision or planner state is changed.
"""
from pathlib import Path
import gzip
import hashlib
import json
import sys


def digest(stream):
    h, length = hashlib.sha256(), 0
    while data := stream.read(1024 * 1024):
        h.update(data); length += len(data)
    return h.hexdigest(), length


def capture(source, destination, started_unix):
    source, destination = Path(source), Path(destination)
    if not source.exists() or source.stat().st_mtime < started_unix:
        return None
    assert not destination.exists()
    compressed = destination.with_suffix('.json.gz')
    assert not compressed.exists()
    before = source.stat()
    temporary = compressed.with_suffix('.gz.partial')
    h, size = hashlib.sha256(), 0
    with source.open('rb') as src, temporary.open('wb') as output:
        with gzip.GzipFile(fileobj=output, mode='wb', compresslevel=1, mtime=0) as gz:
            while data := src.read(1024 * 1024):
                h.update(data); size += len(data); gz.write(data)
    with gzip.open(temporary, 'rb') as stream:
        decoded = digest(stream)
    with source.open('rb') as stream:
        unchanged = digest(stream)
    after = source.stat()
    assert decoded == unchanged == (h.hexdigest(), size)
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    temporary.replace(compressed)
    folder = destination.parent.parent
    report = dict(original_path=str(destination), archive_path=str(compressed),
                  source_capture_path=str(source.resolve()), original_sha256=h.hexdigest(), original_bytes=size,
                  compressed_bytes=compressed.stat().st_size, full_byte_roundtrip_verified=True,
                  source_unchanged_during_capture=True, representation_only=True,
                  restore_command=f'{sys.executable} {folder.parent.parent / "archive_model_evidence.py"} restore {folder}')
    (destination.parent / 'scene_meshes_archive.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
