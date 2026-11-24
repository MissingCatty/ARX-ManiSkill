"""Small shared utilities. Importing this package never starts a GPU or simulator."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[3]
VERSION = 'arx_data_factory_v1'
FLOOR = 1073741824
STAGES = ('pregrasp', 'approach', 'close', 'lift', 'hold')


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def space(path, required=0):
    if shutil.disk_usage(path).free - required <= FLOOR:
        raise OSError('Preserve the registered 1 GiB local storage reserve')


def stamp():
    return {'unix': time.time(), 'monotonic': time.monotonic()}


def verify_files(files):
    changed = [p for p, expected in files.items() if not Path(p).is_file() or sha(p) != expected]
    if changed:
        raise RuntimeError('Registered input/source changed: ' + str(changed[:10]))
