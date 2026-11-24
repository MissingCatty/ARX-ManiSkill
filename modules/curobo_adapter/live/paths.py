"""Paths and file helpers shared by the isolated live diagnostic."""
from pathlib import Path
import hashlib
import json
import sys

BASE = Path(__file__).resolve().parents[1]
MS = BASE.parent / 'maniskill'
ARX = BASE.parent / 'arx_core'
sys.path[:0] = [str(BASE), str(MS)]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def guard(reg):
    for path, expected in reg['runtime_source_hashes'].items():
        if sha(path) != expected:
            raise RuntimeError('Frozen runtime source changed: '+path)
