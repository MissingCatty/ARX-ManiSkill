"""Shared-memory candidate delivery and bounded background artifact storage.

The generator owns a transport until its consumer commits it. Persisting the
human-readable snapshot overlaps planning; commit joins writes and propagates
errors before a caller can mark the request complete.
"""
import atexit
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import inspect
import json
from multiprocessing import shared_memory
from pathlib import Path
import os
import threading
import uuid

from .layout import ensure_run_path

FORMAT = 'arx_candidate_json_shm_v1'
MAX_TRANSPORT_BYTES = 32*1024*1024
MAX_PENDING_TRANSPORTS = 4
MAX_PENDING_BYTES = 64*1024*1024


class ArtifactWriteError(RuntimeError):
    pass


@dataclass(frozen=True)
class EncodedJSON:
    data: bytes


def encode(value, *, indent=None):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=indent,
        default=lambda item: item.tolist(), separators=(',', ':') if indent is None else None).encode()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    try:
        with pending.open('xb') as stream:
            if isinstance(value, EncodedJSON):
                value = json.loads(value.data)
            stream.write(encode(value, indent=2)+b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        pending.replace(path)
        return {'path': str(path), 'size': path.stat().st_size}
    finally:
        pending.unlink(missing_ok=True)


class BackgroundWriter:
    """A single ordered writer with backpressure and mandatory error readback."""
    def __init__(self, *, capacity=8):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='arx-artifact-writer')
        self.capacity = threading.BoundedSemaphore(capacity)
        self.futures = []
        self.closed = False

    def submit(self, path, value):
        if self.closed:
            raise RuntimeError('Artifact writer is closed')
        self.capacity.acquire()
        try:
            future = self.executor.submit(atomic_json, Path(path), value)
        except BaseException:
            self.capacity.release()
            raise
        future.add_done_callback(lambda _: self.capacity.release())
        self.futures.append(future)
        return future

    def flush(self):
        futures, self.futures = self.futures, []
        failures = []
        records = []
        for future in futures:
            try:
                records.append(future.result())
            except Exception as error:
                failures.append(error)
        if failures:
            raise ArtifactWriteError('Background artifact storage failed: '+str(failures[0])) from failures[0]
        return records

    def close(self):
        if self.closed:
            return
        try:
            self.flush()
        finally:
            self.closed = True
            self.executor.shutdown(wait=True)


@dataclass
class Published:
    descriptor: dict
    memory: shared_memory.SharedMemory
    writer: BackgroundWriter


_owned = {}
_borrowed = {}
_lock = threading.RLock()


def key(directory):
    return str(ensure_run_path(directory))


def publish_candidate_pool(directory, payload):
    directory = ensure_run_path(directory)
    token = uuid.uuid4().hex
    data = encode(payload)
    if not 0 < len(data) <= MAX_TRANSPORT_BYTES:
        raise ValueError('Candidate transport size exceeds the bounded shared buffer')
    name = key(directory)
    with _lock:
        if name in _owned or name in _borrowed:
            raise RuntimeError('Candidate transport already exists for this observation')
        if (len(_owned) >= MAX_PENDING_TRANSPORTS or
                sum(item.descriptor['nbytes'] for item in _owned.values())+len(data) > MAX_PENDING_BYTES):
            raise RuntimeError('Pending candidate snapshots must be consumed or committed before publishing more')
        target = directory/'candidates/candidate_pool.json'
        if target.exists():
            raise RuntimeError('Candidate snapshot already exists; do not overwrite an inference')
        memory = shared_memory.SharedMemory(create=True, size=len(data))
        writer = BackgroundWriter()
        try:
            memory.buf[:len(data)] = data
            descriptor = {'format': FORMAT, 'name': memory.name, 'nbytes': len(data),
                'sha256': hashlib.sha256(data).hexdigest(), 'token': token,
                'directory': name, 'count': len(payload['candidates']), 'owner_pid': os.getpid()}
            writer.submit(target, EncodedJSON(data))
            _owned[name] = Published(descriptor, memory, writer)
            return dict(descriptor)
        except BaseException:
            writer.close()
            memory.close(); memory.unlink()
            raise


def validate_descriptor(directory, descriptor):
    if (descriptor.get('format') != FORMAT or descriptor.get('directory') != key(directory)
            or not isinstance(descriptor.get('nbytes'), int)
            or not 0 < descriptor['nbytes'] <= MAX_TRANSPORT_BYTES
            or not isinstance(descriptor.get('token'), str) or len(descriptor['token']) != 32
            or not isinstance(descriptor.get('sha256'), str) or len(descriptor['sha256']) != 64
            or not isinstance(descriptor.get('count'), int) or descriptor['count'] < 0):
        raise ValueError('Candidate transport does not belong to this observation')


def read_candidate_transport(directory, descriptor):
    validate_descriptor(directory, descriptor)
    options = {'name': descriptor['name'], 'create': False}
    if 'track' in inspect.signature(shared_memory.SharedMemory).parameters:
        options['track'] = False
    memory = shared_memory.SharedMemory(**options)
    if 'track' not in options and descriptor['owner_pid'] != os.getpid():
        # These RPC workers are independent subprocesses, with independent trackers.
        from multiprocessing import resource_tracker
        resource_tracker.unregister(memory._name, 'shared_memory')
    try:
        if memory.size != descriptor['nbytes']:
            raise ValueError('Candidate shared buffer size differs')
        data = bytes(memory.buf[:descriptor['nbytes']])
        if hashlib.sha256(data).hexdigest() != descriptor['sha256']:
            raise ValueError('Candidate shared buffer changed')
        payload = json.loads(data)
        if len(payload['candidates']) != descriptor['count']:
            raise ValueError('Candidate transport count differs')
        return payload
    finally:
        memory.close()


def adopt_candidate_transport(directory, descriptor, producer_socket):
    validate_descriptor(directory, descriptor)
    name = key(directory)
    with _lock:
        if name in _owned or name in _borrowed:
            raise RuntimeError('A candidate observation already has an outstanding producer')
        _borrowed[name] = {'descriptor': dict(descriptor), 'producer_socket': str(producer_socket)}


def candidate_transport(directory):
    name = key(directory)
    with _lock:
        if name in _owned:
            return dict(_owned[name].descriptor)
        if name in _borrowed:
            return dict(_borrowed[name]['descriptor'])
    return None


def commit_candidate_transport(directory, token):
    name = key(directory)
    with _lock:
        owned = _owned.get(name)
        if owned is None or owned.descriptor['token'] != token:
            raise ValueError('Candidate commit token does not match its producer')
    try:
        records = owned.writer.flush()
        receipt = {'persisted': True, 'token': token, 'artifacts': records,
                   'transport_sha256': owned.descriptor['sha256']}
        atomic_json(Path(directory)/'candidate_snapshot_commit.json', receipt)
        return receipt
    finally:
        with _lock:
            _owned.pop(name, None)
        try:
            owned.writer.close()
        finally:
            owned.memory.close(); owned.memory.unlink()


def finish_candidate_transport(directory, *, client_factory=None):
    name = key(directory)
    with _lock:
        own = _owned.get(name)
        borrowed = _borrowed.get(name)
    if own is not None:
        return commit_candidate_transport(directory, own.descriptor['token'])
    if borrowed is None:
        return None
    if client_factory is None:
        from .client import Client
        client_factory = Client
    try:
        response = client_factory(borrowed['producer_socket']).call({'command': 'artifact_commit',
            'directory': name, 'token': borrowed['descriptor']['token']})
        if response.get('status') != 'ok' or not response.get('result', {}).get('persisted'):
            raise ArtifactWriteError('Remote candidate snapshot failed: '+str(response))
        return response['result']
    finally:
        with _lock:
            _borrowed.pop(name, None)


def close_owned_transports():
    failures = []
    for directory in list(_owned):
        try:
            finish_candidate_transport(directory)
        except Exception as error:
            failures.append(error)
    if failures:
        raise ArtifactWriteError('Outstanding candidate storage failed: '+str(failures[0])) from failures[0]


atexit.register(close_owned_transports)
