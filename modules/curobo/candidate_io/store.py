"""One candidate decode per request, shared arm views and asynchronous receipts."""
from contextvars import ContextVar
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import time

from arx_project.artifacts import BackgroundWriter, encode, read_candidate_transport
from arx_project.json_io import read_json, relocate_paths
from arx_project.layout import relocation_aliases

_active = ContextVar('arx_candidate_request', default=None)
REFERENCE = 'arx_candidate_pool_reference_v1'


def directory_key(directory):
    return str(Path(directory).resolve())


def pool_file(directory):
    return Path(directory)/'candidates/candidate_pool.json'


def read_snapshot(directory):
    path = pool_file(directory)
    if not path.is_file() and (Path(directory)/'graspcl/candidate_pool.json').is_file():
        path = Path(directory)/'graspcl/candidate_pool.json'
    if path.is_file():
        raw = path.read_bytes()
        payload = relocate_paths(json.loads(raw), relocation_aliases())
        return payload, {'source': str(path.resolve()), 'snapshot_sha256': hashlib.sha256(raw).hexdigest()}
    reference_path = path.with_name('candidate_pool_reference.json')
    reference = read_json(reference_path)
    if reference['format'] != REFERENCE:
        raise ValueError('Unknown candidate reference format')
    source = Path(reference['source'])
    if not source.is_absolute():
        source = (reference_path.parent/source).resolve()
    raw = source.read_bytes()
    payload = json.loads(raw)
    if reference.get('snapshot_sha256'):
        valid = hashlib.sha256(raw).hexdigest() == reference['snapshot_sha256']
    else:
        valid = hashlib.sha256(encode(payload)).hexdigest() == reference.get('transport_sha256')
    if not valid:
        raise ValueError('Canonical candidate snapshot changed')
    payload = relocate_paths(payload, relocation_aliases())
    end = reference['retained_prefix_count']
    if not isinstance(end, int) or not 0 <= end <= len(payload['candidates']):
        raise ValueError('Candidate reference prefix is invalid')
    return {**payload, 'candidates': payload['candidates'][:end]}, {
        'source': str(source), 'snapshot_sha256': hashlib.sha256(raw).hexdigest()}


class CandidateScope:
    def __init__(self, directory, descriptor=None):
        self.root = Path(directory).resolve()
        started = time.perf_counter()
        if descriptor is None:
            self.payload, self.provenance = read_snapshot(self.root)
            self.source = 'single_snapshot_read'
        else:
            self.payload = read_candidate_transport(self.root, descriptor)
            self.provenance = {'source': str(pool_file(self.root)),
                'transport_sha256': descriptor['sha256'], 'commit_owner_pid': descriptor['owner_pid']}
            self.source = 'shared_memory'
        if not isinstance(self.payload.get('candidates'), list):
            raise ValueError('Candidate payload requires its ordered candidate list')
        self.initial_digest = hashlib.sha256(encode(self.payload)).hexdigest()
        self.views = {directory_key(self.root): self.payload}
        self.writer = BackgroundWriter()
        self.started = started
        self.load_seconds = time.perf_counter()-started
        self.accesses = 0
        self.view_records = {}
        self.token = None

    def __enter__(self):
        if _active.get() is not None:
            raise RuntimeError('Candidate requests cannot overlap in the same planner context')
        self.token = _active.set(self)
        return self

    def view(self, directory):
        self.accesses += 1
        name = directory_key(directory)
        if name not in self.views:
            raise RuntimeError('Arm candidate view has not been bound: '+name)
        return self.views[name]

    def bind(self, directory, source_directory):
        directory = Path(directory).resolve()
        if not directory.is_relative_to(self.root):
            raise ValueError('Arm candidate view escapes its observation')
        payload = self.view(source_directory)
        name = directory_key(directory)
        if name in self.views and self.views[name] is not payload:
            raise RuntimeError('Arm candidate view cannot be rebound to another pool')
        self.views[name] = payload
        self.record_view(directory, len(payload['candidates']))

    def record_view(self, directory, count, bound=None):
        self.view_records[directory_key(directory)] = {
            'format': REFERENCE, 'source': str(pool_file(self.root)),
            **self.provenance, 'retained_prefix_count': count,
            'total_candidates': len(self.payload['candidates']),
            **({'search_bound': bound} if bound is not None else {})}

    def limit(self, directory, count, bound):
        payload = self.view(directory)
        if not isinstance(count, int) or not 0 <= count <= len(payload['candidates']):
            raise ValueError('Candidate prefix bound is outside its source pool')
        self.views[directory_key(directory)] = {**payload,
            'candidates': payload['candidates'][:count], 'search_bound': dict(bound)}
        self.record_view(directory, count, dict(bound))

    @contextmanager
    def window(self, directory, start, stop):
        """Temporarily expose an ordered interval without copying its candidates."""
        payload = self.view(directory)
        if (type(start) is not int or type(stop) is not int
                or not 0 <= start < stop <= len(payload['candidates'])):
            raise ValueError('Candidate window is outside its source pool')
        key = directory_key(directory)
        self.views[key] = {**payload, 'candidates':payload['candidates'][start:stop]}
        try:
            yield
        finally:
            self.views[key] = payload

    def __exit__(self, kind, error, traceback):
        try:
            if hashlib.sha256(encode(self.payload)).hexdigest() != self.initial_digest:
                raise RuntimeError('Planning modified the canonical candidate pool')
            # Shared-memory source snapshots may still be writing in the producer.
            # The transport digest identifies that exact immutable content.
            for directory, record in self.view_records.items():
                self.writer.submit(Path(directory)/'candidates/candidate_pool_reference.json', record)
            self.writer.flush()
            self.writer.submit(self.root/'candidate_io.json', {
                'source': self.source, 'candidate_count': len(self.payload['candidates']),
                'decode_count': 1, 'candidate_accesses': self.accesses,
                'full_arm_pool_copies': 0, 'load_seconds': self.load_seconds,
                'canonical_content_sha256': self.initial_digest,
                'candidate_order_poses_and_scores_unchanged': True,
                'recording': 'background_thread_with_request_completion_barrier'})
            self.writer.flush()
        finally:
            try:
                self.writer.close()
            finally:
                if self.token is not None:
                    _active.reset(self.token)


def candidate_scope(directory, descriptor=None):
    return CandidateScope(directory, descriptor)


def candidate_pool(directory):
    scope = _active.get()
    return scope.view(directory) if scope is not None else read_snapshot(directory)[0]


def bind_candidate_view(directory, source_directory):
    scope = _active.get()
    if scope is None:
        raise RuntimeError('Candidate binding requires one active planner request')
    scope.bind(directory, source_directory)


def limit_candidate_view(directory, count, bound):
    scope = _active.get()
    if scope is None:
        raise RuntimeError('Candidate prefixing requires one active planner request')
    scope.limit(directory, count, bound)


def candidate_window(directory, start, stop):
    scope = _active.get()
    if scope is None:
        raise RuntimeError('Candidate windows require one active planner request')
    return scope.window(directory, start, stop)
