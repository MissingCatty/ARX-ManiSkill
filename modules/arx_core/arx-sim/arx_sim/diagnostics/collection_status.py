"""Read-only collection telemetry. Does not author any control or USD state."""
import hashlib
import os
from pathlib import Path
import time
import uuid


def identity():
    paths = {'initial_setup': os.environ.get('ARX_HISTORICAL_SETUP'),
             'catalog': os.environ.get('ARX_TABLETOP_CATALOG'),
             'setup': str(Path(os.environ['ARX_REPRO_REFERENCE'])/'setup.json')
             if os.environ.get('ARX_REPRO_REFERENCE') else None}
    return {'session_id': uuid.uuid4().hex, 'pid': os.getpid(),
            'gpu_uuid': os.environ.get('ARX_SIM_GPU_UUID'),
            'instrumentation_sha256': {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ('curobo_runner.py','lilo_observation.py','research_fingerprint.py','collection_status.py')},
            'input_sha256': {key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                             if path and Path(path).is_file() else None
                             for key, path in paths.items()}}


def snapshot(bridge):
    active = getattr(bridge, 'active', None)
    pending = getattr(bridge, 'research_pending_capture', None)
    return {'version': 'arx_collection_status_v1', **bridge.collection_identity,
            'wall_time': time.time(),
            'active_trajectory_request_id': active['request']['id'] if active else None,
            'controller_busy': bridge.context.controller.active is not None,
            'setup_pending': pending['request']['id'] if pending else None,
            'streaming': bool(bridge.streaming),
            'renderer': bridge.context.config['simulation']['renderer'],
            'recorders': {name: str(getattr(bridge, attribute)['directory'])
                          if getattr(bridge, attribute, None) is not None else None
                          for name, attribute in (('teacher', 'lilo_recording'), ('video', 'video_recording'))}}
