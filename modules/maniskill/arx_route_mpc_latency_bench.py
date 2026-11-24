"""Keep smooth MPC control while removing synchronous video and counter writes."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import arx_route_mpc_smooth_bench as smooth
import native_pair_timing as timing
import resident_backend as service

SELF = Path(__file__).resolve()
benchmark = smooth.benchmark
ROLLING_REQUESTS = ('route_step', 'step')
MAX_PENDING_FRAMES = 4


def substitute(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Frozen benchmark source does not match the expected version')
    return source.replace(old, new)


def low_latency_server():
    """Reuse service startup and request handling with checkpoints at phase boundaries."""
    source = inspect.getsource(service.serve)
    source = substitute(source, "                    command = message.pop('command', 'request')",
                         "                    received_ns = time.monotonic_ns()\n"
                         "                    command = message.pop('command', 'request')")
    source = substitute(source, "                    elif command == 'shutdown':",
                         "                    elif command == 'shutdown':\n"
                         "                        save(root / (args.kind + '_ready.json'), ready)")
    source = substitute(source, "                        save(root / (args.kind + '_ready.json'), ready)\n                    else:",
                         """                        checkpoint_started = time.monotonic_ns()
                        checkpoint = message.get('mpc') not in ROLLING_REQUESTS or response['status'] != 'ok'
                        if checkpoint:
                            save(root / (args.kind + '_ready.json'), ready)
                        if message.get('mpc'):
                            response['transport'] = {
                                'received_ns': received_ns,
                                'reply_ready_ns': time.monotonic_ns(),
                                'counter_checkpoint': checkpoint,
                                'counter_checkpoint_seconds': (time.monotonic_ns() - checkpoint_started) / 1e9,
                            }
                    else:""")
    namespace = {**service.__dict__, 'ROLLING_REQUESTS': ROLLING_REQUESTS}
    exec(compile(source, str(SELF) + ':serve', 'exec'), namespace)
    return namespace['serve']


class VideoQueue:
    def __init__(self):
        self.encoder = ThreadPoolExecutor(max_workers=1, thread_name_prefix='arx-video')
        self.pending = deque()

    def append(self, writer, image):
        if len(self.pending) == MAX_PENDING_FRAMES:
            self.pending.popleft().result()
        # Copy before giving the array to another thread. The renderer can reuse its memory.
        self.pending.append(self.encoder.submit(writer.append_data, image.copy()))

    def finish(self):
        self.encoder.shutdown(wait=True)
        for frame in self.pending:
            frame.result()


def record_frames(execution):
    env = execution.env
    if execution.sensor_writer is None or env._hidden_objects:
        execution.video.append(execution.writer, env.render().cpu().numpy()[0])
        if execution.sensor_writer is not None:
            obs = env.get_obs()
            rgb = obs['sensor_data'][env.perception_camera]['rgb']
            execution.video.append(execution.sensor_writer, rgb.cpu().numpy()[0])
        return
    scene = env.scene
    scene.update_render(update_sensors=True, update_human_render_cameras=True)
    overview = scene.get_human_render_camera_images('overview')['overview']
    camera = scene.sensors[env.perception_camera]
    camera.capture()
    wrist = camera.get_obs(depth=False, position=False, segmentation=False)['rgb']
    execution.video.append(execution.writer, overview.cpu().numpy()[0])
    execution.video.append(execution.sensor_writer, wrist.cpu().numpy()[0])


def video_timing_execute():
    source = inspect.getsource(timing.execute)
    source = substitute(source, '            super().__init__(*args,**kwargs)',
                         '            super().__init__(*args,**kwargs)\n            self.video = VideoQueue()')
    start = source.index('                self.writer.append_data(')
    end = source.index("                totals['recording_capture_encode_seconds']", start)
    source = source[:start] + '                record_frames(self)\n' + source[end:]
    source = substitute(source, '                return super().finish()',
                         '                try:\n                    self.video.finish()\n'
                         '                finally:\n                    super().finish()')
    namespace = {**timing.__dict__, 'VideoQueue': VideoQueue, 'record_frames': record_frames}
    exec(compile(source, str(SELF) + ':execute', 'exec'), namespace)
    return namespace['execute']


@contextmanager
def capture_rpc_timing(trial):
    records = []
    rpc = smooth.mpc.rpc

    def measured_rpc(address, message, timeout=1800):
        started_ns = time.monotonic_ns()
        reply = rpc(address, message, timeout=timeout)
        finished_ns = time.monotonic_ns()
        if 'transport' in reply:
            transport = reply['transport']
            records.append({'request': message['mpc'], 'phase': message.get('phase', 'wrist'),
                            'round_trip_seconds': (finished_ns - started_ns) / 1e9,
                            'to_server_seconds': (transport['received_ns'] - started_ns) / 1e9,
                            'service_seconds': reply['wall_seconds'],
                            'counter_checkpoint': transport['counter_checkpoint'],
                            'counter_checkpoint_seconds': transport['counter_checkpoint_seconds'],
                            'to_client_seconds': (finished_ns - transport['reply_ready_ns']) / 1e9})
        return reply

    try:
        with patch.object(smooth.mpc, 'rpc', measured_rpc):
            yield
    finally:
        if records:
            with (trial / 'mpc_rpc_timing.jsonl').open('x') as trace:
                trace.writelines(json.dumps(record) + '\n' for record in records)


def prepare(out, backend_gpu, scene_gpu, cases):
    source = substitute(inspect.getsource(smooth.prepare), "'_SMOOTH01'", "'_LATENCY01'")
    namespace = {**smooth.__dict__, 'SELF': SELF}
    exec(compile(source, str(SELF) + ':prepare', 'exec'), namespace)
    namespace['prepare'](out, backend_gpu, scene_gpu, cases)
    reference = SELF.parent / 'outputs/arx_graspcl_route_mpc_smooth_20261118_01/route_mpc_results.json'
    assert benchmark.read(reference)['verified']
    dependencies = [Path(m.__file__).resolve() for m in (smooth, service, timing)]
    for path in dependencies:
        shutil.copy2(path, out / 'source' / path.name)
    for path in (out / 'registration.json', out / benchmark.bench.ARM / 'registration.json'):
        registration = benchmark.read(path)
        registration['source_hashes'].update({str(p): benchmark.sha(p) for p in [*dependencies, reference]})
        registration['runtime_optimization'] = {
            'reference': str(reference),
            'rolling_counter_checkpoint': 'phase boundary, service error, or shutdown',
            'rolling_timing_clock': 'monotonic_ns',
            'video_pending_frames': MAX_PENDING_FRAMES,
            'recording': 'same cameras and cadence, one render refresh and RGB-only wrist for video',
        }
        benchmark.save(path, registration)


def main():
    stage = sys.argv[1]
    if stage == 'prepare':
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--output', type=Path, required=True)
        parser.add_argument('--backend-gpu', required=True)
        parser.add_argument('--scene-gpu', required=True)
        parser.add_argument('--case', action='append', default=[])
        args = parser.parse_args(sys.argv[2:])
        prepare(args.output.resolve(), args.backend_gpu, args.scene_gpu, args.case or ['I0117'])
        return
    smooth.route.RouteMPC = smooth.SmoothRouteMPC
    benchmark.SELF = SELF
    if stage == 'serve' and sys.argv[sys.argv.index('--kind') + 1] == 'curobo':
        with patch.object(service, 'serve', low_latency_server()):
            benchmark.main()
    elif stage == 'execute':
        trial = Path(sys.argv[sys.argv.index('--output') + 1]).resolve()
        with patch.object(timing, 'execute', video_timing_execute()), capture_rpc_timing(trial):
            benchmark.main()
    else:
        benchmark.main()


if __name__ == '__main__':
    main()
