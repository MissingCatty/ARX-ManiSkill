"""Standalone stdlib sidecar for registered ARX collectors (no robot imports).

Run this separately, preferably as a user systemd service. Process death is
recoverable; manual STOP, expired budgets and broken evidence are not retries.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


AUTO_STOP = 'watchdog: unexpected process exit'
AUTO_REASONS = {AUTO_STOP, 'infrastructure/record/resource fault'}
FINAL = {'quota_complete', 'budget_exhausted'}
INTEGRITY_ERRORS = (
    'registered input/source changed', 'frozen physical configuration mismatch',
    'cross-environment', 'missing evidence', 'without registered command request',
    'shared raw writer failed', 'identity mismatch', 'recording-quality',
    'source frame count', 'changed after data acceptance',
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def exclusive_stop(path, value):
    try:
        with Path(path).open('x') as stream:
            json.dump(value, stream); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno())
        return True
    except FileExistsError:
        return False


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def storage_ready(cfg, mountinfo='/proc/self/mountinfo'):
    """Do not mistake the root filesystem beneath an unmounted data disk for it."""
    required = cfg.get('required_bulk_mount')
    if required:
        matches = []
        for line in Path(mountinfo).read_text().splitlines():
            before, after = line.split(' - ', 1)
            fields = before.split(); filesystem = after.split()[0]
            if fields[4] == required['target']:
                matches.append(filesystem)
        if required['fstype'] not in matches:
            return False
    marker = cfg.get('storage_readiness_marker')
    if marker:
        try:
            if Path(marker['path']).read_text() != marker['contents']:
                return False
        except OSError:
            return False
    return True


def process_info(pid):
    """A zombie is dead; UID, command and boot/start ticks prevent PID reuse."""
    if int(pid) <= 0:
        return None
    proc = Path('/proc') / str(pid)
    try:
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        args = (proc / 'cmdline').read_bytes().split(b'\0')
        return {'pid': int(pid), 'uid': proc.stat().st_uid, 'start_ticks': int(fields[19]),
                'boot_id': boot_id(), 'argv': [s.decode(errors='replace') for s in args if s]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def owned_process(meta, campaign, role, remembered=None):
    info = process_info(meta['pid'])
    if not info or info['uid'] != os.getuid():
        return None
    args = info['argv']
    module = ('scripts.curobo_arx.data_factory.launch' if role == 'coordinator'
              else 'scripts.curobo_arx.data_factory.workers' if role == 'worker'
              else 'podman')
    scoped = str(Path(campaign).resolve())
    if role == 'group':
        valid = (any(Path(a).name == 'podman' for a in args[:1]) and
                 meta['container'] in args and scoped in args)
    else:
        valid = module in args and (scoped in args or str(Path(scoped) / 'registration.json') in args)
    if not valid:
        return None
    expected = remembered or meta
    if any(k in expected and expected[k] != info[k] for k in ('start_ticks', 'boot_id')):
        return None
    return info


def group_records(campaign):
    campaign = Path(campaign)
    for path in sorted((campaign / 'group_registrations').glob('*.json')):
        if path.name.endswith('.process.json'):
            continue
        group = read(path)
        if group['group_id'] != path.stem:
            raise ValueError('Group registration identity mismatch')
        if group['registration_sha256'] != sha(campaign / 'registration.json'):
            raise ValueError('Group registration belongs to a different campaign')
        yield group, path.with_name(path.stem + '.process.json')


def container_running(name):
    result = subprocess.run(['podman', 'container', 'exists', name], capture_output=True, timeout=15)
    if result.returncode == 1:
        return None  # Positively absent.
    if result.returncode != 0:
        raise RuntimeError('Container inventory is unavailable; absence is unverified')
    result = subprocess.run(['podman', 'inspect', '--type', 'container', '--format',
                             '{{.State.Running}}', name], capture_output=True, text=True, timeout=15)
    if result.returncode or result.stdout.strip() not in ('true', 'false'):
        raise RuntimeError('Container state is unverified: ' + name)
    return result.stdout.strip() == 'true'


def manual_stop(campaign):
    path = Path(campaign) / 'STOP'
    if not path.exists():
        return False
    try:
        return read(path).get('reason') not in AUTO_REASONS
    except (ValueError, AttributeError):
        return True


def fatal_evidence(campaign):
    paths = [Path(campaign) / 'FAULT.json']
    paths += list((Path(campaign) / 'workers').glob('*/FAULT.json'))
    paths += list((Path(campaign) / 'groups').glob('*/FAULT.json'))
    for path in paths:
        if path.exists():
            error = str(read(path).get('error', '')).lower()
            if any(token in error for token in INTEGRITY_ERRORS):
                return str(path)
    return None


class Watchdog:
    def __init__(self, config):
        self.config_path = Path(config).resolve()
        self.cfg = read(config)
        self.home = self.config_path.parent
        self.state_path = self.home / 'state.json'
        self.state = read(self.state_path) if self.state_path.exists() else {
            'campaign': self.cfg['initial_campaign'], 'phase': 'watching',
            'restart_times': [], 'consecutive_bad': 0, 'identities': {}}
        if self.state['phase'] == 'watchdog_exit':
            self.state['phase'] = self.state.pop('resume_phase', 'watching')
        if (self.state['phase'] == 'draining' and
                self.state.get('drain_boot_id') != boot_id()):
            self.state.update(drain_started=time.time(), drain_boot_id=boot_id())
        self.terminate = False

    def event(self, event, **details):
        with (self.home / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps({'wall': time.time(), 'event': event, **details}) + '\n')
            stream.flush(); os.fsync(stream.fileno())

    def save(self, **details):
        self.state.update(details, checked_unix=time.time(), watchdog_pid=os.getpid())
        write(self.state_path, self.state)

    def alive(self, path, role):
        path = Path(path)
        if not path.is_file():
            return None
        meta = read(path)
        info = owned_process(meta, self.state['campaign'], role, self.state['identities'].get(str(path)))
        if info:
            self.state['identities'][str(path)] = {k: info[k] for k in ('start_ticks', 'boot_id')}
        return info

    def health(self):
        p = Path(self.state['campaign'])
        if manual_stop(p):
            return 'manual_stop', 'Collection STOP was requested by the user'
        if not storage_ready(self.cfg):
            return 'resource_wait', 'Registered data mount/readiness marker is not available'
        state = read(p / 'state.json')
        launch = read(p / 'launch.json') if (p / 'launch.json').exists() else {}
        reg = read(p / 'registration.json')
        deadline = launch.get('stop_unix', reg.get('deadline_unix', float('inf')))
        if state['status'] in FINAL or time.time() >= deadline:
            return 'complete', 'Original task ended or original deadline expired'
        fatal = fatal_evidence(p)
        if fatal:
            return 'blocked', 'Configuration/record integrity fault: ' + fatal
        if not self.alive(p / 'coordinator_process.json', 'coordinator'):
            return 'failed', 'Registered coordinator exited'
        startup = time.time() - launch.get('started_unix', time.time()) < self.cfg.get('worker_startup_grace_seconds', 240)
        for kind in ('planner', reg['candidate_backend']):
            meta = p / 'workers' / kind / 'process.json'
            if meta.exists() and not self.alive(meta, 'worker'):
                return 'failed', 'Registered worker exited: ' + kind
            if not meta.exists() and not startup:
                return 'failed', 'Registered worker was never created: ' + kind
        for group, meta in group_records(p):
            directory = p / 'groups' / group['group_id']
            if (directory / 'block_result.json').exists() and read(directory / 'block_result.json').get('idle') is True:
                continue  # Normal object/renderer rotation.
            if not meta.exists():
                continue  # Coordinator may be seeding the cache before Popen.
            if not self.alive(meta, 'group'):
                return 'failed', 'Registered simulator process exited: ' + group['group_id']
            if time.time() - meta.stat().st_mtime > 60 and container_running('arx-data-factory-' + group['group_id']) is not True:
                return 'failed', 'Registered simulator container exited: ' + group['group_id']
        # Slow initialization or zero new successes is not process death.
        return 'healthy', 'Coordinator, resident workers and dispatched simulator processes are alive'

    def begin_recovery(self, reason):
        ident = 'incident_' + str(time.time_ns())
        incident = self.home / ident
        incident.mkdir()
        write(incident / 'trigger.json', {'wall': time.time(), 'campaign': self.state['campaign'], 'reason': reason})
        exclusive_stop(Path(self.state['campaign']) / 'STOP',
            {'reason': AUTO_STOP, 'wall': time.time(), 'watchdog': str(self.home), 'incident': str(incident)})
        self.event('unexpected_exit', campaign=self.state['campaign'], reason=reason, incident=str(incident))
        self.save(phase='draining', incident=str(incident), drain_started=time.time(),
                  drain_boot_id=boot_id(), signalled=[], worker_stop_started=None)

    def drain(self):
        p = Path(self.state['campaign'])
        if manual_stop(p):
            self.save(phase='manual_stop'); return
        if not storage_ready(self.cfg):
            self.save(health='resource_wait', reason='Waiting for the registered data mount before inspecting old execution')
            return
        if time.time() - self.state['drain_started'] > self.cfg.get('drain_seconds', 1080):
            self.save(phase='blocked', reason='Old execution did not exit; no second virtual World will be started')
            return
        if self.alive(p / 'coordinator_process.json', 'coordinator'):
            return  # The existing coordinator drains its own actions.
        containers = []
        for group, meta in group_records(p):
            name = 'arx-data-factory-' + group['group_id']
            running = container_running(name)
            containers.append((name, running))
            if running is not None and name not in self.state['signalled']:
                if running:
                    subprocess.run(['podman', 'kill', '--signal', 'TERM', name], capture_output=True, timeout=15)
                else:
                    # Only an already stopped, exactly registered container.
                    subprocess.run(['podman', 'rm', name], capture_output=True, timeout=15)
                self.state['signalled'].append(name)
        if any(value is not None for _, value in containers):
            return
        alive = []
        for path in (p / 'workers').glob('*/process.json'):
            exclusive_stop(path.parent / 'STOP', {'reason': AUTO_STOP, 'wall': time.time()})
            info = self.alive(path, 'worker')
            if info:
                alive.append(info)
        if alive:
            started = self.state.get('worker_stop_started')
            if started is None:
                self.state['worker_stop_started'] = time.time()
            elif time.time() - started > 30:
                for info in alive:
                    # Validate identity again immediately before signalling.
                    fresh = process_info(info['pid'])
                    if fresh and all(fresh[k] == info[k] for k in ('uid', 'start_ticks', 'boot_id', 'argv')):
                        os.kill(info['pid'], signal.SIGTERM)
            return
        for group, meta in group_records(p):
            if meta.exists() and self.alive(meta, 'group'):
                return
        self.save(phase='waiting_resources', retry_after=time.time() + self.cfg.get('backoff_seconds', 60))

    def recover(self):
        now = time.time()
        if now < self.state.get('retry_after', 0):
            return
        history = [t for t in self.state['restart_times'] if now - t < 3600]
        if len(history) >= self.cfg.get('max_restarts_per_hour', 4):
            self.save(retry_after=history[0] + 3600, reason='Restart rate limit; waiting without dropping evidence')
            return
        p = Path(self.state['campaign'])
        if manual_stop(p):
            self.save(phase='manual_stop'); return
        deadline = (read(p / 'launch.json')['stop_unix'] if (p / 'launch.json').exists()
                    else read(p / 'registration.json')['deadline_unix'])
        if now >= deadline:
            self.save(phase='complete'); return
        request = Path(self.state['incident']) / 'recovery_request.json'
        result_path = request.with_name('recovery_result.json')
        if result_path.exists() and read(result_path).get('status') == 'prepared':
            # Adopt a completed helper after the watchdog itself was restarted.
            self.save(phase='prepared', prepared_campaign=read(result_path)['campaign'])
            return
        out = Path(self.cfg['project_root']) / 'runs' / (self.cfg['campaign_prefix'] + '_' + str(time.time_ns()))
        write(request, {'parent': str(p), 'out': str(out), 'watchdog_config': str(self.config_path),
                        'result': str(request.with_name('recovery_result.json'))})
        command = [self.cfg['collector_python'], str(Path(__file__).with_name('watchdog_recovery.py')), str(request)]
        try:
            with request.with_name('recovery.log').open('a') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        except subprocess.TimeoutExpired:
            self.save(phase='blocked', reason='Recovery preparation timed out; preserve its directory and inspect recovery.log')
            return
        receipt = read(request.with_name('recovery_result.json')) if request.with_name('recovery_result.json').exists() else {}
        if result.returncode or receipt.get('status') == 'blocked':
            self.save(phase='blocked', reason=receipt.get('error', 'Recovery helper failed; inspect recovery.log'))
        elif receipt.get('status') == 'resource_wait':
            self.save(retry_after=now + self.cfg.get('resource_poll_seconds', 60), reason=receipt['reason'])
        elif receipt.get('status') == 'manual_stop':
            self.save(phase='manual_stop')
        elif receipt.get('status') == 'complete':
            self.save(phase='complete')
        elif receipt.get('status') == 'prepared':
            self.save(phase='prepared', prepared_campaign=receipt['campaign'], restart_times=history)
        else:
            self.save(phase='blocked', reason='Recovery result is incomplete')

    def start_prepared(self):
        out = Path(self.state['prepared_campaign'])
        if manual_stop(self.state['campaign']) or (self.home / 'STOP').exists():
            self.save(phase='manual_stop'); return
        if (out / 'BOOTSTRAP_FAULT.json').exists():
            self.save(phase='blocked', reason='Collector bootstrap failed; inspect BOOTSTRAP_FAULT.json')
            return
        meta = out / 'coordinator_process.json'
        if (out / 'launch.json').exists() or (meta.exists() and owned_process(read(meta), out, 'coordinator')):
            self.event('restarted', parent=self.state['campaign'], campaign=str(out))
            self.save(campaign=str(out), phase='watching', consecutive_bad=0, identities={},
                      restart_times=self.state['restart_times'] + [time.time()])
            return
        intent = out / 'watchdog_launch_intent.json'
        if intent.exists() and time.time() - intent.stat().st_mtime < 60:
            return
        write(intent, {'wall': time.time(), 'config': str(self.config_path)})
        command = [sys.executable, str(Path(__file__).resolve()), '--config', str(self.config_path), '--exec-collector', str(out)]
        with (out / 'coordinator.log').open('a') as log:
            subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)

    def step(self):
        if (self.home / 'STOP').exists():
            self.save(phase='manual_stop', reason='Watchdog explicitly stopped'); return False
        if self.terminate:
            self.save(phase='watchdog_exit', resume_phase=self.state['phase'],
                      reason='Watchdog received an exit signal; no explicit persistent STOP was requested')
            return False
        phase = self.state['phase']
        if phase == 'watching':
            action, reason = self.health()
            bad = self.state['consecutive_bad'] + 1 if action == 'failed' else 0
            self.save(health=action, reason=reason, consecutive_bad=bad)
            if action == 'failed' and bad >= self.cfg.get('failure_confirmations', 2):
                self.begin_recovery(reason)
            elif action in ('manual_stop', 'complete', 'blocked'):
                self.save(phase=action)
        elif phase == 'draining':
            self.drain()
        elif phase == 'waiting_resources':
            self.recover()
        elif phase == 'prepared':
            self.start_prepared()
        self.save()
        return self.state['phase'] not in ('manual_stop', 'complete')

    def run(self):
        with (self.home / 'watchdog.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._run_locked()

    def _run_locked(self):
        def stop(*_):
            self.terminate = True
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        self.event('watchdog_attached', campaign=self.state['campaign'], pid=os.getpid())
        augmentation = None
        if self.cfg.get('augmentation'):
            expected=self.cfg['augmentation'].get('sha256')
            if expected and sha(self.cfg['augmentation']['config'])!=expected:
                raise ValueError('Registered augmentation configuration changed')
            try:
                from .augmentation_watchdog import AugmentationWatchdog
            except ImportError:
                from augmentation_watchdog import AugmentationWatchdog
            augmentation = AugmentationWatchdog(self.cfg['augmentation']['config'], self.config_path,
                lambda: self.terminate or (self.home / 'STOP').exists())
            # Native recovery may wait on full record hashes. Keep the other
            # queue monitored throughout that wait without sharing its Worlds.
            thread = threading.Thread(target=augmentation.run, daemon=True, name='seed-augmentation-monitor')
            thread.start()
        native_terminal = False
        while True:
            try:
                if not native_terminal and not self.step():
                    native_terminal = True
                if self.terminate or (self.home / 'STOP').exists():
                    break
                if native_terminal and (augmentation is None or augmentation.state['phase'] in ('complete', 'manual_stop')):
                    break
                if augmentation is not None and not thread.is_alive():
                    self.event('augmentation_monitor_exited')
                    raise SystemExit(1)  # systemd reattaches monitors; collectors survive.
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                # Unknown ownership/inventory/record conditions never trigger a launch.
                self.event('check_error', error=str(error))
                self.save(health='unverified', last_check_error=str(error))
            for _ in range(self.cfg.get('poll_seconds', 20)):
                if self.terminate or (self.home / 'STOP').exists():
                    break
                time.sleep(1)


def exec_collector(config, campaign):
    cfg = read(config); campaign = Path(campaign).resolve()
    lock = (campaign / 'watchdog_spawn.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    parent = read(campaign / 'registration.json').get('resume', {}).get('parent')
    if ((Path(config).parent / 'STOP').exists() or (campaign / 'launch.json').exists()
            or (parent and manual_stop(parent))):
        return
    existing = campaign / 'coordinator_process.json'
    if existing.exists() and owned_process(read(existing), campaign, 'coordinator'):
        return
    command = [cfg['collector_python'], '-m', 'scripts.curobo_arx.data_factory.launch', str(campaign),
               '--compute-gpu', cfg['compute_uuid'], '--sim-gpus', *cfg['sim_uuids']]
    info = process_info(os.getpid())
    write(existing, {'pid': os.getpid(), 'start_ticks': info['start_ticks'], 'boot_id': info['boot_id'],
        'command': command, 'cwd': cfg['project_root'], 'owned': True, 'wall': time.time(), 'mode': 'watchdog_recovery'})
    # Keep spawn exclusivity across exec, even if the watchdog itself restarts.
    os.set_inheritable(lock.fileno(), True)
    os.chdir(cfg['project_root'])
    env = os.environ.copy(); env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    os.execve(command[0], command, env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--exec-collector')
    parser.add_argument('--exec-augmentation')
    parser.add_argument('--exec-aug-job')
    parser.add_argument('--stop', action='store_true', help='Explicit persistent monitor stop; never inferred from SIGTERM')
    args = parser.parse_args()
    if args.stop:
        exclusive_stop(Path(args.config).resolve().parent / 'STOP',
                       {'reason': 'User explicitly stopped watchdog', 'wall': time.time()})
    elif args.exec_augmentation or args.exec_aug_job:
        try:
            from augmentation_watchdog import bootstrap
            bootstrap(args.config, campaign=args.exec_augmentation, job=args.exec_aug_job)
        except BlockingIOError:
            pass
        except Exception as error:
            fault = {'wall': time.time(), 'error': repr(error)}
            if args.exec_augmentation:
                write(Path(args.exec_augmentation) / 'BOOTSTRAP_FAULT.json', fault)
            else:
                write(read(args.exec_aug_job)['result'], {'status': 'blocked', **fault})
            raise
    elif args.exec_collector:
        try:
            exec_collector(args.config, args.exec_collector)
        except BlockingIOError:
            pass  # A previous bootstrap/collector still owns the spawn lock.
        except Exception as error:
            write(Path(args.exec_collector) / 'BOOTSTRAP_FAULT.json',
                  {'wall': time.time(), 'error': repr(error)})
            raise
    else:
        Watchdog(args.config).run()
