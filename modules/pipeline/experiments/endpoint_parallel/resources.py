"""Resource checks for measuring planner latency without competing GPU jobs."""
import subprocess


def compute_processes():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=5)
    return parse_processes(result.stdout)


def parse_processes(text):
    rows = []
    for line in text.splitlines():
        if not line.strip(): continue
        parts = [value.strip() for value in line.split(',')]
        if len(parts) != 2: raise ValueError('Malformed NVML process inventory')
        # NVML reports [N/A] on a failed card. Keep that card occupied and
        # let the other healthy cards continue through resource admission.
        rows.append({'uuid': parts[0], 'pid': int(parts[1]) if parts[1].isdigit() else None})
    return rows


def foreign_processes(gpu, own_pid, processes):
    return [p['pid'] for p in processes if p['uuid'] == gpu and p['pid'] != own_pid]


def idle_devices(gpus, processes, excluded=()):
    occupied = {p['uuid'] for p in processes}
    return [g for g in gpus if g['uuid'] not in excluded and g['uuid'] not in occupied
            and g['free_mib'] >= 10000 and g['utilization_percent'] == 0]
