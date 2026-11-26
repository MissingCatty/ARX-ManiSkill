"""Follow the bounded pilot and advance only after independent latency gates."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback

from arx_project.layout import ROOT, ensure_run_path, now, write_json
from .followup_state import completed_controller
from .serialization_report import report
from .serialization_review import review
from .validation_followup import publish
from .validation_launch import controller_unit, watch
from .validation_replay import prepare


def raw(path): return json.loads(Path(path).read_text())


def wait_for_result(run, status):
    expected = raw(run/'registration.json')['planning_requests']
    while True:
        progress = raw(run/'progress.json')
        write_json(status, {'at': now(), 'phase': 'following', 'run': str(run.relative_to(ROOT)),
            **{k: progress.get(k) for k in ['completed_requests', 'total_requests', 'current']}})
        if (run/'STOP').exists(): raise RuntimeError('Serialization run stopped; keep original records')
        if (run/'controller.json').exists():
            output = subprocess.check_output(['systemctl', '--user', 'show', controller_unit(run),
                '-p', 'MainPID', '-p', 'ActiveState', '-p', 'ExecMainStatus'], text=True)
            fields = dict(line.split('=', 1) for line in output.strip().splitlines())
            if completed_controller(fields, lambda: raw(run/'progress.json'), expected) is not None:
                return review(run)
        time.sleep(5)


def pointer(run, phase):
    path = ROOT/'tmp/indexed_grasp_planning/funnel_active.json';value = raw(path)
    value.update(at=now(), phase=phase, binding_serialization_run=str(run.relative_to(ROOT)))
    write_json(path, value)


def main(root, pilot):
    if root.exists(): raise ValueError('Preserve previous serialization follow-up registrations')
    root.mkdir(parents=True);status = root/'status.json'
    write_json(root/'registration.json', {'at': now(), 'pilot': str(pilot.relative_to(ROOT)),
        'full_config': 'configs/experiments/indexed_grasp_binding_serialization_full.json',
        'maximum_additional_planning_requests_after_pilot': 138, 'physical_grasps': 0})
    checked = wait_for_result(pilot, status);report(pilot)
    if not checked['timing_gate_passed']:
        pointer(pilot, 'binding_serialization_pilot_complete');publish(root)
        write_json(status, {'at': now(), 'phase': 'complete', 'full_started': False,
            'reason': 'pilot_latency_gates_not_all_passed', 'independent_review_passed': True})
        return
    full = ROOT/'tmp/indexed_grasp_planning/yaml69_20261126_01'
    prepare(full, ROOT/'configs/experiments/indexed_grasp_binding_serialization_full.json');watch(full)
    pointer(full, 'binding_serialization_full_running');publish(root)
    checked = wait_for_result(full, status);report(full)
    pointer(full, 'binding_serialization_full_complete');publish(root)
    write_json(status, {'at': now(), 'phase': 'complete', 'full_started': True,
        'checked_requests': checked['checked_requests'], 'checked_endpoints': checked['checked_endpoints'],
        'timing_gate_passed': checked['timing_gate_passed'], 'physical_grasps': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True);parser.add_argument('--pilot', type=Path, required=True)
    args = parser.parse_args();root = ensure_run_path(args.root)
    try: main(root, ensure_run_path(args.pilot))
    except BaseException:
        write_json(root/'STOP', {'at': now(), 'error': traceback.format_exc(), 'automatic_retry': False})
        raise
