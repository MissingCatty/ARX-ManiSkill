"""Bounded layout pilot and conditional full comparison, without retries."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback

from arx_project.layout import ROOT, ensure_run_path, now, sha256, write_json
from .followup_state import completed_controller
from .layout_registration import augment
from .layout_report import report
from .layout_review import review
from .validation_followup import publish
from .validation_launch import controller_unit, watch
from .validation_replay import prepare


def raw(path): return json.loads(Path(path).read_text())


def wait(run, status):
    reg = raw(run/'registration.json')
    while True:
        progress = raw(run/'progress.json')
        write_json(status, {'at': now(), 'phase': 'following', 'run': str(run.relative_to(ROOT)),
            **{k: progress.get(k) for k in ['completed_requests', 'total_requests', 'current']}})
        if (run/'STOP').exists(): raise RuntimeError('Collision-layout run stopped; preserve its records')
        if (run/'controller.json').exists():
            values = subprocess.check_output(['systemctl', '--user', 'show', controller_unit(run),
                '-p', 'MainPID', '-p', 'ActiveState', '-p', 'ExecMainStatus'], text=True, timeout=20)
            fields = dict(line.split('=', 1) for line in values.strip().splitlines())
            if completed_controller(fields, lambda: raw(run/'progress.json'), reg['planning_requests']) is not None:
                return review(run)
        time.sleep(5)


def pointer(run, phase, root):
    path = ROOT/'tmp/indexed_grasp_planning/funnel_active.json';active = raw(path)
    active.update(at=now(), phase=phase, layout_run=str(run.relative_to(ROOT)),
                  layout_followup=str(root.relative_to(ROOT)))
    write_json(path, active)


def main(root, pilot):
    if root.exists(): raise ValueError('Do not duplicate the collision-layout follow-up')
    root.mkdir(parents=True);status = root/'status.json'
    write_json(root/'registration.json', {'at': now(), 'pilot': str(pilot.relative_to(ROOT)),
        'maximum_additional_planning_requests_after_pilot': 138, 'physical_grasps': 0,
        'source_sha256': sha256(__file__), 'automatic_default_activation': False})
    checked = wait(pilot, status);report(root, pilot)
    if not checked['timing_gate_passed']:
        pointer(pilot, 'layout_pilot_complete', root);publish(root)
        write_json(status, {'at': now(), 'phase': 'complete', 'full_started': False,
            'reason': 'pilot_latency_gates_not_all_passed', 'independent_review_passed': True})
        return
    full = ROOT/'tmp/indexed_grasp_planning/layout69_20261126_01'
    prepare(full, ROOT/'configs/experiments/indexed_grasp_kernel_layout_full.json');augment(full);watch(full)
    pointer(full, 'layout_full_running', root);report(root, pilot, full);publish(root)
    checked = wait(full, status);report(root, pilot, full)
    pointer(full, 'layout_full_complete', root);publish(root)
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
