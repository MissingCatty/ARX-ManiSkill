"""Bounded pilot, independent review, then optional full planning comparison."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback

from arx_project.layout import ROOT, ensure_run_path, now, write_json
from .followup_state import completed_controller
from .mesh_vote_registration import augment
from .mesh_vote_report import report
from .validation_followup import publish
from .validation_launch import controller_unit, watch
from .validation_replay import prepare
from .validation_review import validate


def raw(path): return json.loads(Path(path).read_text())


def wait(run, status):
    reg = raw(run/'registration.json')
    while True:
        p = raw(run/'progress.json')
        write_json(status, {'at': now(), 'phase': 'following', 'run': str(run.relative_to(ROOT)),
            **{k: p.get(k) for k in ['completed_requests', 'total_requests', 'current']}})
        if (run/'STOP').exists(): raise RuntimeError('Mesh query comparison stopped; preserve original records')
        if (run/'controller.json').exists():
            values = subprocess.check_output(['systemctl', '--user', 'show', controller_unit(run),
                '-p', 'MainPID', '-p', 'ActiveState', '-p', 'ExecMainStatus'], text=True)
            fields = dict(line.split('=', 1) for line in values.strip().splitlines())
            if completed_controller(fields, lambda: raw(run/'progress.json'), reg['planning_requests']) is not None:
                result = validate(run)
                geometry = raw(run/'resident/mesh_vote_geometry_preflight/acceptance.json')
                ready = raw(run/'resident/mesh_vote_readiness.json')
                if not geometry['passed'] or not ready['original_and_fast_full_IK_have_separate_CUDA_graphs']:
                    raise ValueError('Actual geometry or independent graph preflight did not pass')
                for case in reg['cases']:
                    for variant in reg['variants']:
                        receipt = raw(run/'plans'/case['id']/variant/'mesh_vote.json')
                        if receipt['variant'] != reg['variant_modes'][variant] or not all(receipt[k] for k in
                                ['geometry_and_thresholds_unchanged', 'separate_CUDA_graphs', 'original_module_unchanged']):
                            raise ValueError('Measured collision variant lost its actual evidence')
                result.update(exact_mesh_sdf_points=geometry['checked_sdf_points'],
                              original_and_fast_CUDA_graphs_separate=True, profiling_enabled=False)
                write_json(run/'independent_review.json', result)
                return result
        time.sleep(5)


def pointer(run, phase, root):
    path = ROOT/'tmp/indexed_grasp_planning/funnel_active.json';a = raw(path)
    a.update(at=now(), phase=phase, mesh_vote_run=str(run.relative_to(ROOT)),
             mesh_vote_followup=str(root.relative_to(ROOT)))
    write_json(path, a)


def main(root, pilot):
    if root.exists(): raise ValueError('Do not duplicate the mesh query follow-up')
    root.mkdir(parents=True);status = root/'status.json'
    write_json(root/'registration.json', {'at': now(), 'pilot': str(pilot.relative_to(ROOT)),
        'maximum_additional_planning_requests_after_pilot': 138, 'physical_grasps': 0,
        'automatic_default_activation': False})
    checked = wait(pilot, status);report(root, pilot)
    if not checked['timing_gate_passed']:
        pointer(pilot, 'mesh_vote_pilot_complete', root);publish(root)
        write_json(status, {'at': now(), 'phase': 'complete', 'full_started': False,
            'reason': 'pilot_latency_gates_not_all_passed', 'independent_review_passed': True})
        return
    full = ROOT/'tmp/indexed_grasp_planning/meshvote69_20261126_01'
    prepare(full, ROOT/'configs/experiments/indexed_grasp_mesh_vote_full.json');augment(full);watch(full)
    pointer(full, 'mesh_vote_full_running', root);report(root, pilot, full);publish(root)
    checked = wait(full, status);report(root, pilot, full)
    pointer(full, 'mesh_vote_full_complete', root);publish(root)
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
