"""Review C's bounded pilot, then compare all inputs only when it saves time."""
import argparse
from pathlib import Path
import subprocess
import time
import traceback

from arx_project.layout import ROOT, ensure_run_path, now, read_json, write_json
from .binding_replay import launch
from .followup import try_publish, wait_for_exit
from .review import review


def finish_config(run, phase):
    path = ROOT/'configs/experiments/indexed_grasp_planning.json'
    config = read_json(path)
    config['execution'].update(current_phase=phase,
        next_stage='analyze_binding_results_then_geometric_IK_and_index_variants')
    config['implementation_status'] = phase+'_geometric_and_index_pending'
    for variant in config['variants']:
        if variant['id'] == 'binding_reuse': variant['state'] = phase
    config['binding_reuse_implementation']['accepted_run'] = str(run.relative_to(ROOT))
    write_json(path, config)


def follow(root, pilot, full, gpu):
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/'started.json', {'at': now(), 'pilot': str(pilot), 'full': str(full)})
    wait_for_exit(pilot, root/'status.json')
    checked = review(pilot)
    accepted = read_json(pilot/'acceptance.json')
    if not checked['passed'] or not all(accepted['control_successes_retained'].values()):
        raise RuntimeError('C pilot FK or original planning success retention failed')
    if not accepted['all_original_selections_retained']:
        raise RuntimeError('C pilot changed the selected grasp or arm; diagnose before continuing')
    control, optimized = (accepted['summaries'][key] for key in ['control', 'reuse'])
    saving = 1-optimized['total_seconds']/control['total_seconds']
    policy = read_json(ROOT/'configs/experiments/indexed_grasp_binding_reuse.json')
    useful = (optimized['full_world_rebinds'] < control['full_world_rebinds']
        and saving >= policy['full_replay_gate']['minimum_total_wall_reduction_fraction'])
    write_json(root/'pilot_gate.json', {'at': now(), 'independent_review_passed': True,
        'original_successes_and_selections_retained': True,
        'total_wall_reduction_fraction': saving,
        'world_rebinds_control': control['full_world_rebinds'],
        'world_rebinds_reuse': optimized['full_world_rebinds'],
        'full_replay_admitted': useful, 'physical_grasps': 0})
    if not useful:
        finish_config(pilot, 'binding_pilot_no_material_speedup')
        published = try_publish(root)
        write_json(root/'status.json', {'at': now(), 'phase': 'complete',
            'pilot_gate': 'insufficient_total_speedup_no_full_replay',
            'planning_requests': accepted['planning_requests'], 'website_published': published,
            'physical_grasps': 0})
        return
    deadline = time.monotonic()+6*3600
    while True:
        free = int(subprocess.check_output(['nvidia-smi', '--id='+gpu, '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'], text=True, timeout=15).strip())
        if free >= 10000: break
        write_json(root/'status.json', {'at': now(), 'phase': 'waiting_free_gpu',
            'free_mib': free, 'gpu': gpu, 'full_registration_not_started': True})
        if time.monotonic() >= deadline:
            raise RuntimeError('No planning GPU memory became available before the queue deadline')
        time.sleep(15)
    launch(full, gpu)
    try_publish(root)
    wait_for_exit(full, root/'status.json')
    checked = review(full)
    if not checked['passed']:
        raise RuntimeError('C full comparison failed independent physical-arm FK')
    finish_config(full, 'binding_reuse_replay_complete')
    published = try_publish(root)
    write_json(root/'status.json', {'at': now(), 'phase': 'complete',
        'full_run': str(full.relative_to(ROOT)), 'independent_review_passed': True,
        'planning_requests': read_json(full/'acceptance.json')['planning_requests'],
        'website_published': published, 'physical_grasps': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'pilot', 'full']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--gpu', required=True)
    args = parser.parse_args()
    root = ensure_run_path(args.root)
    if (root/'started.json').exists() or (root/'STOP').exists():
        raise ValueError('Preserve the existing C follow-up; do not start it twice')
    try:
        follow(root, ensure_run_path(args.pilot), ensure_run_path(args.full), args.gpu)
    except BaseException:
        root.mkdir(parents=True, exist_ok=True)
        write_json(root/'STOP', {'at': now(), 'error': traceback.format_exc(), 'physical_grasps': 0})
        raise
