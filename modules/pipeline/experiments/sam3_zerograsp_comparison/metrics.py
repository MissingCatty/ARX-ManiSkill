"""Paired rates and measured warm-request latency, with explicit sample counts."""
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from arx_project.layout import ROOT, read_json, write_json


def stamp(value):
    return datetime.fromisoformat(value).timestamp()


def request_index(roots):
    index = {}
    for root in roots:
        path = root/'requests.jsonl'
        if not path.exists():
            continue
        pending = {}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            directory = row.get('directory', '')
            if '/trials/' not in directory:
                continue
            tail = directory.split('/trials/', 1)[1]
            key = (tail, row['kind'])
            if row['event'] == 'request':
                pending[key] = row
            elif row['event'] == 'response' and key in pending:
                begin = pending.pop(key)
                index.setdefault(key, []).append({'begin': stamp(begin['at']), 'end': stamp(row['at']),
                    'seconds': stamp(row['at'])-stamp(begin['at']),
                    'approach': bool(begin.get('approach')), 'lift': bool(begin.get('lift')),
                    'result': {k: v for k, v in row.get('response', {}).get('result', {}).items()
                               if k in ['success', 'wall_seconds', 'seconds']}})
    return index


def baseline_roots(source):
    roots = [Path(source)]
    reg = read_json(Path(source)/'registration.json')
    predecessor = reg.get('continuation', {}).get('predecessor')
    if predecessor:
        roots.append(ROOT/predecessor)
    return roots


def measured_times(pair, trial, index, *, baseline=False):
    trial = Path(trial)
    if pair.get('common_precondition_failure'):
        return {}
    if baseline:
        source = ROOT/pair['observation']
        sam = read_json(source/'sam3.json').get('wall_seconds')
        ray_path, grasp_path = source/'rayst3r.json', source/'graspgenx_raw/inference.json'
        generation = (read_json(ray_path)['wall_seconds']+read_json(grasp_path)['wall_seconds']
                      if ray_path.exists() and grasp_path.exists() else None)
        # The successful handoff is copied to trial/; requests belong to its original segment.
        if pair['baseline_planning_success']:
            history = read_json(trial/'stationing/prepared_plan_handoff.json')['priority_history']
            scope = history[-1]['scope']
            number = len(history)-1
            tail = pair['id']+f'/stationing/segments/{number:02d}_{scope}/observation'
        else:
            tail = pair['observation'].split('/trials/', 1)[1]
    else:
        source = trial
        sam = None
        report = read_json(trial/'zerograsp.json') if (trial/'zerograsp.json').exists() else {}
        generation = report.get('seconds')
        tail = pair['id']
    planning_events = [e for e in index.get((tail, 'curobo_dual_arm'), [])
                       if not e['approach'] and not e['lift']]
    plan = read_json(trial/'plan_result.json') if (trial/'plan_result.json').exists() else {}
    planning = plan.get('wall_seconds')
    if planning is None and plan.get('reason') == 'zerograsp_no_postprocessed_candidate':
        planning = 0.
    values = {'shared_sam3_seconds': sam, 'generation_seconds': generation,
              'initial_planning_seconds': planning}
    if generation is not None and planning is not None:
        values['generation_and_initial_planning_seconds'] = generation+planning
    terminal = trial/'final_state.json'
    if planning_events and terminal.exists():
        end = terminal.stat().st_mtime
        event = planning_events[-1]
        if end >= event['end']:
            values['execution_with_replanning_seconds'] = end-event['end']
            values['warm_request_to_terminal_seconds'] = (
                (generation or 0.)+event['seconds']+end-event['end'])
    values['timing_source'] = 'resident synchronized stage reports; RPC response to final-state timestamp for execution'
    return {key: value for key, value in values.items() if value is not None}


def distribution(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return {'samples': len(values), 'median_seconds': float(np.median(values)) if values else None,
            'p90_seconds': float(np.percentile(values, 90)) if values else None}


def summarize(root, phase):
    root = Path(root)
    reg = read_json(root/'registration.json')
    pairs = reg['cohort']['pairs']
    rows = []
    interrupted = 0
    for pair in pairs:
        claim = root/'claims'/(pair['id']+'.json')
        if not claim.exists():
            continue
        if read_json(claim)['state'] != 'complete':
            interrupted += 1
            continue
        new = read_json(root/'trials'/pair['id']/'result.json')
        rows.append({'id': pair['id'], 'layout': pair['entry']['layout'],
            'baseline_planning_success': pair['baseline_planning_success'],
            'baseline_success': pair['baseline_success'], 'baseline_reason': pair['baseline_reason'],
            'zerograsp_planning_success': bool(new.get('planning_success')),
            'zerograsp_success': bool(new['success']), 'zerograsp_reason': new['reason']})
    n = len(rows)
    result = {'phase': phase, 'registered_pairs': len(pairs), 'completed_pairs': n,
        'interrupted': interrupted, 'not_started': len(pairs)-n-interrupted,
        'sam3_excluded': sum(e['kind'] == 'sam3_failure' for e in reg['cohort']['excluded']),
        'baseline_interrupted': sum(e['kind'] == 'interrupted' for e in reg['cohort']['excluded']),
        'denominator': n, 'results': rows}
    for method in ['baseline', 'zerograsp']:
        planning = sum(r[method+'_planning_success'] for r in rows)
        success = sum(r[method+'_success'] for r in rows)
        result[method] = {'planning_passes': planning, 'successes': success,
            'planning_rate': planning/n if n else None, 'grasp_rate': success/n if n else None}
    write_json(root/'summary.json', result)
    write_json(root/'progress.json', {k: v for k, v in result.items() if k != 'results'})
    return result
