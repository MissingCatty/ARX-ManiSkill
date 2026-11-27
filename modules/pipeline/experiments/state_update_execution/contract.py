"""One-shot scheduling and executed-event record requirements."""
import json
from pathlib import Path

VARIANTS = ['baseline', 'optimized']


def schedule(cases):
    ids = [case['id'] for case in cases]
    if len(ids) != 74 or len(set(ids)) != 74:
        raise ValueError('The complete comparison needs the same 74 distinct scenes')
    return [{'id': case + '_' + mode, 'case_id': case, 'variant': mode}
            for index, case in enumerate(ids)
            for mode in (VARIANTS if index % 2 == 0 else VARIANTS[::-1])]


def stage_name(flags):
    if flags.get('observe') or sum(bool(flags.get(key)) for key in ['approach', 'lift']) > 1:
        raise ValueError('Only initial, measured approach or measured lift is allowed')
    return 'lift' if flags.get('lift') else 'approach' if flags.get('approach') else 'initial'


def check_requests(directory):
    """A missing command is corrupt even if only its ledger/receipt/trace survives."""
    directory = Path(directory)
    ledger = directory / 'request_ledger.jsonl'
    events = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()] if ledger.exists() else []
    stages = {event['stage'] for event in events}
    for stage in ['initial', 'approach', 'lift']:
        if (directory / ('execution_planner_' + stage + '.json')).exists():
            stages.add(stage)
    if (directory / 'pregrasp_state.json').exists(): stages.add('approach')
    if (directory / 'closed_state.json').exists(): stages.add('lift')
    trace = directory / 'execution.jsonl'
    if trace.exists():
        occurred = {json.loads(line)['stage'] for line in trace.read_text().splitlines() if line.strip()}
        if 'approach' in occurred: stages.add('approach')
        if occurred & {'lift', 'hold'}: stages.add('lift')
    for stage in stages:
        names = [stage + '_metrics.json', 'execution_planner_' + stage + '.json']
        names += ['plan_result.json'] if stage == 'initial' else [stage + '_plan_result.json',
                    'pregrasp_state.json' if stage == 'approach' else 'closed_state.json']
        for name in names:
            if not (directory / name).is_file(): raise ValueError('Missing requested ' + stage + ' record: ' + name)
        matching = [event['event'] for event in events if event['stage'] == stage]
        if matching != ['started', 'complete']:
            raise ValueError('Request ledger is incomplete or repeated: ' + stage)
    return sorted(stages)
