"""Frozen-source succession and transport evidence for this repeat."""
from pathlib import Path
import json

from arx_project.layout import ROOT, read_json, sha256

METHODS = ('graspgenx', 'zerograsp')


def source_succession(previous, activation):
    proof = read_json(activation/'activation_registration.json')
    if not read_json(activation/'post_activation_verification.json')['passed']:
        raise RuntimeError('Candidate I/O activation was not accepted')
    changes = {}
    for filename, original in previous['source_hashes'].items():
        path = Path(filename)
        actual = sha256(path)
        if actual == original:
            continue
        relative = path.relative_to(ROOT).as_posix()
        archived = activation/'source_before'/relative
        if not archived.is_file() or sha256(archived) != original:
            raise RuntimeError('Unregistered predecessor source change: '+relative)
        if relative in proof['files']:
            if actual != proof['files'][relative]['after_sha256']:
                raise RuntimeError('Candidate I/O implementation changed after validation: '+relative)
        elif relative == 'configs/experiments/active_experiment.json':
            old, new = read_json(archived), read_json(path)
            excluded = {'at', 'configuration_hashes', 'candidate_io_software'}
            if {k:v for k,v in old.items() if k not in excluded} != {k:v for k,v in new.items() if k not in excluded}:
                raise RuntimeError('Physical configuration changed')
        elif relative == 'configs/modules.json':
            old, new = read_json(archived), read_json(path)
            for name in ['curobo_dual_arm', 'curobo_dual_arm_map']:
                record = new['modules'][name]
                if record.pop('candidate_transport_format', None) != 'arx_candidate_json_shm_v1':
                    raise RuntimeError('Planner cannot consume shared candidates')
                record.pop('candidate_data_delivery', None)
                old_files = old['modules'][name].get('source_files', [])
                if set(record.pop('source_files', [])) != set(old_files) | set(proof['runtime_dependencies']):
                    raise RuntimeError('Planner source dependencies changed')
                if old_files:record['source_files'] = old_files
            if old != new:
                raise RuntimeError('Non-I/O module configuration changed')
        else:
            raise RuntimeError('Unexpected software change: '+relative)
        changes[relative] = {'before':original, 'after':actual}
    return changes


def verify_io(directory, *, require_decode):
    directory = Path(directory)
    commit = read_json(directory/'candidate_snapshot_commit.json')
    if commit.get('persisted') is not True:
        raise RuntimeError('Candidate snapshot was not committed')
    if require_decode:
        evidence = read_json(directory/'candidate_io.json')
        if (evidence.get('source') != 'shared_memory' or evidence.get('decode_count') != 1
                or evidence.get('full_arm_pool_copies') != 0
                or evidence.get('candidate_order_poses_and_scores_unchanged') is not True):
            raise RuntimeError('Planning did not use the registered optimized candidate handoff')
        return evidence
    return None


def completed_rows(method_root):
    method_root = Path(method_root)
    if not (method_root/'registration.json').exists():return []
    reg = read_json(method_root/'registration.json')
    rows = []
    for pair in reg['cohort']['pairs']:
        claim = method_root/'claims'/(pair['id']+'.json')
        if not claim.exists() or read_json(claim).get('state') != 'complete':continue
        out = method_root/'trials'/pair['id']
        receipt = read_json(claim)
        if sha256(out/'result.json') != receipt['result_sha256']:
            raise RuntimeError('Completed result changed: '+pair['id'])
        if sha256(out/'latency.json') != receipt['latency_sha256']:
            raise RuntimeError('Completed latency changed: '+pair['id'])
        rows.append({'id':pair['id'], 'layout':pair['entry']['layout'],
            'result':read_json(out/'result.json'), 'latency':read_json(out/'latency.json')})
    return rows
