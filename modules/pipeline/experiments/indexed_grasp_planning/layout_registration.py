"""Bind the accepted component diagnosis and actual GPU snapshots to each run."""
import json
from pathlib import Path

from arx_project.layout import ROOT, sha256, write_json


def augment(root):
    reg = json.loads((root/'registration.json').read_text())
    source = ROOT/reg['diagnostic_source']
    previous = json.loads((source/'registration.json').read_text())
    checked = json.loads((source/'independent_review.json').read_text())
    if (source/'STOP').exists() or not all(checked[k] for k in
            ['passed', 'record_integrity', 'all_original_selections_and_branches_retained',
             'actual_workloads_verified', 'original_planning_decisions_retained']):
        raise ValueError('Actual query diagnosis has not passed independent review')
    if checked['component_probe_samples'] != 5 or checked['checked_requests'] != 10:
        raise ValueError('The five-scene actual-query diagnosis is incomplete')
    for name, digest in previous['source_sha256'].items():
        archived = source/'source_registered'/Path(name).relative_to(ROOT)
        if sha256(archived) != digest: raise ValueError('Diagnosis source archive changed')
    for name in ['registration.json', 'independent_review.json', 'acceptance.json']:
        reg['input_sha256'][str(source/name)] = sha256(source/name)
    for case in previous['cases']:
        directory = source/'plans'/case['id']/'captured_1'
        paths = [directory/'query_probe.json']+[directory/'query_components'/name for name in
            ['actual_queries.npz', 'actual_meshes.npz', 'scene.json', 'components.json']]
        for path in paths: reg['input_sha256'][str(path)] = sha256(path)
    write_json(root/'registration.json', reg);write_json(root/'registration_pending.json', reg)
    return reg
