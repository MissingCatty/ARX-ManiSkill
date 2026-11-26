"""Add immutable diagnostic scenes and actual mesh files to the run contract."""
import json
from pathlib import Path
import shutil

from arx_project.layout import ROOT, sha256, write_json


def augment(root):
    reg = json.loads((root/'registration.json').read_text())
    source = ROOT/reg['diagnostic_source'];previous = json.loads((source/'registration.json').read_text())
    checked = json.loads((source/'independent_review.json').read_text())
    if (source/'STOP').exists() or not all(checked[k] for k in
            ['passed', 'record_integrity', 'all_original_selections_and_branches_retained']):
        raise ValueError('IK diagnosis did not pass its independent checks')
    for name, digest in previous['source_sha256'].items():
        if sha256(source/'source_registered'/Path(name).relative_to(ROOT)) != digest:
            raise ValueError('Original diagnosis source archive changed')
    inputs = reg['input_sha256']
    for name in ['registration.json', 'independent_review.json', 'acceptance.json']:
        inputs[str(source/name)] = sha256(source/name)
    for case in previous['cases']:
        directory = source/'plans'/case['id']/'profiled_1'
        for path in directory.glob('ik_sample_*.trace.json'):
            inputs[str(path)] = sha256(path)
        path = directory/'arms/right/plan/scene.json';inputs[str(path)] = sha256(path)
        for mesh in json.loads(path.read_text()).get('mesh', {}).values():
            if mesh.get('file_path'):
                path = Path(mesh['file_path']);inputs[str(path)] = sha256(path)
    path = Path(__file__).resolve()
    if str(path) not in reg['source_sha256']:
        target = root/'source_registered'/path.relative_to(ROOT);target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target);reg['source_sha256'][str(path)] = sha256(path)
    write_json(root/'registration.json', reg);write_json(root/'registration_pending.json', reg)
    return reg
