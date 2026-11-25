"""Carry accepted results forward and execute only previously unclaimed slots."""
from copy import deepcopy
import os
from pathlib import Path
import shutil

from arx_project.artifacts import atomic_json
from arx_project.json_io import read_json
from arx_project.layout import ROOT, now, sha256
from .contracts import METHODS


def partition(registration, acceptance, method):
    ids = [pair['id'] for pair in registration['cohort']['pairs']]
    rows = [row for row in acceptance['results'] if row['method'] == method]
    if len(rows) != len(ids) or {r['id'] for r in rows} != set(ids):
        raise RuntimeError('Accepted predecessor ledger does not cover the exact cohort')
    result = {state: [r['id'] for r in rows if r['state'] == state]
              for state in ['complete', 'interrupted', 'not_started']}
    if sum(map(len, result.values())) != len(ids):
        raise RuntimeError('Unknown or duplicate predecessor slot state')
    return result


def execution_pairs(registration):
    inherited = registration.get('continuation')
    allowed = set(inherited['pending_case_ids']) if inherited else None
    return [(index, pair) for index, pair in enumerate(registration['cohort']['pairs'])
            if allowed is None or pair['id'] in allowed]


def inherited_terminal_method(states):
    return bool(states['complete'] or states['interrupted']) and not states['not_started']


def verify_inheritance(root, registration, *, full=False):
    lineage = registration.get('continuation')
    if not lineage:
        return
    predecessor = ROOT/lineage['predecessor']
    for name, digest in lineage['predecessor_evidence'].items():
        if sha256(predecessor/name) != digest:
            raise RuntimeError('Predecessor evidence changed: '+name)
    for name, digest in lineage['inherited_essential_files'].items():
        if sha256(Path(root)/name) != digest:
            raise RuntimeError('Inherited claim/result changed: '+name)
    if full:
        for name, record in lineage['inherited_files'].items():
            source, target = predecessor/name, Path(root)/name
            if record['storage'] == 'shared_immutable':
                a, b = source.stat(), target.stat()
                if ((a.st_dev, a.st_ino) != (b.st_dev, b.st_ino)
                        or b.st_size != record['size'] or b.st_mtime_ns != record['mtime_ns']):
                    raise RuntimeError('Inherited immutable file changed: '+name)
            elif sha256(target) != record['sha256']:
                raise RuntimeError('Inherited evidence differs: '+name)


def prepare(root, predecessor):
    root, predecessor = Path(root).resolve(), Path(predecessor).resolve()
    if root.exists():
        raise RuntimeError('Continuation requires a new directory')
    previous = read_json(predecessor/'registration.json')
    acceptance = read_json(predecessor/'acceptance.json')
    if (not acceptance['passed'] or not acceptance['record_integrity']
            or not (predecessor/'STOP').is_file()
            or not read_json(predecessor/'services_stopped.json')['all_owned_exited']):
        raise RuntimeError('Stopped predecessor is not accepted and fully exited')
    slots = {method: partition(previous, acceptance, method) for method in METHODS}
    if not any(states['not_started'] for states in slots.values()):
        raise RuntimeError('No unclaimed scene remains; do not create another continuation')
    changes = {}
    allowed = {'campaign.py', 'worker.py', 'services.py', 'metrics.py', 'audit.py',
               'publication.py', 'test_contract.py', 'continuation.py', 'test_continuation.py'}
    sources = {}
    for filename, digest in previous['source_hashes'].items():
        actual = sha256(filename)
        path = Path(filename)
        if actual != digest:
            if path.parent != Path(__file__).parent or path.name not in allowed:
                raise RuntimeError('Unapproved continuation source change: '+filename)
            changes[filename] = {'before': digest, 'after': actual}
        sources[filename] = actual
    for path in [*Path(__file__).parent.glob('*.py'), ROOT/'modules/curobo/scene_cache.py']:
        sources[str(path)] = sha256(path)
    for filename, record in previous['weights'].items():
        if Path(filename).stat().st_size != record['size'] or sha256(filename) != record['sha256']:
            raise RuntimeError('Model weights changed: '+filename)
    root.mkdir(parents=True)
    for folder in ['preflight', 'methods', 'source_registered']:
        (root/folder).mkdir()
    shutil.copy2(ROOT/'website/内容/迭代/sam3-graspgenx-v6.md', root/'previous_website_page.md')
    for filename in sources:
        dest = root/'source_registered'/Path(filename).relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(filename, dest)
    essential, files = {}, {}
    immutable = {'.mp4', '.npy', '.npz', '.ply', '.png', '.jpg', '.jpeg', '.jsonl'}

    def copy_file(source, destination):
        source, destination = Path(source), Path(destination)
        relative = str(destination.relative_to(root))
        if source.suffix in immutable and source.stat().st_size >= 1024**2:
            os.link(source, destination)
            stat = destination.stat()
            files[relative] = {'storage': 'shared_immutable', 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        else:
            shutil.copy2(source, destination)
            digest = sha256(source)
            if sha256(destination) != digest:
                raise RuntimeError('Evidence copy mismatch: '+relative)
            files[relative] = {'storage': 'independent_copy', 'sha256': digest}
            if ('/claims/' in relative or relative.endswith('/result.json')
                    or relative.endswith('/latency.json')):
                essential[relative] = digest

    for method in METHODS:
        dest, source = root/'methods'/method, predecessor/'methods'/method
        for folder in ['preflight', 'resident', 'claims', 'trials']:
            (dest/folder).mkdir(parents=True, exist_ok=True)
        for name in ['gripper_descriptor.json', 'object_catalog.json', 'common_initial_conditions.json']:
            shutil.copy2(source/name, dest/name)
        for case in slots[method]['complete']+slots[method]['interrupted']:
            copy_file(source/'claims'/(case+'.json'), dest/'claims'/(case+'.json'))
            if (source/'trials'/case).exists():
                shutil.copytree(source/'trials'/case, dest/'trials'/case, copy_function=copy_file)
    lineage = {'at': now(), 'predecessor': str(predecessor.relative_to(ROOT)),
        'authorization': '用户明确要求继续；保留既有结果、中断及STOP，仅执行未领取槽，不重试',
        'predecessor_evidence': {name: sha256(predecessor/name) for name in
            ['registration.json', 'acceptance.json', 'STOP', 'services_stopped.json']},
        'method_slots': slots, 'inherited_essential_files': essential, 'inherited_files': files,
        'original_slot_indices_and_seeds_preserved': True}
    reg = {**deepcopy(previous), 'at': now(), 'source_hashes': sources,
        'run_type': 'continue_unclaimed_io74_with_bounded_mesh_cache',
        'user_authorization': lineage['authorization'], 'continuation': lineage,
        'bounded_scene_mesh_cache': True, 'automatic_retries': False,
        'new_scene_slot_budget': sum(len(s['not_started']) for s in slots.values()),
        'predecessor_source_changes': changes}
    atomic_json(root/'registration.json', reg)
    for method in METHODS:
        per_method = {**deepcopy(reg), 'method': method, 'campaign': str(root.relative_to(ROOT)),
            'physical_slot_budget': sum(not pair.get('common_precondition_failure')
                for pair in reg['cohort']['pairs'] if pair['id'] in slots[method]['not_started']),
            'required_components': ['rayst3r', 'graspgenx', 'curobo_dual_arm'] if method == 'graspgenx'
                else ['zerograsp', 'curobo_dual_arm']}
        per_method['continuation']['pending_case_ids'] = slots[method]['not_started']
        per_method['continuation']['inherited_interrupted_case_ids'] = slots[method]['interrupted']
        atomic_json(root/'methods'/method/'registration.json', per_method)
        if inherited_terminal_method(slots[method]):
            atomic_json(root/'methods'/method/'complete.json', {'at':now(),
                'completed':len(slots[method]['complete']),
                'inherited_interrupted':len(slots[method]['interrupted']),
                'authorized_pending_schedule_exhausted':True,'inherited_only':True})
            atomic_json(root/'methods'/method/'services_stopped.json',
                {'at':now(),'all_owned_exited':True,'models_started':False})
    verify_inheritance(root, reg, full=True)
    atomic_json(root/'inheritance_verified.json', {'at': now(), 'passed': True,
        'methods': {method: {state: len(ids) for state, ids in states.items()}
                    for method, states in slots.items()},
        'changed_sources': changes, 'scene_robot_model_physics_and_success_unchanged': True})
    from .metrics import summarize
    summarize(root)
    print('REGISTERED_CONTINUATION_IO74', root, flush=True)
