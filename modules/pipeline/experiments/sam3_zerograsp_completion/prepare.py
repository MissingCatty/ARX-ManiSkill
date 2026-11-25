"""Preserve interrupted attempts and register exactly two newly authorized ones."""
from copy import deepcopy
from pathlib import Path
import os
import shutil

from arx_project.artifacts import atomic_json
from arx_project.json_io import read_json
from arx_project.layout import ROOT, now, sha256
from modules.pipeline.experiments.sam3_zerograsp_comparison.io_parallel.contracts import METHODS
from modules.pipeline.experiments.sam3_zerograsp_comparison.io_parallel.continuation import partition, verify_inheritance


def completion_slots(previous, acceptance, authorized):
    if not acceptance.get('passed') or not acceptance.get('record_integrity'):
        raise RuntimeError('Predecessor records were not accepted')
    if len(authorized) != 2 or len(set(authorized)) != 2:
        raise RuntimeError('Exactly two distinct interrupted cases are authorized')
    original = {method: partition(previous, acceptance, method) for method in METHODS}
    if (len(original['graspgenx']['complete']) != 74 or original['graspgenx']['interrupted']
            or original['graspgenx']['not_started']):
        raise RuntimeError('GraspGenX must finish all 74 cases first')
    zero = original['zerograsp']
    if (len(zero['complete']) != 72 or zero['not_started']
            or set(zero['interrupted']) != set(authorized)):
        raise RuntimeError('Makeup authorization does not match the two interrupted cases')
    slots = deepcopy(original)
    slots['zerograsp']['not_started'] = list(zero['interrupted'])
    slots['zerograsp']['interrupted'] = []
    return original, slots


def prepare(root, predecessor, configuration):
    root, predecessor = Path(root).resolve(), Path(predecessor).resolve()
    if root.exists():
        raise RuntimeError('Makeup requires a fresh registration')
    previous = read_json(predecessor/'registration.json')
    acceptance = read_json(predecessor/'acceptance.json')
    if not read_json(predecessor/'services_stopped.json')['all_owned_exited']:
        raise RuntimeError('Predecessor services are still active')
    original, slots = completion_slots(previous, acceptance, configuration['authorized_makeup_cases'])
    sources = dict(previous['source_hashes'])
    for filename, digest in sources.items():
        if sha256(filename) != digest:
            raise RuntimeError('Frozen execution source changed: '+filename)
    for path in [*Path(__file__).parent.glob('*.py'), ROOT/'configs/experiments/sam3_zerograsp_complete74.json']:
        sources[str(path)] = sha256(path)
    for filename, record in previous['weights'].items():
        if Path(filename).stat().st_size != record['size'] or sha256(filename) != record['sha256']:
            raise RuntimeError('Frozen model weights changed: '+filename)
    root.mkdir(parents=True)
    for name in ['preflight', 'methods', 'source_registered', 'historical_attempts']:
        (root/name).mkdir()
    shutil.copy2(ROOT/'website/内容/迭代/sam3-graspgenx-v6.md', root/'previous_website_page.md')
    for filename in sources:
        target = root/'source_registered'/Path(filename).relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(filename, target)
    essential, files = {}, {}
    immutable = {'.mp4', '.npy', '.npz', '.ply', '.png', '.jpg', '.jpeg', '.jsonl'}

    def copy_file(source, destination):
        source, destination = Path(source), Path(destination)
        relative = str(destination.relative_to(root))
        if (not relative.startswith('historical_attempts/')
                and source.suffix in immutable and source.stat().st_size >= 1024**2):
            os.link(source, destination)
            stat = destination.stat()
            files[relative] = {'storage':'shared_immutable', 'size':stat.st_size, 'mtime_ns':stat.st_mtime_ns}
        else:
            shutil.copy2(source, destination)
            digest = sha256(source)
            if sha256(destination) != digest:
                raise RuntimeError('Inherited evidence copy differs: '+relative)
            files[relative] = {'storage':'independent_copy', 'sha256':digest}
            if '/claims/' in relative or relative.endswith(('/result.json', '/latency.json')):
                essential[relative] = digest

    for method in METHODS:
        dest, source = root/'methods'/method, predecessor/'methods'/method
        for name in ['preflight', 'resident', 'claims', 'trials']:
            (dest/name).mkdir(parents=True, exist_ok=True)
        for name in ['gripper_descriptor.json', 'object_catalog.json', 'common_initial_conditions.json']:
            shutil.copy2(source/name, dest/name)
        for case in slots[method]['complete']:
            copy_file(source/'claims'/(case+'.json'), dest/'claims'/(case+'.json'))
            shutil.copytree(source/'trials'/case, dest/'trials'/case, copy_function=copy_file)
    for case in original['zerograsp']['interrupted']:
        dest = root/'historical_attempts/methods/zerograsp'
        (dest/'claims').mkdir(parents=True, exist_ok=True)
        (dest/'trials').mkdir(exist_ok=True)
        source = predecessor/'methods/zerograsp'
        copy_file(source/'claims'/(case+'.json'), dest/'claims'/(case+'.json'))
        if (source/'trials'/case).exists():
            shutil.copytree(source/'trials'/case, dest/'trials'/case, copy_function=copy_file)
    evidence_names = ['registration.json', 'acceptance.json', 'services_stopped.json']
    evidence_names.extend(name for name in ['STOP', 'ABORT'] if (predecessor/name).is_file())
    lineage = {'at':now(), 'predecessor':str(predecessor.relative_to(ROOT)),
        'authorization':configuration['user_authorization'],
        'predecessor_evidence':{name:sha256(predecessor/name) for name in evidence_names},
        'source_method_slots':original, 'method_slots':slots,
        'inherited_essential_files':essential, 'inherited_files':files,
        'original_slot_indices_and_seeds_preserved':True,
        'explicitly_authorized_makeup_cases':configuration['authorized_makeup_cases'],
        'historical_interruptions_preserved':'historical_attempts/methods/zerograsp'}
    reg = {**deepcopy(previous), 'at':now(), 'source_hashes':sources,
        'run_type':'explicitly_authorized_two_interrupted_makeup',
        'user_authorization':configuration['user_authorization'], 'continuation':lineage,
        'bounded_scene_mesh_cache':True, 'automatic_retries':False,
        'new_scene_slot_budget':2, 'new_scene_ids':0, 'new_attempt_budget':2,
        'maximum_attempts_per_case':1, 'maximum_attempts_scope':'this_registration',
        'historical_attempts_are_preserved':True, 'predecessor_source_changes':{}}
    reg.pop('startup_dependency', None)
    reg['configuration']['user_authorization'] = configuration['user_authorization']
    atomic_json(root/'registration.json', reg)
    for method in METHODS:
        per_method = {**deepcopy(reg), 'method':method, 'campaign':str(root.relative_to(ROOT)),
            'physical_slot_budget':len(slots[method]['not_started']),
            'required_components':['zerograsp','curobo_dual_arm'] if method == 'zerograsp'
                else ['rayst3r','graspgenx','curobo_dual_arm']}
        per_method['continuation']['pending_case_ids'] = slots[method]['not_started']
        per_method['continuation']['inherited_interrupted_case_ids'] = []
        atomic_json(root/'methods'/method/'registration.json', per_method)
        if not slots[method]['not_started']:
            atomic_json(root/'methods'/method/'complete.json', {'at':now(), 'completed':74,
                'inherited_interrupted':0, 'authorized_pending_schedule_exhausted':True, 'inherited_only':True})
            atomic_json(root/'methods'/method/'services_stopped.json',
                {'at':now(),'all_owned_exited':True,'models_started':False})
    verify_inheritance(root, reg, full=True)
    atomic_json(root/'inheritance_verified.json', {'at':now(),'passed':True,
        'inherited_completed':146,'new_authorized_attempts':2,'historical_interrupted':2,
        'scene_robot_model_physics_and_success_unchanged':True})
    from modules.pipeline.experiments.sam3_zerograsp_comparison.io_parallel.metrics import summarize
    summarize(root)
    return reg
