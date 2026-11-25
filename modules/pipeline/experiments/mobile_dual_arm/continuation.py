"""Resume only unclaimed slots under an explicit, preserved campaign lineage."""
from pathlib import Path
import os
import shutil

from arx_project.layout import ROOT, now, read_json, sha256, write_json


def execution_entries(registration):
    continuation = registration.get('continuation')
    allowed = set(continuation['pending_case_ids']) if continuation else None
    # Preserve original slot indices, including their per-slot model seeds.
    return [(number, entry) for number, entry in enumerate(registration['schedule'])
            if allowed is None or entry['id'] in allowed]


def verify_inheritance(root, registration, *, full=False):
    carried = registration.get('continuation')
    if not carried:
        return
    schedule = [entry['id'] for entry in registration['schedule']]
    completed, interrupted, pending = [set(carried[name]) for name in
        ['completed_case_ids', 'interrupted_case_ids', 'pending_case_ids']]
    predecessor = ROOT/carried['predecessor']
    accepted = read_json(predecessor/'acceptance.json')
    expected = [{row['id'] for row in accepted['results'] if row['state']==state}
                for state in ['complete', 'interrupted', 'not_started']]
    if ([completed, interrupted, pending] != expected or not accepted['record_integrity']
            or sha256(predecessor/'STOP') != carried['original_STOP_sha256']):
        raise RuntimeError('Continuation differs from the accepted predecessor ledger')
    if (completed & interrupted or completed & pending or interrupted & pending
            or completed | interrupted | pending != set(schedule)
            or len(pending) != carried['new_execution_slot_budget']
            or registration.get('automatic_retries')):
        raise RuntimeError('Continuation slot partition or retry policy differs')
    for name, expected in carried['inherited_claims_and_results'].items():
        if sha256(root/name) != expected:
            raise RuntimeError('Inherited claim/result changed: '+name)
    if full:
        for name, expected in carried['inherited_files'].items():
            if sha256(root/name) != expected:
                raise RuntimeError('Inherited evidence copy differs: '+name)
        for name, expected in carried.get('shared_immutable_files', {}).items():
            source, target = Path(expected['source']), root/name
            a, b = source.stat(), target.stat()
            if ((a.st_dev, a.st_ino) != (b.st_dev, b.st_ino)
                    or b.st_size != expected['size'] or b.st_mtime_ns != expected['mtime_ns']):
                raise RuntimeError('Inherited immutable evidence binding differs: '+name)
    for case in pending:
        if (root/'claims'/(case+'.json')).exists() or (root/'trials'/case).exists():
            raise RuntimeError('Continuation pending slot already claimed: '+case)


def adopt(root, predecessor, authorization):
    """Copy accepted evidence once, without changing the stopped predecessor."""
    root, predecessor = Path(root).resolve(), Path(predecessor).resolve()
    reg = read_json(root/'registration.json')
    previous = read_json(predecessor/'registration.json')
    acceptance = read_json(predecessor/'acceptance.json')
    if not acceptance['record_integrity'] or not (predecessor/'STOP').is_file():
        raise RuntimeError('Predecessor stop/integrity evidence is missing')
    if not read_json(predecessor/'services_stopped.json')['all_owned_exited']:
        raise RuntimeError('Predecessor services remain active')
    for key in ['schedule', 'weights', 'dataset_freeze_sha256', 'success', 'placement',
                'station_search_options', 'component_modules']:
        if reg[key] != previous[key]:
            raise RuntimeError('Continuation changes frozen method or scenes: '+key)
    if list((root/'claims').iterdir()) or list((root/'trials').iterdir()):
        raise RuntimeError('Continuation destination is not empty')
    archive = root/'registration_revisions/before_inheritance'
    archive.mkdir(parents=True)
    shutil.copy2(root/'registration.json', archive/'registration.json')
    completed = [row['id'] for row in acceptance['results'] if row['state']=='complete']
    interrupted = [row['id'] for row in acceptance['results'] if row['state']=='interrupted']
    pending = [row['id'] for row in acceptance['results'] if row['state']=='not_started']
    inherited_files, shared, essential = {}, {}, {}
    immutable_suffixes = {'.mp4', '.npy', '.npz', '.ply', '.png', '.jpg', '.jpeg', '.jsonl'}
    def copy_evidence(source, destination):
        source, destination = Path(source), Path(destination)
        if source.suffix in immutable_suffixes and source.stat().st_size >= 1024*1024:
            os.link(source, destination)
        else:
            shutil.copy2(source, destination)
        return destination
    for case in completed+interrupted:
        trial = predecessor/'trials'/case
        if trial.exists():
            shutil.copytree(trial, root/'trials'/case, copy_function=copy_evidence)
        claim = predecessor/'claims'/(case+'.json')
        shutil.copy2(claim, root/'claims'/claim.name)
        files = [root/'claims'/claim.name]
        files += sorted(path for path in (root/'trials'/case).rglob('*') if path.is_file())
        for destination in files:
            relative = str(destination.relative_to(root))
            source = predecessor/relative
            a, b = source.stat(), destination.stat()
            if (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino):
                shared[relative] = {'source': str(source), 'size': b.st_size, 'mtime_ns': b.st_mtime_ns}
                continue
            expected = sha256(source)
            if destination.is_symlink() or destination.stat().st_nlink != 1 or sha256(destination) != expected:
                raise RuntimeError('Continuation evidence copy failed: '+relative)
            inherited_files[relative] = expected
            if relative.startswith('claims/') or relative.endswith('/result.json'):
                essential[relative] = expected
    reg['continuation'] = {
        'at': now(), 'predecessor': str(predecessor.relative_to(ROOT)),
        'authorization': authorization, 'new_execution_slot_budget': len(pending),
        'completed_case_ids': completed, 'interrupted_case_ids': interrupted,
        'pending_case_ids': pending, 'inherited_claims_and_results': essential,
        'inherited_files': inherited_files, 'original_slot_indices_and_seeds_preserved': True,
        'shared_immutable_files': shared,
        'original_STOP_sha256': sha256(predecessor/'STOP'),
    }
    reg['user_authorization'] = authorization
    write_json(root/'registration.json', reg)
    verify_inheritance(root, reg, full=True)
    write_json(root/'inheritance_verified.json', {'at': now(), 'passed': True,
        'complete': len(completed), 'interrupted': len(interrupted), 'pending': len(pending),
        'independent_mutable_copies': len(inherited_files), 'shared_immutable_files': len(shared),
        'predecessor_STOP_sha256': sha256(predecessor/'STOP')})
