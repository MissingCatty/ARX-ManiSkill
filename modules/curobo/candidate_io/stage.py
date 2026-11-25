"""Prepare the I/O-only refactor without editing a running campaign's sources."""
import argparse
from pathlib import Path

from arx_project.layout import ROOT, ensure_run_path, now, sha256, write_json


def change(source, old, new, count=1):
    if source.count(old) != count:
        raise RuntimeError('Source contract changed: '+old[:100])
    return source.replace(old, new)


def dual(source):
    source = change(source, 'from modules.curobo.dual_arm.model import ArmModel',
        'from modules.curobo.dual_arm.model import ArmModel\n'
        'from modules.curobo.candidate_io.store import candidate_scope, candidate_pool, bind_candidate_view')
    source = change(source, "        candidates = directory/'candidates/candidate_pool.json'\n"
        "        if not (work/'candidates/candidate_pool.json').exists():\n"
        '            write_json(work/\'candidates/candidate_pool.json\', read_json(candidates))',
        '        if planned_open:\n            bind_candidate_view(work, directory)')
    original = "    def request(self, directory, observe=False, approach=False, lift=False, selector='graspcl', arm=None):"
    wrapper = """    def request(self, directory, observe=False, approach=False, lift=False, selector='graspcl', arm=None,
                candidate_transport=None):
        if observe or int(approach)+int(lift) > 1:
            raise ValueError('V3 starts from the fixed head view and accepts one planning stage')
        if approach or lift:
            if candidate_transport is not None:
                raise ValueError('Execution replanning does not consume a new candidate pool')
            return self._request(directory, observe, approach, lift, selector, arm)
        started = time.perf_counter()
        with candidate_scope(directory, candidate_transport):
            result = self._request(directory, observe, approach, lift, selector, arm)
        result['wall_seconds'] = time.perf_counter()-started
        write_json(Path(directory)/'plan_result.json', result)
        return result

    def _request(self, directory, observe=False, approach=False, lift=False, selector='graspcl', arm=None):"""
    source = change(source, original, wrapper)
    return change(source, "read_json(directory/'candidates/candidate_pool.json')['candidates'][selected['index']]",
        "candidate_pool(directory)['candidates'][selected['index']]")


def accelerated(source):
    source = change(source, 'from .adapter import DualArmPlanner', 'from .adapter import DualArmPlanner\n'
        'from modules.curobo.candidate_io.store import candidate_pool, limit_candidate_view')
    source = change(source, "        path = directory/'candidates/candidate_pool.json'\n        candidates = read_json(path)",
        '        candidates = candidate_pool(directory)')
    return change(source, "            bounded = {**candidates, 'candidates': candidates['candidates'][:limit]}\n"
        "            bounded['search_bound'] = receipt\n            write_json(path, bounded)",
        '            limit_candidate_view(directory, limit, receipt)')


def placement(source):
    source = change(source, 'from modules.curobo.dual_arm.accelerated_adapter import AcceleratedDualArmPlanner',
        'from modules.curobo.dual_arm.accelerated_adapter import AcceleratedDualArmPlanner\n'
        'from modules.curobo.candidate_io.store import candidate_scope, candidate_pool')
    source = change(source, "read_json(Path(directory)/'candidates/candidate_pool.json')['candidates']",
        "candidate_pool(directory)['candidates']")
    source = change(source, "read_json(directory/'candidates/candidate_pool.json')['candidates']",
        "candidate_pool(directory)['candidates']")
    source = change(source, "        if set(flags) - {'arm'}:", "        if set(flags) - {'arm', 'candidate_transport'}:")
    source = change(source, "        return self.endpoint(directory, flags.get('arm'))",
        "        return self.endpoint(directory, flags.get('arm'), flags.get('candidate_transport'))")
    return change(source, '    def endpoint(self, directory, side):',
        '    def endpoint(self, directory, side, candidate_transport=None):\n'
        '        with candidate_scope(directory, candidate_transport):\n'
        '            return self._endpoint(directory, side)\n\n'
        '    def _endpoint(self, directory, side):')


def contact(source):
    return change(source, "            candidates = read_json(directory/'candidates/candidate_pool.json')['candidates']",
        "            from modules.curobo.candidate_io.store import candidate_pool\n"
        "            candidates = candidate_pool(directory)['candidates']")


def short_lift(source):
    return change(source, "        pool = json.loads((self.out.parent / 'graspcl/candidate_pool.json').read_text())['candidates']",
        "        from modules.curobo.candidate_io.store import candidate_pool\n"
        "        pool = candidate_pool(self.out.parent)['candidates']")


def contract(source):
    source = change(source, 'def make_candidates(directory):',
        'def make_candidates(directory, *, background=False):\n    from arx_project.artifacts import publish_candidate_pool')
    source = change(source, "    dest=directory/'candidates';dest.mkdir()",
        "    dest=directory/'candidates';dest.mkdir()\n"
        "    persist_pool = (lambda payload: publish_candidate_pool(directory, payload)) if background else (lambda payload: save(dest/'candidate_pool.json', payload))")
    return change(source, "    save(dest/'candidate_pool.json',{'candidates':candidates,'protocol':VERSION,",
        "    persist_pool({'candidates':candidates,'protocol':VERSION,")


def high_workspace(source):
    return change(source,'            make_candidates(fresh)',
        '            make_candidates(fresh, background=True)')


def zerograsp(source):
    source = change(source, '        directory = Path(directory)\n',
        '        directory = Path(directory)\n        from arx_project.artifacts import publish_candidate_pool\n')
    source = change(source, "        write_json(dest/'candidate_pool.json', {'candidates': candidates,",
        "        transport = publish_candidate_pool(directory, {'candidates': candidates,")
    return change(source, "        write_json(directory/'zerograsp.json', report)\n        return report",
        "        write_json(directory/'zerograsp.json', report)\n"
        "        report['candidate_transport'] = transport\n        return report")


def client(source):
    source = change(source, '    def __init__(self,path,timeout=900):\n        self.path=str(path);self.timeout=timeout',
        '    def __init__(self,path,timeout=900,*,candidate_transport=None):\n'
        '        self.path=str(path);self.timeout=timeout\n'
        '        self.accepts_candidate_transport=candidate_transport')
    source = change(source, '    def call(self,message):\n',
        '    def call(self,message):\n'
        '        from .candidate_rpc import call_with_candidate_transport\n'
        '        return call_with_candidate_transport(self, message, self._call)\n\n'
        '    def _call(self,message):\n')
    return source


def service(source):
    source = change(source, 'from .registry import create_worker, module_records',
        'from .registry import create_worker, module_records\n'
        'from .artifacts import close_owned_transports, finish_candidate_transport, commit_candidate_transport')
    source = change(source, "        if not warm.get('success') or not warm.get('target_found',True):raise RuntimeError('Module warmup failed: '+str(warm))",
        "        if not warm.get('success') or not warm.get('target_found',True):raise RuntimeError('Module warmup failed: '+str(warm))\n"
        '        finish_candidate_transport(args.warmup_trial)')
    source = change(source, "                        elif command=='shutdown':\n",
        "                        elif command=='artifact_commit':\n"
        "                            directory=ensure_run_path(message.pop('directory'))\n"
        "                            result=commit_candidate_transport(directory,message.pop('token'))\n"
        "                            response={'status':'ok','result':result}\n"
        "                        elif command=='shutdown':\n"
        '                            close_owned_transports()\n')
    return change(source, '        finally:args.socket.unlink(missing_ok=True)',
        '        finally:\n            args.socket.unlink(missing_ok=True)\n            close_owned_transports()')


TRANSFORMS = {
    'modules/curobo/dual_arm/adapter.py': dual,
    'modules/curobo/dual_arm/accelerated_adapter.py': accelerated,
    'modules/curobo/dual_arm/placement_service.py': placement,
    'modules/curobo/fingertip_contact/planning.py': contact,
    'modules/maniskill/short_lift_planner.py': short_lift,
    'modules/pipeline/experiments/repaired_grasp_benchmark/repair_contract.py': contract,
    'modules/zerograsp/arx_grasp_adapter.py': zerograsp,
    'arx_project/client.py': client,
    'arx_project/service.py': service,
    'modules/pipeline/experiments/mobile_dual_arm/workspace_priority/runtime.py': high_workspace,
}


def prepare(root):
    root = ensure_run_path(root)
    if (root/'staging.json').exists():
        raise RuntimeError('Staging already exists; preserve its source evidence')
    records = {}
    for relative, transform in TRANSFORMS.items():
        original = ROOT/relative
        text = original.read_text()
        updated = transform(text)
        compile(updated, str(original), 'exec')
        target = root/'staged_source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(updated)
        previous = root/'source_before'/relative
        previous.parent.mkdir(parents=True, exist_ok=True)
        previous.write_text(text)
        records[relative] = {'before_sha256': sha256(original), 'after_sha256': sha256(target)}
    write_json(root/'staging.json', {'at': now(), 'files': records,
        'scope': 'Candidate data delivery, shared request views, asynchronous snapshots',
        'robot_physics_and_planner_algorithm_modified': False})
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print('STAGED', len(prepare(args.root)), 'source files')
