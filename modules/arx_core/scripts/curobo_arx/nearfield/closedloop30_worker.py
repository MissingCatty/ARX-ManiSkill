"""Frozen cuRobo planner with finite, recorded entry-goal selection."""
import argparse
import copy
from pathlib import Path
import shutil
from scripts.curobo_arx.data_factory.common import read, write
from scripts.curobo_arx.seed_augmentation import worker
from .closure_teacher import Teacher as OriginalTeacher
from .closedloop30_rules import entry_candidates


class Teacher(OriginalTeacher):
    def handle(self, request):
        if request['action'] != 'adaptive_entry': return super().handle(request)
        if request.get('label') != 'state_generation': raise ValueError('Entry fallback has invalid motion role')
        out = Path(request['output'])
        matches = [t for t in self.registration['seed_trials'] if t['trial_id'] in {p.name for p in (out, *out.parents)}]
        if len(matches) != 1: raise ValueError('Entry selection lacks a unique registered case')
        rows = []; selected = None; result = None
        for index, candidate in enumerate(entry_candidates(matches[0])):
            folder = out/'entry_goal_search'/f'{index:02d}'; folder.mkdir(parents=True, exist_ok=False)
            changed = {**copy.deepcopy(request), 'action': 'plan', 'goal_world': candidate['goal_world'], 'output': str(folder)}
            write(folder/'request.json', changed)
            result = super().handle(changed); write(folder/'reply.json', result)
            rows.append({**candidate, 'rejected': result.get('rejected'), 'directory': str(folder)})
            write(out/'entry_goal_search.json', {'candidates': rows, 'selected': selected, 'physical_commands': 0})
            if result.get('rejected'):
                # The initial fallback applies specifically to an unreachable H.
                # Resource/errors raise; collision/visibility failures are not hidden.
                if result['rejected'] != 'no_ik_solution': break
                continue
            selected = candidate
            for file in folder.iterdir():
                if file.is_file() and file.name not in ('request.json', 'reply.json'):
                    dest = out/file.name
                    if dest.exists(): raise RuntimeError('Do not overwrite prior entry evidence')
                    shutil.copy2(file, dest)
            break
        write(out/'entry_goal_search.json', {'candidates': rows, 'selected': selected, 'physical_commands': 0,
              'Gstar_used': False, 'nominal_H_world': matches[0]['entry_world']})
        return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--directory', required=True); p.add_argument('--registration', required=True)
    a = p.parse_args(); worker.Teacher = Teacher; worker.serve(a.directory, read(a.registration))
