"""CPU-only service fixture for testing generator/planner artifact lifecycle."""
import argparse
from pathlib import Path
import time

from arx_project.layout import write_json
from modules.curobo.candidate_io.loader import load_staged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--staged', type=Path, required=True)
    parser.add_argument('--kind', required=True)
    for name in ['root', 'socket', 'warmup-trial', 'descriptor']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    modules = load_staged(args.staged, planning=False, client=True, service=True)
    from arx_project import artifacts
    from modules.curobo.candidate_io.store import candidate_pool, candidate_scope, bind_candidate_view
    original_write = artifacts.atomic_json

    def slow_write(path, value):
        if 'test_trial' in str(path) and Path(path).name == 'candidate_pool.json':
            time.sleep(.6)
            if 'write_failure' in str(path):
                raise OSError('injected disk full')
        return original_write(path, value)

    artifacts.atomic_json = slow_write

    class Worker:
        def request(self, directory, candidate_transport=None):
            directory = Path(directory)
            if args.kind == 'zerograsp':
                from modules.curobo.candidate_io.test_io import payload
                value = payload(0 if 'empty_pool' in str(directory) else 5)
                descriptor = artifacts.publish_candidate_pool(directory, value)
                return {'success': True, 'candidate_count': len(value['candidates']),
                        'candidate_transport': descriptor}
            existed = (directory/'candidates/candidate_pool.json').exists()
            with candidate_scope(directory, candidate_transport):
                count = len(candidate_pool(directory)['candidates'])
                bind_candidate_view(directory/'arms/right', directory)
            write_json(directory/'consumer_started.json', {'snapshot_present_at_entry': existed,
                'candidate_count': count})
            return {'success': True, 'candidate_count': count, 'snapshot_present_at_entry': existed}

    modules['service'].serve(args, worker=Worker())


if __name__ == '__main__':
    main()
