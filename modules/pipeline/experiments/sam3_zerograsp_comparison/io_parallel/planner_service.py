"""Same registered planner, with bounded native mesh-cache lifetime."""
import argparse
from pathlib import Path

from arx_project.artifacts import atomic_json
from arx_project.layout import now
from arx_project.registry import create_worker
from arx_project.service import serve
from modules.curobo.scene_cache import mesh_stores, prune_inactive_meshes


class BoundedPlanner:
    def __init__(self, planner, root):
        self.planner, self.root, self.requests = planner, Path(root), 0

    def collect(self):
        import torch
        return prune_inactive_meshes(mesh_stores(self.planner), torch.cuda.synchronize)

    def request(self, directory, **flags):
        import torch
        before = self.collect()
        result = self.planner.request(directory, **flags)
        after = self.collect()
        self.requests += 1
        free, total = torch.cuda.mem_get_info()
        atomic_json(self.root/'mesh_cache_lifecycle.json', {
            'at': now(), 'requests_completed': self.requests,
            'before': before, 'after': after,
            'device_free_bytes': free, 'device_total_bytes': total,
            'geometry_collision_and_solver_settings_changed': False})
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', required=True)
    for name in ['root', 'socket', 'warmup-trial', 'descriptor']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    worker = BoundedPlanner(create_worker(args.kind, args), args.root)
    worker.collect()
    serve(args, worker)


if __name__ == '__main__':
    main()
