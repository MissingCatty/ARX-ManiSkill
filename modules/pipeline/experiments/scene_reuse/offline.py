"""Compare cached arrays with the actual native Mesh loader on frozen inputs."""
import argparse
from pathlib import Path
import time

import numpy as np

from arx_project.layout import ROOT, ensure_run_path, now, read_json, sha256, write_json
from modules.curobo.scene_reuse.prepared_meshes import PreparedMeshCache

CONFIG = ROOT / 'configs/experiments/scene_reuse_cpu.json'


def run(output):
    from curobo._src.geom.types import Mesh
    cfg = read_json(CONFIG); source = ROOT / cfg['source_run']
    cache_root = ROOT / 'tmp/curobo/mesh_cache' / output.name / 'offline'
    cache = PreparedMeshCache(cache_root)
    rows = []; seen = set()
    for cid in cfg['cases']:
        for arm in ['right', 'left']:
            state = read_json(source / 'trials' / cid / 'optimized/arms' / arm / 'state.json')
            for name, geometry in state['scene_meshes'].items():
                path = Path(geometry['file_path']); digest = sha256(path)
                if digest != geometry['file_sha256']:
                    raise ValueError('Frozen collision geometry changed')
                if digest in seen: continue
                seen.add(digest)
                mesh = Mesh(name=name, file_path=str(path), pose=[0, 0, 0, 1, 0, 0, 0])
                began = time.perf_counter(); old = mesh.get_mesh_data(); original_seconds = time.perf_counter()-began
                cold = cache.get(mesh); warm = cache.get(mesh)
                disk_cache = PreparedMeshCache(cache_root); disk = disk_cache.get(mesh)
                for item in [cold, warm, disk]:
                    if not all(np.array_equal(a, b) and a.dtype == b.dtype
                               for a, b in zip(old, [item.vertices, item.faces])):
                        raise ValueError('CPU cache changed original native geometry')
                rows.append({'name': name, 'source': str(path.relative_to(ROOT)), 'sha256': digest,
                             'vertices': len(old[0]), 'triangles': len(old[1]),
                             'original_CPU_seconds': original_seconds, 'all_arrays_exact': True,
                             'cold_cache_seconds': cache.events[-2]['seconds'],
                             'memory_hit_seconds': cache.events[-1]['seconds'],
                             'disk_hit_seconds': disk_cache.events[-1]['seconds']})
    result = {'at': now(), 'passed': True, 'meshes': len(rows), 'rows': rows, 'cache': cache.report(),
              'new_GPU_calls': 0, 'new_planning_requests': 0,
              'config_sha256': sha256(CONFIG), 'source_run': cfg['source_run']}
    write_json(output / 'offline_review.json', result)
    print({'meshes': len(rows), 'arrays_exact': True}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    run(ensure_run_path(parser.parse_args().output))
