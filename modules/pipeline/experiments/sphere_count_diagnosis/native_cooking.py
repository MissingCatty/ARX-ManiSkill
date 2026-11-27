"""Export PhysX convex cooking results on CPU without creating a scene."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from arx_project.assets import resource_path
from arx_project.layout import ROOT, now, sha256, write_json, ensure_run_path
from modules.curobo.fingertip_contact.geometry import origin


def export(root):
    import sapien
    if root.exists():
        raise RuntimeError('Keep prior cooked geometry exports')
    root.mkdir(parents=True)
    asset = resource_path('arx_robot')
    tree = ET.parse(asset/'robot.urdf').getroot()
    links = {link.get('name'): link for link in tree.findall('link')}
    material = sapien.physx.PhysxMaterial(.8, .8, 0.)
    inputs = {str(asset/'robot.urdf'): sha256(asset/'robot.urdf')}
    records = []
    for arm in ['right', 'left']:
        for number in range(21, 29):
            canonical = 'right_link'+str(number)
            physical = canonical if arm == 'right' else 'left_link'+str(number-10)
            parts = {};sources = []
            for i, collision in enumerate(links[physical].findall('collision')):
                mesh = collision.find('geometry/mesh')
                path = asset/mesh.get('filename')
                scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
                shape = sapien.physx.PhysxCollisionShapeConvexMesh(str(path), scale.astype(np.float32), material)
                # get_vertices returns the mesh's unscaled local vertices.
                vertices = np.asarray(shape.get_vertices(), dtype=float)*scale
                transform = origin(collision.find('origin'))
                parts['part_'+str(i)] = vertices@transform[:3, :3].T+transform[:3, 3]
                inputs[str(path)] = sha256(path)
                sources.append({'file': str(path), 'native_cooked_vertices': len(vertices),
                    'native_affine_rank': int(np.linalg.matrix_rank(vertices-vertices.mean(0)))})
            target = root/arm/(canonical+'.npz');target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **parts)
            records.append({'arm': arm, 'physical_link': physical, 'canonical_link': canonical,
                'file': str(target.relative_to(ROOT)), 'sha256': sha256(target),
                'parts': len(parts), 'sources': sources})
            print('COOKED', arm, canonical, len(parts), 'parts', flush=True)
    write_json(root/'index.json', {'at': now(), 'native_geometry_source': 'SAPIEN PhysxCollisionShapeConvexMesh',
        'simulation_scene_created': False, 'renderer_created': False, 'GPU_used': False,
        'physical_grasps': 0, 'planning_requests': 0, 'same_URDF_mesh_scale_and_origin': True,
        'records': records, 'input_sha256': inputs, 'export_source_sha256': sha256(Path(__file__))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    export(ensure_run_path(args.root))
