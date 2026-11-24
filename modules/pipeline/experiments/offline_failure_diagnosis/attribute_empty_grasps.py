"""Independent URDF FK and exact triangle/box tests for saved grasp endpoints."""
from arx_project.assets import load_object_catalog
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

from inspect_failures import DEFAULT, PROJECT, read, surface_mesh, observed


class FK:
    def __init__(self):
        root = ET.parse(PROJECT.parent/'maniskill/assets/arx/robot.urdf').getroot()
        self.by_child = {j.find('child').get('link'): j for j in root.findall('joint')}

    def at(self, name, q, base):
        if name not in self.by_child: return base.copy()
        j = self.by_child[name]
        t = np.eye(4)
        origin = j.find('origin')
        if origin is not None:
            t[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            t[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        motion = np.eye(4)
        if j.get('type') != 'fixed':
            axis = np.fromstring(j.find('axis').get('xyz', '1 0 0'), sep=' ')
            if j.get('type') == 'prismatic': motion[:3, 3] = axis*q[j.get('name')]
            else: motion[:3, :3] = Rotation.from_rotvec(axis*q[j.get('name')]).as_matrix()
        return self.at(j.find('parent').get('link'), q, base) @ t @ motion


def triangles_intersect_box(vertices, faces, extents, center):
    """13-axis separating-axis test; checks triangle interiors, not sampled points."""
    tri = vertices[np.asarray(faces)] - center
    half = extents/2
    overlap = np.ones(len(tri), dtype=bool)
    overlap &= np.all(tri.min(1) <= half+1e-8, axis=1)
    overlap &= np.all(tri.max(1) >= -half-1e-8, axis=1)
    edges = [tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 1], tri[:, 0]-tri[:, 2]]
    axes = [np.cross(edges[0], edges[1])]
    for edge in edges:
        for unit in np.eye(3): axes.append(np.cross(edge, unit))
    for axis in axes:
        proj = np.einsum('nki,ni->nk', tri, axis)
        radius = np.abs(axis) @ half
        overlap &= (proj.min(1) <= radius+1e-10) & (proj.max(1) >= -radius-1e-10)
    return int(overlap.sum())


def main():
    reg = read(DEFAULT/'registration.json')
    cases = read(PROJECT/'experiments/offline_failure_diagnosis/results_20261123/cases.json')
    catalog = load_object_catalog(DEFAULT/'object_catalog.json')['models']
    desc = read(DEFAULT/'gripper_descriptor.json')
    ext, off = [np.asarray(desc['parameters'][key]) for key in ['extents_open', 'offset_open']]
    r_eg = np.asarray(desc['T_eef_generator'])[:3, :3]
    fk, cache, result = FK(), {}, []
    for old in cases:
        d = DEFAULT/'trials'/old['id']
        entry, state = read(d/'entry.json'), read(d/'state.json')
        selected = read(d/'plan_result.json')['selected']
        rows = [json.loads(s) for s in (d/'execution.jsonl').open()]
        ix = next(i for i, r in enumerate(rows) if r['stage'] == 'close')
        before = rows[ix-1]
        q = dict(zip(state['joint_names'], before['qpos']))
        eef_actual = fk.at('right_eef_link', q, np.asarray(state['links']['base_link']))
        eef_goal = np.asarray(selected['grasp_world'])
        src_gen = np.asarray(selected['candidate_metadata']['source_generator_pose_world'])
        if entry['model_id'] not in cache: cache[entry['model_id']] = surface_mesh(catalog[entry['model_id']])
        mesh = cache[entry['model_id']]
        target = np.eye(4)
        pose = before['object_pose']
        target[:3, 3] = pose[:3]
        target[:3, :3] = Rotation.from_quat([*pose[4:7], pose[3]]).as_matrix()
        vertices = np.asarray(mesh.vertices) @ target[:3, :3].T + target[:3, 3]

        def triangle_count(eef):
            gen = (vertices-eef[:3, 3]) @ eef[:3, :3] @ r_eg
            return triangles_intersect_box(gen, mesh.faces, ext, off)

        pc = np.load(d/'completed_points_world.npy')
        pc_gen = (pc-src_gen[:3, 3]) @ src_gen[:3, :3]
        surface_gen = (vertices-src_gen[:3, 3]) @ src_gen[:3, :3]
        pmin, pmax = pc.min(0), pc.max(0)
        bbox_center_error = float(np.linalg.norm((pmin+pmax-vertices.min(0)-vertices.max(0))/2)*1000)
        row = {'id': old['id'], 'branch': old['branch'], 'reason': old['reason'],
            'independent_urdf_fk_position_error_mm': float(np.linalg.norm(eef_actual[:3, 3]-eef_goal[:3, 3])*1000),
            'independent_urdf_fk_rotation_error_deg': float(Rotation.from_matrix(eef_actual[:3, :3]@eef_goal[:3, :3].T).magnitude()*180/np.pi),
            'actual_eef_world': eef_actual.tolist(),
            'mesh_triangles_in_goal_pad_box': triangle_count(eef_goal),
            'mesh_triangles_in_actual_pad_box': triangle_count(eef_actual),
            'model_input_points_in_original_generator_cavity': int(np.all(np.abs(pc_gen-off)<=ext/2, axis=1).sum()),
            'target_nearest_z_in_generator_mm': float(surface_gen[:, 2].min()*1000),
            'fingertip_z_in_generator_mm': float(desc['parameters']['fingertip_depth']*1000),
            'reconstruction_bbox_center_error_mm': bbox_center_error,
            'perception_surface_error_median_mm': old['reconstruction_to_surface_median_mm'],
            'pad_center_to_surface_mm': old['pad_center_to_surface_mm']}
        result.append(row)
    summary = {'physical_records': len(result),
        'maximum_independent_fk_error_mm': max(r['independent_urdf_fk_position_error_mm'] for r in result),
        'maximum_independent_fk_rotation_deg': max(r['independent_urdf_fk_rotation_error_deg'] for r in result),
        'diffusion_selected': sum(r['branch']=='diff' for r in result),
        'diffusion_goal_mesh_disjoint': sum(r['branch']=='diff' and r['mesh_triangles_in_goal_pad_box']==0 for r in result),
        'diffusion_actual_mesh_disjoint': sum(r['branch']=='diff' and r['mesh_triangles_in_actual_pad_box']==0 for r in result),
        'diffusion_original_generator_cavity_empty': sum(r['branch']=='diff' and r['model_input_points_in_original_generator_cavity']==0 for r in result),
        'simulation_started': False, 'cases': result}
    out = Path(__file__).parent/'results_20261123/empty_grasp_attribution.json'
    out.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'}, indent=2))
    print('examples',json.dumps([r for r in result if r['id'] in ['D0002_S1_open_bottled_water','D0015_S1_open_remote_control','D0003_S1_open_mushroom']], indent=2))


if __name__ == '__main__': main()
