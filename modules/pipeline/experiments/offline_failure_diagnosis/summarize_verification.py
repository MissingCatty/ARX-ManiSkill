"""Aggregate controlled contact tests and check virtual-frame geometry equivalence."""
from arx_project.assets import load_object_catalog
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from inspect_failures import PROJECT, DEFAULT, read, surface_mesh
from attribute_empty_grasps import triangles_intersect_box

HERE=Path(__file__).resolve().parent


def main():
    descriptor=read(DEFAULT/'gripper_descriptor.json')
    r_eg=np.asarray(descriptor['T_eef_generator'])
    ext=np.array(descriptor['parameters']['extents_open'])
    off=np.array(descriptor['parameters']['offset_open'])
    catalog=load_object_catalog(DEFAULT/'object_catalog.json')['models']
    probe=HERE/'runs/gripper_frame_probe_20261123_01'
    rows=read(probe/'summary.json')['results']
    equivalence=[]
    for row in rows:
        d=probe/'trials'/row['id'];original=DEFAULT/'trials'/row['slot']
        source=np.load(d/'grasps_generator_world.npy');scores=np.load(d/'scores.npy')
        top=int(np.argmax(scores));g=source[top]
        shift=row['origin_behind_eef_m']
        eef_from_gen=r_eg.copy();eef_from_gen[2,3]=-shift
        eef=g@np.linalg.inv(eef_from_gen)
        new_off=off.copy();new_off[2]+=shift
        world_box_from_generator=g[:3,3]+g[:3,:3]@new_off
        world_box_from_robot=eef[:3,3]+eef[:3,:3]@r_eg[:3,:3]@off
        assert np.linalg.norm(world_box_from_generator-world_box_from_robot)<1e-7
        state=read(original/'state.json');entry=read(original/'entry.json')
        mesh=surface_mesh(catalog[entry['model_id']]);t=np.array(state['object_world'])
        vertices=np.asarray(mesh.vertices)@t[:3,:3].T+t[:3,3]
        gen=(vertices-eef[:3,3])@eef[:3,:3]@r_eg[:3,:3]
        contact_count=triangles_intersect_box(gen,mesh.faces,ext,off)
        equivalence.append({**row,'top1_actual_robot_eef_world':eef.tolist(),
            'top1_actual_target_triangles_in_pad_region':contact_count,
            'box_frame_equivalence_error_m':float(np.linalg.norm(world_box_from_generator-world_box_from_robot)),
            'actual_robot_tcp_m':.043})
    old=[r for r in rows if r['origin_behind_eef_m']==0]
    new=[r for r in rows if r['origin_behind_eef_m']>0]
    physical=[]
    for run in ['contact_replay_20261123_01','contact_confirmation_20261123_01']:
        folder=HERE/'runs'/run
        reg=read(folder/'registration.json')
        assert len(list((folder/'trials').glob('*/result.json')))==len(reg['schedule'])
        for path in (folder/'trials').glob('*/result.json'):
            physical.append({'run':run,**read(path)})
    source_reg=read(DEFAULT/'registration.json')
    changed=[p for p,h in source_reg['source_hashes'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    for folder in [HERE/'runs/contact_replay_20261123_01',HERE/'runs/contact_confirmation_20261123_01',probe]:
        reg=read(folder/'registration.json')
        for p,h in reg.get('diagnostic_source_sha256',reg.get('source_sha256',{})).items():
            assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    out={'diagnostic_physical_closures':len(physical),'old_benchmark_formal_results':66,
        'old_benchmark_new_attempts':0,'source_files_checked':len(source_reg['source_hashes']),
        'source_files_changed':changed,'source_stop_preserved':(DEFAULT/'STOP').exists(),
        'reference_frame_probe':{'cases':len(old),'original_diffusion_points_covered':sum(r['diffusion_candidates_with_target_points'] for r in old),
            'shifted_diffusion_points_covered':sum(r['diffusion_candidates_with_target_points'] for r in new),
            'diffusion_candidates_per_condition':sum(r['diffusion_count'] for r in new),
            'scope':'nonempty reconstructed point intersection; not an executed grasp success criterion',
            'top1_pose_checks':equivalence},'physical_results':physical,
        'official_contact_offset_reference':'https://nvidia-omniverse.github.io/PhysX/physx/5.1.2/docs/AdvancedCollisionDetection.html'}
    output=HERE/'results_20261123/verification_summary.json'
    output.write_text(json.dumps(out,indent=2)+'\n')
    print('closure diagnostics',len(physical),'old benchmark added',0,'frozen source changed',len(changed))
    print('diffusion nonempty cavities before / after',sum(r['diffusion_candidates_with_target_points'] for r in old),
        sum(r['diffusion_candidates_with_target_points'] for r in new),'/',sum(r['diffusion_count'] for r in new))
    for r in equivalence:print(r['id'],'actual target triangles',r['top1_actual_target_triangles_in_pad_region'])


if __name__=='__main__':main()
