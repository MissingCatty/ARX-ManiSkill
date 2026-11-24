"""Export LiLo-native trajectories and a conditional offline entry index.

The entry index records teacher support, never a VLA success probability.
"""
import argparse
from pathlib import Path
import json
import numpy as np
from scripts.curobo_arx.data_factory.common import read,sha,write
from scripts.curobo_arx.data_factory.export import export as base_export
from scripts.curobo_arx.data_factory.geometry import rigid,grasp_comparison_frame


def entry_record(root):
    root=Path(root);meta=read(root/'episode.json');quality=read(root/'quality_receipt.json');physical=read(root/'physical_receipt.json')
    if not meta['write_complete'] or not quality['pass'] or not physical['pass']:
        raise ValueError('Entry index requires complete accepted physical teacher episode')
    actual=meta['actual_entry'];snapshot=actual['snapshot']
    g=rigid(meta['goal']['grasp_world']);h=rigid(meta['sampled_entry']['world'])
    if 'pad_center_eef_m' in meta['goal']:
        contact=grasp_comparison_frame(meta['goal'])
        if not np.allclose(contact[:3,3],meta['goal']['target_contact_world'],atol=1e-8,rtol=0):
            raise ValueError('SDK contact-center metadata differs from executed grasp; exclude this source version from production export')
    measured=rigid(snapshot['link_world']['right_eef_link'])
    d=np.linalg.inv(g)@h;dm=np.linalg.inv(g)@measured
    if not np.allclose(g@dm,measured,atol=1e-8) or not np.allclose(g@d,h,atol=1e-8):
        raise ValueError('G/D/H frame convention does not reconstruct actual entry')
    joints=dict(zip(snapshot['joint_names'],snapshot['joint_positions']))
    camera=np.asarray(actual['camera_world_optical']);target=np.asarray(snapshot['objects']['test_object']['world'])
    center=np.linalg.inv(camera)@target[:,3];k=np.asarray(actual['intrinsics']);uv=(k@center[:3])[:2]/center[2]
    return {'schema':'arx_sdg_entry_index_v1','episode_id':meta['attempt_id'],'episode_path':str(root),
        'object_id':meta['object_id'],'geometry_group':meta['geometry_group'],'asset_sha256':meta['asset_sha256'],
        'split':meta['split'],'goal_id':meta['goal']['goal_id'],'source_goal_id':meta['goal'].get('source_goal_id',meta['goal']['goal_id']),
        'candidate_source':meta['goal'].get('candidate_source',{'name':'legacy_candidate_source'}),
        'contact_symmetry':meta['goal'].get('contact_symmetry','legacy_TCP'),
        'support_kind':'independent_successful_physical_teacher_demonstration',
        'VLA_checkpoint_evaluation':None,'VLA_success_label':None,
        'G_world':g.tolist(),'H_requested_world':h.tolist(),'H_actual_world':measured.tolist(),
        'D_requested':d.tolist(),'D_actual':dm.tolist(),'relative_frame':'D = inverse(G_world) @ H_world',
        'grasp_group':meta['goal']['group'],'approach_axis_G':[0,0,1],'closing_axis_G':[0,1,0],
        'distance_mm':meta['sampled_entry']['distance_mm'],'entry_sample':meta['sampled_entry'],
        'gripper_identity':'frozen_ARX_right_parallel_gripper','eef_link':'right_eef_link',
        'finger_joint_state_m':{n:joints[n] for n in ('right_joint27','right_joint28')},
        'opening_convention':'Original ARX prismatic joint metres, with unchanged gripper geometry; not another gripper width',
        'arm_joint_names':snapshot['right_arm'],'arm_joint_positions_rad':[joints[n] for n in snapshot['right_arm']],
        'camera_role':actual['camera_role'],'camera_intrinsics':actual['intrinsics'],
        'camera_world_optical':actual['camera_world_optical'],'frame_time':actual['frame_time'],
        'target_center_uv':uv.tolist(),'target_center_camera_z_m':float(center[2]),
        'visibility_scope':'Recorded center projection and original frustum gate; no additional occlusion certificate',
        'stage':'approach_start_before_command','entry_raw_observation':actual,
        'state_convention':'grasp_goal_world_error_v1','action_shape':[8,7],'sample_seconds':.2,
        'quality_receipt_sha256':sha(root/'quality_receipt.json'),'physical_receipt_sha256':sha(root/'physical_receipt.json'),
        'source_identity':meta['source_identity'],'renderer':meta['renderer']}


def export(campaign,out):
    campaign=Path(campaign).resolve();out=Path(out).resolve()
    base_export(campaign,out)
    manifest=read(out/'manifest.json');episodes={}
    for split in ('train','validation','development'):
        for line in (out/(split+'.jsonl')).open():
            row=json.loads(line);episodes[row['episode_id']]=row['episode_path']
    if len(episodes)!=manifest['episodes']:raise ValueError('Export/index episode membership mismatch')
    with (out/'entry_index.jsonl').open('x') as stream:
        for ident,path in episodes.items():
            record=entry_record(path)
            if record['episode_id']!=ident:raise ValueError('Entry index identity mismatch')
            stream.write(json.dumps(record,separators=(',',':'))+'\n')
    write(out/'entry_index_manifest.json',{'schema':'arx_sdg_entry_index_v1','entries':len(episodes),
        'index_sha256':sha(out/'entry_index.jsonl'),'lilo_manifest_sha256':sha(out/'manifest.json'),
        'scope':'Offline teacher-entry support only; no online adapter, no policy training or VLA success labels',
        'export_source':str(campaign),'source_registration_sha256':sha(campaign/'registration.json')})
    return {'episodes':len(episodes),'action_chunks':manifest['chunks'],'entry_index_entries':len(episodes)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign');p.add_argument('--out',required=True)
    a=p.parse_args();print(export(a.campaign,a.out))
