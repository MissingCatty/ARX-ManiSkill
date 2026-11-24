"""Build a frozen ARX FK witness map and orientation coverage statistics."""
import argparse
from collections import Counter
import csv
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc
import yaml

from 公共 import ROOT, MS, REFERENCE, MODEL, URDF, TCP, JOINTS, FIELDS, ORIENTATIONS, DOWN_ORIENTATIONS
from 公共 import read, save, sha, now, orientation_ids, voxel_ids, unpack_voxels, static_scene
from 机器人几何 import RobotGeometry


def aggregate(out, bank, resolution, name):
    ids, _ = voxel_ids(bank[:, 6:9], resolution)
    rotations = Rotation.from_quat(np.asarray(bank[:, 9:13])[:, [1, 2, 3, 0]]).as_matrix()
    orientation, down = orientation_ids(rotations)
    keys = ids*ORIENTATIONS+orientation
    seen = np.unique(keys)
    unique, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    orient_ids, orient_counts = np.unique(seen//ORIENTATIONS, return_counts=True)
    assert np.array_equal(unique, orient_ids)
    seen_direction = (seen % ORIENTATIONS)//12 % 8
    down_ids, down_counts = np.unique((seen//ORIENTATIONS)[seen_direction < 2], return_counts=True)
    down_total = np.zeros(len(unique), dtype=np.int16)
    down_total[np.searchsorted(unique, down_ids)] = down_counts
    best_margin = np.zeros(len(unique), dtype=np.float32);best_condition = best_margin.copy()
    np.maximum.at(best_margin, inverse, bank[:, 13]);np.maximum.at(best_condition, inverse, bank[:, 15])
    mean_margin = np.bincount(inverse, weights=bank[:, 13])/counts
    mean_condition = np.bincount(inverse, weights=bank[:, 15])/counts
    path = out / (name + '.npz')
    np.savez_compressed(path, voxel_id=unique, cells=unpack_voxels(unique), sample_count=counts,
                        orientation_count=orient_counts.astype(np.int16), down_orientation_count=down_total,
                        margin_mean_rad=mean_margin.astype(np.float32), margin_best_rad=best_margin,
                        condition_mean=mean_condition.astype(np.float32), condition_best=best_condition,
                        resolution_m=np.array(resolution))
    if resolution == .02:np.save(out/'已覆盖位姿格.npy', seen, allow_pickle=False)
    return {'file': path.name, 'resolution_m': resolution, 'voxels': len(unique),
            'observed_pose_bins': len(seen), 'sample_count_sum': int(counts.sum()),
            'maximum_observed_orientation_fraction': float(orient_counts.max()/ORIENTATIONS),
            'maximum_down_orientation_fraction': float(down_total.max()/DOWN_ORIENTATIONS)}


def independent_check(out, geometry, bank, checker, torch, tcp):
    sys.path.insert(0, str(MS/'arx_pipeline'))
    from scene_tools import ArmFK
    fk = ArmFK();rng = np.random.default_rng(91120)
    indices = rng.choice(len(bank), size=min(256, len(bank)), replace=False)
    rows = np.asarray(bank[indices]);p_errors=[];r_errors=[]
    for row in rows:
        transform = fk.fk(row[:6]) @ tcp
        p_errors.append(np.linalg.norm(transform[:3, 3]-row[6:9]))
        rotation = Rotation.from_quat(row[9:13][[1,2,3,0]]).as_matrix()
        r_errors.append(Rotation.from_matrix(transform[:3, :3].T@rotation).magnitude())
    assert max(p_errors) < 2e-5 and max(r_errors) < 2e-4
    q = torch.as_tensor(rows[:16, :6], device='cuda')
    _, _, _, linear, angular = geometry.fk(q)
    maximum_jac = 0.
    for i, row in enumerate(rows[:16]):
        for j in range(6):
            plus = row[:6].astype(float).copy();minus=plus.copy();plus[j]+=1e-5;minus[j]-=1e-5
            a=fk.fk(plus)@tcp;b=fk.fk(minus)@tcp
            dp=(a[:3,3]-b[:3,3])/2e-5
            dr=Rotation.from_matrix(a[:3,:3]@b[:3,:3].T).as_rotvec()/2e-5
            maximum_jac=max(maximum_jac,float(np.abs(dp-linear[i,:,j].cpu().numpy()).max()),
                            float(np.abs(dr-angular[i,:,j].cpu().numpy()).max()))
    assert maximum_jac < 3e-4, maximum_jac
    assert bool(checker.validate(torch.as_tensor(rows[:, :6].copy(), device='cuda')[:,None]).all())
    result={'passed':True,'independent_fk_samples':len(rows),'maximum_position_error_m':float(max(p_errors)),
            'maximum_orientation_error_rad':float(max(r_errors)),'jacobian_finite_difference_samples':16,
            'maximum_jacobian_element_error':maximum_jac,'original_checker_revalidation_passed':True,
            'physical_attempts':0}
    save(out/'独立模型核验.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--输出',type=Path,required=True);parser.add_argument('--采样幂',type=int,default=22)
    parser.add_argument('--批大小',type=int,default=8192);parser.add_argument('--种子',type=int,default=20261120)
    parser.add_argument('--显卡',required=True);args=parser.parse_args()
    assert 10 <= args.采样幂 <= 24 and args.批大小>0
    os.environ['CUDA_VISIBLE_DEVICES']=args.显卡
    out=args.输出.resolve();out.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__).resolve(),Path(__file__).with_name('公共.py'),Path(__file__).with_name('机器人几何.py'),
             MODEL,URDF,TCP,REFERENCE/'state.json']
    frozen={str(p):sha(p) for p in sources}
    state=read(REFERENCE/'state.json');tcp=np.asarray(read(TCP)['T_eef_action'])
    assert np.allclose(tcp[:3,3],[0,0,.043])
    snapshot=out/'配置快照';snapshot.mkdir()
    for p in [MODEL,URDF,TCP]:shutil.copy2(p,snapshot/p.name)
    for p in sources[:3]:shutil.copy2(p,snapshot/p.name)
    save(snapshot/'静态场景.json',static_scene(state))
    save(out/'原始依据.json',{'time':now(),'source_sha256':frozen,'reference_state':str(REFERENCE/'state.json'),
        'sampling':'scrambled Sobol in unchanged six-joint limits with original 1e-4 rad inset',
        'seed':args.种子,'requested_samples':2**args.采样幂,'batch_size':args.批大小,'gpu_uuid':args.显卡,
        'current_clutter_used_in_generic_map':False,'physical_attempts':0,'robot_motion_commands':0,
        'reference_state_used_only_for_base_table_and_finger_calibration':True,
        'query_grasp_solutions_added_to_map':False})
    import torch
    from curobo.collision_checking import RobotCollisionChecker,RobotCollisionCheckerCfg
    began=time.monotonic();geometry=RobotGeometry(tcp,device='cuda')
    actual_limits=np.array([state['joint_limits'][state['joint_names'].index(n)] for n in JOINTS])
    assert np.abs(actual_limits-np.stack([geometry.lower,geometry.upper],axis=1)).max()<1e-5
    cfg=RobotCollisionCheckerCfg.load_from_config(robot_config=str(MODEL),scene_model=static_scene(state),
            n_meshes=1,n_cuboids=8,collision_activation_distance=0.)
    checker=RobotCollisionChecker(cfg);checker.self_collision_cost=None
    engine=qmc.Sobol(d=6,scramble=True,seed=args.种子)
    lower=geometry.lower+1e-4;upper=geometry.upper-1e-4
    total=2**args.采样幂;accepted=0;bound_rejected=0;static_rejected=0;max_fk_error=0.
    source_q=np.array([state['qpos'][state['joint_names'].index(n)] for n in JOINTS],dtype=np.float32)
    checker.validate(torch.as_tensor(source_q[None,None],device='cuda'))
    startup=time.monotonic()-began
    save(out/'组件就绪.json',{'time':now(),'static_robot_load_count':1,'startup_seconds':startup,
         'self_collision_policy':'disabled, matching original ARX pipeline','static_collision_policy':'original measured-open-finger spheres and original table/ground; activation distance 0',
         'physical_attempts':0})
    print('READY',args.显卡,'samples',total,'startup_seconds',round(startup,2),flush=True)
    path=out/'关节见证.bin'
    with path.open('xb') as stream,torch.no_grad():
        for offset in range(0,total,args.批大小):
            count=min(args.批大小,total-offset)
            qnp=(lower+engine.random(count)*(upper-lower)).astype(np.float32)
            q=torch.as_tensor(qnp,device='cuda')
            kin=checker.get_kinematics(q[:,None])
            eef_pos=kin.tool_poses.position.reshape(count,3).detach().clone()
            eef_quat=kin.tool_poses.quaternion.reshape(count,4).detach().clone()
            valid=checker.validate(q[:,None]).reshape(count)
            bound=(checker.get_bound(q[:,None]).sum(-1).reshape(count)==0)
            bound_rejected+=int((~bound).sum());static_rejected+=int((bound&~valid).sum())
            selected=valid.nonzero().flatten()
            if len(selected):
                accepted_q=q[selected]
                p,rotation,condition,_,_=geometry.fk(accepted_q)
                qcpu=accepted_q.cpu().numpy();native_r=Rotation.from_quat(eef_quat[selected].cpu().numpy()[:,[1,2,3,0]]).as_matrix()
                native_p=eef_pos[selected].cpu().numpy()+native_r@tcp[:3,3]
                max_fk_error=max(max_fk_error,float(np.linalg.norm(p.cpu().numpy()-native_p,axis=1).max()))
                assert max_fk_error<2e-5
                native_r=native_r@tcp[:3,:3];quaternion=Rotation.from_matrix(native_r).as_quat()[:,[3,0,1,2]]
                margins=np.minimum(qcpu-geometry.lower,geometry.upper-qcpu)
                margin=margins.min(-1);normalized=(margins/(geometry.upper-geometry.lower)).min(-1)
                data=np.column_stack([qcpu,native_p,quaternion,margin,normalized,condition.cpu().numpy()]).astype('<f4')
                assert np.isfinite(data).all()
                stream.write(data.tobytes());accepted+=len(data)
            if (offset//args.批大小)%16==0 or offset+count==total:
                elapsed=time.monotonic()-began
                save(out/'当前进度.json',{'phase':'采样','processed':offset+count,'requested':total,'accepted':accepted,
                     'wall_seconds':elapsed,'sampling_seconds':elapsed-startup,'physical_attempts':0})
                print('SAMPLING',offset+count,'/',total,'valid',accepted,'seconds',round(elapsed,1),flush=True)
        stream.flush();os.fsync(stream.fileno())
    assert accepted>0 and path.stat().st_size==accepted*len(FIELDS)*4
    save(out/'当前进度.json',{'phase':'合并体素','processed':total,'requested':total,'accepted':accepted,'physical_attempts':0})
    bank=np.memmap(path,dtype='<f4',mode='r',shape=(accepted,len(FIELDS)))
    fine=aggregate(out,bank,.02,'精细体素统计');coarse=aggregate(out,bank,.05,'概览体素统计')
    check=independent_check(out,geometry,bank,checker,torch,tcp)
    meta={'schema_version':1,'created_at':now(),'frame':'body4','pose_frame':'action_tcp43',
          'T_eef_action':tcp.tolist(),'reference_base_world':state['links']['body4'],
          'static_world_cuboids':state['scene_cuboids'],'lift_joint_m':state['qpos'][state['joint_names'].index('joint4')],
          'joint_names':JOINTS,'joint_limits':np.stack([geometry.lower,geometry.upper],axis=1).tolist(),
          'fields':FIELDS,'dtype':'<f4','witness_file':path.name,'witnesses':accepted,'sampled_joint_states':total,
          'bounds_rejected':bound_rejected,'static_scene_rejected':static_rejected,'seed':args.种子,
          'orientation_grid':{'azimuth_bins':12,'cos_polar_bins':8,'roll_bins':12,'total_bins':ORIENTATIONS,
             'downward_bins':DOWN_ORIENTATIONS,'downward_definition':'approach z dot body4 +Z < -0.5; within 60 degrees of down'},
          'coverage_semantics':'Observed orientation-bin fraction inside a voxel; sample-dependent lower bound, not true reachability or grasp probability',
          'uncovered_semantics':'unknown, never a proof of unreachable and never a grasp rejection rule',
          'quality_semantics':{'jacobian_condition_index':'sigma_min/sigma_max of [J_position; 0.1m * J_rotation]',
                               'joint_margin_rad':'minimum absolute distance to any arm joint limit'},
          'fine':fine,'coarse':coarse,'startup_seconds':startup,'wall_seconds':time.monotonic()-began,
          'maximum_gpu_fk_position_difference_m':max_fk_error,'physical_attempts':0,'clutter_checked':False}
    save(out/'地图元数据.json',meta)
    assert all(sha(p)==h for p,h in frozen.items()),'Frozen source changed during map build'
    checksums={p.name:sha(p) for p in [path,out/'精细体素统计.npz',out/'概览体素统计.npz',out/'已覆盖位姿格.npy',out/'地图元数据.json']}
    save(out/'完成.json',{'passed':True,'source_hashes_unchanged':True,'independent_checks':check,
         'file_sha256':checksums,'physical_attempts':0,'completed_at':now()})
    save(out/'当前进度.json',{'phase':'完成','processed':total,'accepted':accepted,'wall_seconds':time.monotonic()-began,'physical_attempts':0})
    print('COMPLETE',accepted,'fine_voxels',fine['voxels'],'coarse_voxels',coarse['voxels'],flush=True)


if __name__=='__main__':main()
