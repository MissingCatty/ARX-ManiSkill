"""ARX insertion truncation and orientation proposals; frozen collision model."""
from copy import deepcopy
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
import yaml

import graspcl_pipeline as common
import resident_planner as resident
from postprocess_planner import PostprocessPlanner

TCP_SOURCE = common.ROOT / 'curobo_adapter/research/20261117_07_action_tcp/action_tcp_43mm.json'
PROTOCOL = 'arx_tcp43_native_post_observed_insertion_truncation_v1'


def target_points(directory):
    params = common.read(directory / 'camera/params.json')
    k = params['intrinsics']
    depth = np.asarray(Image.open(directory / 'camera/depth.png'), dtype=float)*params['depth_scale']
    mask = np.asarray(Image.open(directory / 'camera/mask.png')) > 0
    v,u = np.indices(depth.shape)
    keep = mask & np.isfinite(depth) & (depth > .12) & (depth < 2.)
    points = np.stack([(u-k['ppx'])*depth/k['fx'],(v-k['ppy'])*depth/k['fy'],depth],-1)[keep]
    camera = np.asarray(common.read(directory / 'state.json')['camera_world_optical'])
    points = points @ camera[:3,:3].T + camera[:3,3]
    # Same 2 mm observation resolution as GraspCL. Deterministic, no GT mesh.
    if len(points):
        _,indices = np.unique(np.floor(points/.002).astype(np.int64),axis=0,return_index=True)
        points = points[np.sort(indices)]
    return points


def templates(context,state):
    """Gripper geometry in EEF coordinates, read from original live checkers."""
    q = context.measured(state)
    world_eef = np.asarray(state['links']['right_eef_link'])
    inv = np.linalg.inv(world_eef)
    result = []
    for checker,margin in [(context.checker,0.),(context.closed_checker,.008)]:
        kin = checker.get_kinematics(q[None,None])
        spheres = kin.robot_spheres.reshape(-1,4).detach().cpu().numpy().copy()
        params = checker.kinematics.kinematics_config
        names = {int(v):k for k,v in params.link_name_to_idx_map.items()}
        links = np.asarray([names[int(x)] for x in params.link_sphere_idx_map.cpu().tolist()])
        keep = np.isin(links,['right_link26','right_link27','right_link28']) & (spheres[:,3]>0)
        spheres = spheres[keep];links = links[keep]
        xyz = spheres[:,:3] @ context.base[:3,:3].T + context.base[:3,3]
        spheres[:,:3] = xyz @ inv[:3,:3].T + inv[:3,3]
        result.append((spheres,links,margin))
    return result


def box_clearance(spheres,grasp,geometry,margin):
    xyz = spheres[:,:3] @ grasp[:3,:3].T + grasp[:3,3]
    center = np.asarray(geometry['center']).copy();center[2] += margin
    delta = np.abs(xyz-center)-np.asarray(geometry['dims'])/2
    sdf = np.linalg.norm(np.maximum(delta,0),axis=-1)+np.minimum(np.max(delta,axis=-1),0)
    return float(np.min(sdf-spheres[:,3]))


def closing_sweep_points(points,grasp,template):
    spheres,links,_ = template
    bounds = []
    for name in ['right_link27','right_link28']:
        part = spheres[links==name]
        bounds.append((np.min(part[:,:3]-part[:,3,None],axis=0),
                       np.max(part[:,:3]+part[:,3,None],axis=0)))
    low = np.maximum(bounds[0][0],bounds[1][0])
    high = np.minimum(bounds[0][1],bounds[1][1])
    local = (points-grasp[:3,3]) @ grasp[:3,:3]
    keep = (local[:,0]>=low[0]) & (local[:,0]<=high[0]) & (local[:,2]>=low[2]) & (local[:,2]<=high[2])
    return local[keep]


def truncate(candidate,points,geometry,shapes,tcp,maximum_opening):
    """Minimum retraction on a 2 mm grid, bounded by original insertion depth."""
    pose = np.asarray(candidate['pose'],dtype=float)
    depth = float(candidate['metadata']['source_network_depth_m'])
    if not np.isfinite(depth) or depth < 0:
        return None,{'reason':'invalid_source_insertion_depth'}
    distances = np.unique(np.r_[np.arange(0,depth,.002),depth])
    initial_clearance = None
    for distance in distances:
        contact = pose.copy();contact[:3,3] -= distance*pose[:3,2]
        grasp = contact.copy();grasp[:3,3] -= tcp*pose[:3,2]
        clearance = min(box_clearance(s,grasp,geometry,m) for s,_,m in shapes)
        if initial_clearance is None:initial_clearance = clearance
        if clearance < 0:continue
        sweep = closing_sweep_points(points,grasp,shapes[0])
        width = candidate['width']
        if candidate['metadata'].get('orientation_change_rad',0)>1e-8:
            if not len(sweep):continue
            # Rotation changes the object's width in the closing direction.
            # This is observed local geometry, not the original orientation's width.
            width = float(2*np.max(np.abs(sweep[:,1])))
            if width<=0 or width>maximum_opening+1e-6:continue
        support = int(np.count_nonzero(np.abs(sweep[:,1])<=width/2))
        if support == 0:
            # With arbitrary orientation the observed overlap need not be monotonic.
            continue
        result = deepcopy(candidate);result['pose'] = contact;result['width'] = width
        result['point1'] = contact[:3,3]-result['width']/2*contact[:3,1]
        result['point2'] = contact[:3,3]+result['width']/2*contact[:3,1]
        result['metadata'].update(insertion_retraction_m=float(distance),
            remaining_source_insertion_m=float(depth-distance),
            source_network_depth_m=float(depth-distance),
            original_source_insertion_m=depth,original_contact_pose=pose.tolist(),
            observed_target_points_in_closing_sweep=support,
            gripper_table_clearance_before_m=initial_clearance,
            gripper_table_clearance_after_m=clearance,action_tcp_offset_eef_z_m=tcp,
            truncation_protocol=PROTOCOL)
        return result,{'retraction_m':float(distance),'observed_support':support}
    return None,{'reason':'no_table_clear_observed_closing_sweep_within_source_depth',
        'original_depth_m':depth,'initial_gripper_clearance_m':initial_clearance}


class TruncationPlanner(PostprocessPlanner):
    def select(self,state,rows,camera):
        # Reuse the exact native planner implementation with its single explicit
        # calibration literal changed in this separately registered class.
        return self.select_tcp43(state,rows,camera)


# A separate function body keeps all original validation/path/optimizer gates.
import inspect
_source = inspect.getsource(PostprocessPlanner.select)
import textwrap
_source = textwrap.dedent(_source).replace('def select(', 'def select_tcp43(').replace('.040 * contact', '.043 * contact')
_source = _source.replace("'zerograsp_native_post_no_custom_geometry_v1'",repr(PROTOCOL))
_scope = dict(__import__('postprocess_planner').__dict__)
exec(compile(_source,__file__+':registered_tcp43_selector','exec'),_scope)
TruncationPlanner.select_tcp43 = _scope['select_tcp43']


def make_pool(base):
    class TruncationPool(base):
        def __init__(self,root,warmup_trial):
            super().__init__(root,warmup_trial)
            profile = common.read(TCP_SOURCE)
            self.tcp_transform = np.asarray(profile['T_eef_action'])
            assert np.allclose(self.tcp_transform[:3,3],[0,0,.043])
            assert np.allclose(self.tcp_transform[:3,:3],[[0,-1,0],[1,0,0],[0,0,1]])
            from curobo.inverse_kinematics import InverseKinematics,InverseKinematicsCfg
            from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
            context = self.contexts['open']
            # A virtual TCP frame changes only the orientation proposal solver's
            # goal reference. The executable robot and collision model stay original.
            robot = yaml.safe_load(context.model.read_text())
            urdf = ET.parse(robot['robot_cfg']['kinematics']['urdf_path'])
            name = 'right_action_tcp43'
            ET.SubElement(urdf.getroot(),'link',name=name)
            joint = ET.SubElement(urdf.getroot(),'joint',name='proposal_tcp43_fixed',type='fixed')
            ET.SubElement(joint,'parent',link='right_eef_link')
            ET.SubElement(joint,'child',link=name)
            ET.SubElement(joint,'origin',xyz='0 0 0.043',rpy='0 0 1.5707963267948966')
            proposal_urdf = root / 'proposal_tcp43.urdf'
            urdf.write(proposal_urdf,encoding='unicode')
            robot['robot_cfg']['kinematics']['urdf_path'] = str(proposal_urdf)
            robot['robot_cfg']['kinematics']['tool_frames'] = [name]
            proposal_model = root / 'proposal_tcp43.yml'
            proposal_model.write_text(yaml.safe_dump(robot,sort_keys=False))
            self.proposal_frame = name
            cfg = InverseKinematicsCfg.create(robot=str(proposal_model),scene_model=None,
                optimizer_configs=['ik/lbfgs_ik.yml'],self_collision_check=False,
                num_seeds=32,seed_solver_num_seeds=32,position_tolerance=.002,
                orientation_tolerance=1e6,optimizer_collision_activation_distance=.003,
                max_batch_size=64,use_cuda_graph=True,load_collision_spheres=False)
            for rollout in [cfg.core_cfg.metrics_rollout_config,*cfg.core_cfg.optimizer_rollout_configs]:
                for cost in rollout.get_cost_manager_configs():cost.disable_self_collision()
            self.position_solver = InverseKinematics(cfg)
            self.position_solver.update_tool_pose_criteria({name:ToolPoseCriteria.track_position()})
            # Locked finger positions do not affect EEF or action TCP FK. The
            # proposal solver has no finger collision geometry or scene to rebind.
            state = common.read(warmup_trial / 'state.json')
            self.orientation_proposals(context,[np.asarray(state['links']['right_eef_link']) @ self.tcp_transform],context.measured(state))
            common.save(root / 'truncation_components_ready.json',{'action_tcp':profile,
                'position_only_solver_load_count':1,'position_only_warmed':True,
                'position_goal_frame':name,'virtual_goal_frame_model_sha256':common.sha(proposal_model),
                'position_only_used_for_proposals_only':True,'final_full_scene_checks_unchanged':True,
                'physical_startup_attempts':0})

        def orientation_proposals(self,context,poses,start):
            from curobo.types import GoalToolPose,JointState
            results = []
            self.position_solver.reset_seed()
            for offset in range(0,len(poses),64):
                part = poses[offset:offset+64];n = len(part)
                padded = part+[part[-1]]*(64-n)
                local = np.asarray([context.invbase @ p for p in padded])
                quat = Rotation.from_matrix(local[:,:3,:3]).as_quat()[:,[3,0,1,2]]
                goal = GoalToolPose(tool_frames=[self.proposal_frame],
                    position=torch.tensor(local[:,:3,3],dtype=torch.float32,device='cuda').reshape(64,1,1,1,3),
                    quaternion=torch.tensor(np.ascontiguousarray(quat),dtype=torch.float32,device='cuda').reshape(64,1,1,1,4).contiguous())
                js = JointState.from_position(start[None].repeat(64,1),joint_names=resident.frozen.NAMES)
                solved = self.position_solver.solve_pose(goal,current_state=js,return_seeds=32)
                qs = solved.solution.reshape(64,32,6)[:n]
                ok = solved.success.reshape(64,32)[:n]
                for source_pose,solutions,valid in zip(part,qs,ok):
                    rotations = []
                    if valid.any():
                        kin = context.checker.get_kinematics(solutions[valid][:,None])
                        quats = kin.tool_poses.quaternion.detach().cpu().numpy().reshape(-1,4)
                        rotations = context.base[:3,:3] @ Rotation.from_quat(quats[:,[1,2,3,0]]).as_matrix()
                        xyz = kin.tool_poses.position.detach().cpu().numpy().reshape(-1,3)
                        eef_world = xyz @ context.base[:3,:3].T+context.base[:3,3]
                        actual_tcp = eef_world+.043*rotations[:,:,2]
                        assert np.linalg.norm(actual_tcp-source_pose[:3,3],axis=-1).max()<.00205,'Position IK goal must be the actual TCP43, not EEF origin'
                    results.append(rotations)
            return results

        def refine(self,directory,mode):
            started = time.perf_counter()
            state = common.read(directory / 'state.json')
            context = self.bind(directory / 'refinement_model',state)
            pool_path = directory / 'graspcl/candidate_pool.json'
            pool = common.read(pool_path)
            common.save(pool_path.with_name('unrefined_candidate_pool.json'),pool)
            shapes = templates(context,state)
            points = target_points(directory)
            original = pool['candidates'];proposed = [];prepared = deepcopy(original)
            for candidate in prepared:
                if candidate['metadata']['origin'].startswith('demonstration_'):
                    pose = np.asarray(candidate['pose'],dtype=float)
                    pose[:3,3] += .003*pose[:3,2]
                    candidate['pose'] = pose
                    candidate['metadata']['demonstration_tcp40_to_tcp43_translation_m'] = .003
            free = self.orientation_proposals(context,[np.asarray(c['pose']) for c in prepared],context.measured(state)) if mode=='free_orientation' and prepared else [[] for _ in prepared]
            for index,(source,rotations) in enumerate(zip(prepared,free)):
                candidate = deepcopy(source)
                pose = np.asarray(candidate['pose'],dtype=float)
                candidate['pose'] = pose
                candidate['metadata']['source_pool_index'] = index
                candidate['metadata']['orientation_mode'] = mode
                seen = set()
                for rotation in [pose[:3,:3],*rotations]:
                    key = tuple(np.round(rotation,5).flatten())
                    if key in seen:continue
                    seen.add(key)
                    changed = deepcopy(candidate);changed['pose'] = pose.copy()
                    changed['pose'][:3,:3] = rotation
                    changed['metadata']['orientation_change_rad'] = float(Rotation.from_matrix(rotation @ pose[:3,:3].T).magnitude())
                    proposed.append(changed)
            candidates = [];rejected = []
            maximum = sum(state['joint_limits'][state['joint_names'].index(n)][1] for n in resident.frozen.FINGERS)
            for candidate in proposed:
                result,receipt = truncate(candidate,points,state['scene_cuboids']['table'],shapes,.043,maximum)
                if result is None:rejected.append({'source_pool_index':candidate['metadata']['source_pool_index'],**receipt})
                else:candidates.append(result)
            pool.update(candidates=candidates,candidate_protocol=PROTOCOL,
                refinement={'mode':mode,'input_candidates':len(original),'orientation_proposals':len(proposed),
                    'retained':len(candidates),'rejected':rejected,'grid_m':.002,
                    'maximum_retraction':'original candidate insertion depth; no extension behind its source entry surface',
                    'observed_contact_gate':'Necessary observed-target intersection of the physical ARX finger closing sweep; bilateral contact is validated in simulation.',
                    'position_only_is_not_execution_approval':True,'table_closed_margin_m':.008,
                    'action_tcp_profile_sha256':common.sha(TCP_SOURCE),'wall_seconds':time.perf_counter()-started})
            common.save(pool_path,pool)
            common.save(directory / 'truncation.json',pool['refinement'])
            return {'success':True,'mode':mode,'input_candidates':len(original),
                'orientation_proposals':len(proposed),'retained':len(candidates),
                'wall_seconds':time.perf_counter()-started}

        def request(self,directory,refine=False,orientation_mode='fixed_orientation',**kwargs):
            if refine:return self.refine(Path(directory),orientation_mode)
            result = super().request(directory,**kwargs)
            if result.get('selected'):
                selected = result['selected']
                eef = np.asarray(selected['grasp_world'])
                action = eef @ self.tcp_transform
                assert np.allclose(action[:3,3],np.asarray(selected['contact_pose_world'])[:3,3],atol=1e-9)
                selected.update(action_tcp_profile=str(TCP_SOURCE),action_tcp_world=action.tolist(),
                    T_eef_action=self.tcp_transform.tolist(),action_tcp_offset_eef_z_m=.043)
                common.save(Path(directory) / 'plan_result.json',result)
            return result
    return TruncationPool
