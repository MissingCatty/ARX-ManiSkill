"""Real pretrained Graspness inference, with no AnyGrasp/SDG fallback."""
from pathlib import Path
import sys
import time
import numpy as np
from scripts.curobo_arx.data_factory.common import read,write,sha
from scripts.curobo_arx.data_factory.geometry import grasp_pool,handoff,equivalent_grasp
from scripts.curobo_arx.data_factory.neural_input import cloud_input


class NoGrasps(ValueError):
    pass


def sample_cloud(points,count,seed):
    points=np.asarray(points,dtype=np.float32)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Invalid actual ARX point cloud')
    random=np.random.default_rng(seed)
    if len(points)>=count:indices=random.choice(len(points),count,replace=False)
    else:indices=np.r_[np.arange(len(points)),random.choice(len(points),count-len(points),replace=True)]
    return points[indices]


def candidates(rows,row,asset,config,visual_context=None,visual_geometry=None):
    rows=np.asarray(rows,dtype=float)
    if rows.ndim!=2 or rows.shape[1]!=17:raise ValueError('Graspness must return GraspNet [N,17] predictions')
    good=np.isfinite(rows).all(axis=1)&(rows[:,1]>0)&(rows[:,1]<=config['max_gripper_width_m'])
    rows=rows[good];raw_count=len(rows)
    camera=np.asarray(row['camera_world_optical']);obj=np.asarray(row['snapshot']['objects']['test_object']['world'])
    insertion=rows[:,13:16]+rows[:,3,None]*rows[:,[4,7,10]]
    contacts=insertion@camera[:3,:3].T+camera[:3,3]
    local=(contacts-obj[:3,3])@obj[:3,:3]
    rows=rows[(np.abs(local)<=np.asarray(asset['size_m'])/2+.015).all(axis=1)]
    rows=rows[np.argsort(-rows[:,0],kind='stable')][:config['candidate_limit']*4]
    pool=grasp_pool(rows,row['camera_world_optical'],row['snapshot']['objects']['test_object']['world'],
        asset['size_m'],maximum=len(rows))
    ordered=sorted(pool['candidates'],key=lambda g:-g['score']);safe=[];visual_rejections=[]
    if visual_context is not None:
        if visual_geometry is None:raise ValueError('Measured wrist context requires the actual target visual geometry')
        for goal in ordered:
            flips=[];reasons=[]
            for flip in (False,True):
                optical=equivalent_grasp(goal,flip)@np.asarray(visual_context['eef_from_camera'])
                assessment=visual_geometry.inspect(obj,optical,visual_context['intrinsics'],visual_context['shape'],visual_context['clipping_range_m'])
                if assessment['pass']:flips.append(flip)
                else:reasons.append({'flip':flip,'reasons':assessment['reasons']})
            if flips:goal['camera_safe_equivalent_flips']=flips;safe.append(goal)
            else:visual_rejections.append({'goal_id':goal['goal_id'],'trials':reasons})
    else:safe=ordered
    pool['candidates']=safe[:config['candidate_limit']]
    for goal in pool['candidates']:
        goal.update(candidate_source={'name':'Graspness / GSNet','repository':config['repository'],
            'revision':config['revision'],'checkpoint_sha256':config['checkpoint_sha256'],
            'input':'Fresh actual ARX calibrated head linear-Z depth, in metres'},
            pregrasp_distance_mm=config['pregrasp_distance_mm'])
        goal['pregrasp_world']=handoff(goal['grasp_world'],config['pregrasp_distance_mm'],0,0)['world']
    pool.update(candidate_source='Graspness / GSNet',pregrasp_distance_mm=config['pregrasp_distance_mm'],
        ranking='Target-associated high scores after source collision filtering and NMS; bounded before geometric deduplication',raw_predictions=raw_count)
    pool['camera_prefilter_rejections']=visual_rejections
    return pool


class Graspness:
    def __init__(self, registration, directory):
        import torch
        self.config=registration['graspness'];self.directory=Path(directory)
        source=Path(self.config['source'])
        for path in (source,source/'utils',source/'knn',source/'pointnet2'):sys.path.insert(0,str(path))
        from models.graspnet import GraspNet,pred_decode
        from dataset.graspnet_dataset import minkowski_collate_fn
        from collision_detector import ModelFreeCollisionDetector
        from loss_utils import GRASPNESS_THRESHOLD
        if sha(self.config['checkpoint'])!=self.config['checkpoint_sha256']:raise RuntimeError('Graspness checkpoint changed')
        self.torch=torch;self.decode=pred_decode;self.collate=minkowski_collate_fn
        self.collision=ModelFreeCollisionDetector
        torch.manual_seed(registration['seed'])
        self.net=GraspNet(seed_feat_dim=512,is_training=False).to('cuda:0')
        checkpoint=torch.load(self.config['checkpoint'],map_location='cpu',weights_only=True)
        self.net.load_state_dict(checkpoint['model_state_dict'],strict=True);self.net.eval()
        self.visual_geometry={}
        if self.config.get('multi_view'):
            def capture_crop(module,args):self.crop_inputs=(args[0].detach(),args[1].detach())
            self.net.crop.register_forward_pre_hook(capture_crop)
        # The unmodified source FPS operator cannot accept zero graspable points.
        # Reject that observation without launching an invalid CUDA kernel; keep
        # the source model's objectness and graspness thresholds unchanged.
        def no_empty_cloud(module,inputs,outputs):
            valid=(outputs['objectness_score'].argmax(1)==1)&(outputs['graspness_score'].squeeze(1)>GRASPNESS_THRESHOLD)
            if not valid.any():raise NoGrasps('Graspness predicted zero graspable points')
        self.net.graspable.register_forward_hook(no_empty_cloud)

    def handle(self, request):
        if request['action']!='infer':raise ValueError('Unsupported Graspness request')
        from graspnetAPI import GraspGroup
        directory=Path(request['directory']);row=read(directory/'head_observation.json');asset=request['asset']
        if row['object_id']!=asset['object_id'] or row['camera_role']!='head':raise ValueError('Graspness camera/target mismatch')
        if abs(row['frame_time']-row['capture_sim_time'])>1e-4:raise ValueError('Stale Graspness observation')
        began=time.monotonic()
        try:points,region=cloud_input(np.load(directory/'head_depth.npy',allow_pickle=False),row,asset)
        except ValueError as error:return {'candidates':[],'rejected':str(error),'candidate_source':'Graspness / GSNet'}
        sampled=sample_cloud(points,self.config['num_points'],self.config['seed'])
        np.save(directory/'graspness_input_points.npy',sampled)
        batch=self.collate([{'point_clouds':sampled,'coors':sampled/self.config['voxel_size_m'],
            'feats':np.ones_like(sampled,dtype=np.float32)}])
        batch={key:value.to('cuda:0') for key,value in batch.items()}
        self.torch.cuda.synchronize();network_started=time.monotonic()
        try:
            with self.torch.no_grad():
                endpoints=self.net(batch)
                rows=self.decode(endpoints)[0].detach().cpu().numpy()
        except NoGrasps as error:
            return {'candidates':[],'rejected':str(error),'candidate_source':'Graspness / GSNet'}
        self.torch.cuda.synchronize();network_seconds=time.monotonic()-network_started
        np.save(directory/'graspness_raw_grasps.npy',rows)
        group=GraspGroup(rows)
        collision=self.collision(points,voxel_size=self.config['collision_voxel_size_m'])
        colliding=collision.detect(group,approach_dist=.05,collision_thresh=self.config['collision_threshold'])
        group=group[~colliding]
        extension=self.config.get('multi_angle')
        if not len(group) and not extension:return {'candidates':[],'rejected':'All Graspness proposals rejected by source point-cloud collision filtering','candidate_source':'Graspness / GSNet'}
        if len(group):group=group.nms().sort_by_score()
        context=request.get('visual_context');geometry=None
        if context is not None:
            from scripts.curobo_arx.data_factory.visual_geometry import TargetGeometry
            key=asset['asset_sha256']
            if key not in self.visual_geometry:self.visual_geometry[key]=TargetGeometry.from_asset(asset)
            geometry=self.visual_geometry[key]
        pool=candidates(group.grasp_group_array,row,asset,self.config,context,geometry)
        if extension:
            from .multi_angle import decode,merge_pools
            from label_generation import batch_viewpoint_params_to_matrix
            from loss_utils import GRASP_MAX_WIDTH
            views=self.config.get('multi_view')
            if views:
                from .multi_view import proposals
                multi=proposals(self.net,endpoints,self.crop_inputs,views=views['views_per_point'],
                    angles=extension['angles_per_point'],minimum_degrees=views['minimum_angle_degrees'])[0].detach().cpu().numpy()
            else:
                multi=decode(endpoints,batch_viewpoint_params_to_matrix,
                    angles=extension['angles_per_point'],maximum_width=GRASP_MAX_WIDTH)[0].detach().cpu().numpy()
            # Verify the independent multi-roll decoder preserves official top1.
            if not np.array_equal(multi[:len(rows)],rows):
                raise RuntimeError('Multi-angle top1 differs from the pinned source decoder')
            np.save(directory/'graspness_multi_angle_grasps.npy',multi)
            np.savez_compressed(directory/'graspness_decoder_heads.npz',**{
                k:endpoints[k].detach().cpu().numpy() for k in
                ('xyz_graspable','grasp_score_pred','grasp_width_pred','grasp_top_view_xyz')})
            write(directory/'baseline_pool.json',pool)
            wider=GraspGroup(multi)
            extended_collision=np.concatenate([collision.detect(wider[i:i+256],approach_dist=.05,
                collision_thresh=self.config['collision_threshold']) for i in range(0,len(wider),256)])
            if not np.array_equal(extended_collision[:len(rows)],colliding):
                raise RuntimeError('Original source collision results changed under proposal batching')
            colliding=extended_collision
            wider=wider[~colliding]
            if len(wider):wider=wider.nms().sort_by_score()
            expanded=candidates(wider.grasp_group_array,row,asset,
                {**self.config,'candidate_limit':extension['candidate_limit']},context,geometry)
            pool=merge_pools(pool,expanded,extension['candidate_limit'])
            pool['proposal_extension'].update(angles_per_point=extension['angles_per_point'],
                source_top1_bitwise_equal=True,expanded_neural_rows=len(multi),
                same_forward_pass=True,learned_alternative_views=views,
                source_primary_collision_bitwise_equal=True)
        pool['timing']={'network_seconds':network_seconds,'total_seconds':time.monotonic()-began}
        pool['point_cloud']={'context_points':len(points),'target_points':int(region.sum()),
            'sampled_points':len(sampled),'source_depth_sha256':sha(directory/'head_depth.npy')}
        write(directory/'pool.json',pool)
        return pool
