"""ZeroGrasp-derived hard filters with explicit ARX frame/width/depth adaptation."""
from copy import deepcopy
import time

import numpy as np
from modules.graspgenx.arx_frames import generator_to_contact,TCP_M
from modules.graspgenx.postprocessing.nms import native_nms
from .kernels import native_detect,variable_collision

VARIANTS=('native','native_rechecked','arx_observed')


def load_raw(directory,payload):
    directory=directory/'graspgenx_raw'
    generator=np.load(directory/'grasps_generator_world.npy',allow_pickle=False)
    scores=np.load(directory/'scores.npy',allow_pickle=False)
    if generator.shape!=(len(scores),4,4) or not np.isfinite(generator).all() or not np.isfinite(scores).all():
        raise ValueError('Invalid native GraspGenX outputs')
    order=np.argsort(-scores,kind='stable')
    if len(payload['candidates'])!=len(order):raise ValueError('Raw candidate truncation is not permitted')
    poses=np.array([generator_to_contact(generator[i]) for i in order]).reshape(-1,4,4)
    for rank,(index,candidate) in enumerate(zip(order,payload['candidates'])):
        if candidate['metadata']['source_generator_index']!=index or candidate['score']!=scores[index] or not np.allclose(candidate['pose'],poses[rank],atol=2e-9,rtol=0):
            raise ValueError('Raw array/native score rank/TCP43 provenance mismatch')
    return poses,scores[order],order


def hard_nms(poses,scores,widths,keep,translation,rotation):
    ids=np.flatnonzero(keep)
    result=native_nms(poses[ids],scores[ids],widths[ids],translation,np.deg2rad(rotation),ids)
    return result['representatives'],result


def table_bad(model,eef,widths,table):
    # Same V2 distal 10mm/2mm allowance, at each candidate's actual opening.
    center=np.asarray(table['center']);half=np.asarray(table['dims'])/2
    lower,upper=center-half,center+half
    bad=np.zeros(len(eef),bool)
    for part in model.parts:
        direction=0 if not part.finger else np.sign(part.vertices[:,1].mean())
        local=np.broadcast_to(part.vertices,(len(eef),len(part.vertices),3)).copy()
        local[:,:,1]+=(widths/2-.044)[:,None]*direction
        world=np.einsum('kij,kvj->kvi',eef[:,:3,:3],local)+eef[:,None,:3,3]
        for stage,values in [('pre',world-.08*eef[:,None,:3,2]),('grasp',world)]:
            allowance=np.where(part.finger&(part.vertices[:,2]>=model.tip-.010),.002,0.) if stage=='grasp' else np.zeros(len(part.vertices))
            inside_xy=((values[:,:,:2]>lower[:2]+1e-7)&(values[:,:,:2]<upper[:2]-1e-7)).all(2)
            inside_z=(values[:,:,2]>lower[2]+1e-7)&(values[:,:,2]<upper[2]-allowance[None]-1e-7)
            bad|=(inside_xy&inside_z).any(1)
    return bad


def pad_refinement(eef,target,cavity,empty_threshold=10):
    lower,upper=[np.asarray(v) for v in cavity]
    neutral=(lower[1]+upper[1])/2
    widths=np.full(len(eef),.088);advance=np.zeros(len(eef));counts=np.zeros(len(eef),int)
    depth_bad=np.zeros(len(eef),bool)
    # Only move along the unchanged approach axis, at most 5mm. No rotations.
    for k,pose in enumerate(eef):
        local=(target-pose[:3,3])@pose[:3,:3]
        inside=((local[:,0]>=lower[0])&(local[:,0]<=upper[0])&
                (local[:,1]>=lower[1])&(local[:,1]<=upper[1])&
                (local[:,2]>=lower[2]-.005)&(local[:,2]<=upper[2]+.005))
        points=local[inside];counts[k]=len(points)
        if len(points)<empty_threshold:continue
        left=points[np.argmin(points[:,1])];right=points[np.argmax(points[:,1])]
        widths[k]=2*(max(neutral-left[1],right[1]-neutral)+.0025)
        # Both closing-side support points must fit inside the 15mm pad with
        # a 3mm end margin. Choose the smallest feasible axial correction.
        lo=max(left[2],right[2])-(upper[2]-.003)
        hi=min(left[2],right[2])-(lower[2]+.003)
        lo=max(lo,-.005);hi=min(hi,.005)
        if lo>hi:depth_bad[k]=True;continue
        advance[k]=np.clip(0.,lo,hi)
    bad=(counts<empty_threshold)|depth_bad|(widths>.088)|(widths<=0)
    result=eef.copy();result[:,:3,3]+=advance[:,None]*result[:,:3,2]
    return result,np.minimum(widths,.088),advance,counts,bad,depth_bad


class Processor:
    def __init__(self,grippers,*,use_spatial_crop=True,nms_translation_m=.015,nms_rotation_deg=15):
        if not 0<nms_translation_m<=.03 or not 0<nms_rotation_deg<=30:
            raise ValueError('NMS thresholds must be positive and at most 30mm/30deg')
        self.nms_translation_m=float(nms_translation_m)
        self.nms_rotation_deg=float(nms_rotation_deg)
        self.grippers=grippers
        if use_spatial_crop:
            from .spatial_crop import detect
            self.native_detector=detect
        else:self.native_detector=native_detect

    def process(self,poses,scores,observed,target,table,*,variants=VARIANTS):
        if not variants or len(set(variants))!=len(variants) or set(variants)-set(VARIANTS):
            raise ValueError('Unknown or repeated postprocessing variant')
        results={};native_modes=[m for m in ['native','native_rechecked'] if m in variants]
        if native_modes:
            start=time.perf_counter()
            detected=self.native_detector(poses,np.concatenate([target,observed]))
            native_time=time.perf_counter()-start
        for mode in native_modes:
            began=time.perf_counter();out=poses.copy()
            shift=detected['depths']-.04
            out[:,:3,3]+=shift[:,None]*out[:,:3,2]
            widths=detected['widths']
            bad=detected['bad'] if mode=='native' else detected['final_bad']
            # All outputs must respect the real 88mm opening limit.
            stroke_bad=(widths>.088+1e-12)|(widths<=0);bad=bad|stroke_bad
            ids,nms=hard_nms(out,scores,widths,~bad,self.nms_translation_m,self.nms_rotation_deg)
            results[mode]={'poses':out,'widths':widths,'selected':ids,'before_nms':np.flatnonzero(~bad).tolist(),
                'bad':bad,'empty':detected['counts']<10,'cavity_counts':detected['counts'],
                'axial_shift_m':shift,'stroke_bad':stroke_bad,'NMS':nms,
                'nms_parameters':{'translation_m':self.nms_translation_m,'rotation_deg':self.nms_rotation_deg},
                'seconds':native_time+time.perf_counter()-began}
        if 'arx_observed' not in variants:return results
        began=time.perf_counter();eef=poses.copy();eef[:,:3,3]-=TCP_M*eef[:,:3,2]
        eef,widths,shift,counts,bad,depth_bad=pad_refinement(eef,target,self.grippers['right'].cavity)
        arms={}
        for side,model in self.grippers.items():
            candidates=np.flatnonzero(~bad)
            tb=np.zeros(len(eef),bool);pb=np.zeros(len(eef),bool)
            tb[candidates]=table_bad(model,eef[candidates],widths[candidates],table)
            active=np.flatnonzero(~bad&~tb)
            pb[active]=variable_collision(model,eef[active],widths[active],observed)
            arms[side]={'table_bad':tb,'point_bad':pb,'clear':~(bad|tb|pb)}
        keep=arms['right']['clear']|arms['left']['clear']
        out=eef.copy();out[:,:3,3]+=TCP_M*out[:,:3,2]
        ids,nms=hard_nms(out,scores,widths,keep,.015,15)
        results['arx_observed']={'poses':out,'widths':widths,'selected':ids,'before_nms':np.flatnonzero(keep).tolist(),
            'bad':~keep,'empty':counts<10,'cavity_counts':counts,'axial_shift_m':shift,
            'depth_bad':depth_bad,'arms':arms,'NMS':nms,'seconds':time.perf_counter()-began}
        return results


def make_pool(payload,result,source_order,variant):
    candidates=[]
    for index in result['selected']:
        original=payload['candidates'][index];candidate=deepcopy(original)
        pose=np.asarray(result['poses'][index]);width=float(result['widths'][index])
        candidate.update(pose=pose,width=width,point1=pose[:3,3]-width/2*pose[:3,1],point2=pose[:3,3]+width/2*pose[:3,1])
        candidate['metadata'].update(postprocessing=variant,original_native_rank=int(index),
            source_generator_index=int(source_order[index]),original_contact_pose_world=original['pose'],
            refined_contact_pose_world=pose,original_width_m=float(original['width']),
            pose_refinement_applied=bool(abs(result['axial_shift_m'][index])>1e-12),
            axial_refinement_m=float(result['axial_shift_m'][index]),width_refinement_applied=abs(width-original['width'])>1e-12)
        candidates.append(candidate)
    return {'candidates':candidates,'raw_count':len(source_order),'retained_count':len(candidates),
        'protocol':'arx_native_postprocessing_v1','variant':variant,'native_score_order':True,
        'raw_fallback':False,'hard_deletion':True,'refinement_receipt_required':True,
        'nms_parameters':result.get('nms_parameters')}
