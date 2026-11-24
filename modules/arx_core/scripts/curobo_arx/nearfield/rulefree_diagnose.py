"""Revised advisor35: raw six-dimensional VLM targets, CPU records only.

No axial synthesis, corridor decision, action clipping or implicit GRASP.
Geometric references are for offline error measurement, never action generation.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .advisor35_diagnose import OLD,TRAIN,Evidence,dump,csv_file,requests,vector
from .advisor32_rules import parse_prediction,goal_from_delta,local_admission

PAD=np.array([-.006,0.,.0375])


def raw_action(current,prediction):
    parts=parse_prediction('D',prediction)
    if parts is None:
        return {'kind':'model_GRASP','goal_world':None,'source':'explicit_model_token'}
    dp,dr=parts
    return {'kind':'model_REFINE','goal_world':goal_from_delta(current,dp,dr).tolist(),
            'translation_m':dp.tolist(),'rotation_rad':dr.tolist(),'source':'all_six_model_values_unchanged'}


def group_z(rows, by_id):
    motion=[r for r in rows if vector(r['output']) is not None]
    if not motion:return {'motion_rows':0}
    a=np.array([vector(r['output']) for r in motion]);w=np.array([r['sample_weight'] for r in motion]);z=a[:,2]
    def group(mask):
        return {'count':int(mask.sum()),'fraction':float(mask.mean()),
                'sample_weight_fraction':float(w[mask].sum()/w.sum())}
    return {'motion_rows':len(motion),'all_rows':len(rows),'GRASP_rows':len(rows)-len(motion),
        'z_mm_percentiles':np.percentile(z,[0,10,25,50,75,90,100]).tolist(),
        'z_mean_mm':float(z.mean()),'z_std_mm':float(z.std()),
        'z_weighted_mean_mm':float(np.average(z,weights=w)),
        'abs_z_le5':group(abs(z)<=5),'z_negative':group(z<0),
        'abs_z_5_to30':group((abs(z)>5)&(abs(z)<=30)),
        'abs_z_30_to40':group((abs(z)>30)&(abs(z)<40)),
        'z_40_to75':group((z>=40)&(z<=75)),
        'z_65_to75':group((z>=65)&(z<=75)),
        'z_gt75':group(z>75),
        'lateral_gt10':group(np.linalg.norm(a[:,:2],axis=1)>10)}


def z_distribution(ev,out):
    samples=ev.lines(TRAIN/'dataset/paired_samples.jsonl');byid={r['id']:r for r in samples}
    rows=ev.lines(TRAIN/'model_rows/D/train.jsonl');assert len(rows)==435
    groups={}
    for obj in sorted({byid[r['sample_id']]['object_id'] for r in rows}):
        groups['object:'+obj]=[r for r in rows if byid[r['sample_id']]['object_id']==obj]
    for state in sorted({r['instruction'].split('Local state: ')[-1] for r in rows}):
        groups['state:'+state]=[r for r in rows if r['instruction'].split('Local state: ')[-1]==state]
    groups['all']=rows
    groups['large_lateral_gt10mm']=[r for r in rows if vector(r['output']) is not None
                                  and np.linalg.norm(vector(r['output'])[:2])>10]
    result={k:group_z(v,byid) for k,v in groups.items()}
    dump(out/'axial_label_distribution.json',result)
    csv_file(out/'axial_label_distribution.csv',[{'group':k,**v} for k,v in result.items()])
    return result


def intersections(mesh, origin, direction):
    """First surface intersection using original triangles, no primitive proxy."""
    tri=np.asarray(mesh.triangles);o=np.asarray(origin);d=np.asarray(direction)
    e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
    h=np.cross(np.broadcast_to(d,e2.shape),e2);det=np.einsum('ij,ij->i',e1,h)
    valid=abs(det)>1e-12;inv=np.zeros_like(det);inv[valid]=1/det[valid]
    s=o-tri[:,0];u=inv*np.einsum('ij,ij->i',s,h);q=np.cross(s,e1)
    v=inv*(q@d);t=inv*np.einsum('ij,ij->i',e2,q)
    valid &= (u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)&(t>=0)
    inds=np.flatnonzero(valid)
    if not len(inds):return None
    i=inds[np.argmin(t[inds])];normal=np.cross(e1[i],e2[i]);normal/=np.linalg.norm(normal)
    return {'point':o+t[i]*d,'normal':normal,'face':int(i)}


def canonical_reference(mesh,G_object,registered_spam=False):
    """One frozen geometric proposal per evaluated object; no physical credit."""
    import trimesh
    g=np.asarray(G_object);center=g[:3,3]+g[:3,:3]@PAD
    to_box,extents=trimesh.bounds.oriented_bounds(mesh,angle_digits=1);box=np.linalg.inv(to_box)
    closing=g[:3,1].copy()
    if not registered_spam:
        # Keep the nearest OBB normal, not automatically the shortest extent.
        axis=int(np.argmax(abs(box[:3,:3].T@closing)))
        y=box[:3,axis].copy();y*=1 if y@closing>=0 else -1
        close_index=axis;local=(to_box@np.r_[center,1])[:3];local[axis]=0
        center=(box@np.r_[local,1])[:3];closing=y
    else:
        close_index=int(np.argmax(abs(box[:3,:3].T@closing)))
    half=max(.5,float(extents.max())*2)
    hits=[intersections(mesh,center+sign*half*closing,-sign*closing) for sign in (-1,1)]
    valid=all(h is not None for h in hits)
    normals=None
    if valid:
        normals=[float(np.rad2deg(np.arccos(np.clip(abs(h['normal']@closing),0,1)))) for h in hits]
        # Use actual surface midpoint while retaining the declared axis and height.
        if not registered_spam:center=(hits[0]['point']+hits[1]['point'])/2
    approach=g[:3,2]-closing*(closing@g[:3,2]);approach/=np.linalg.norm(approach)
    across=np.cross(closing,approach);rotation=np.column_stack([across,closing,approach])
    goal=np.eye(4);goal[:3,:3]=rotation;goal[:3,3]=center-rotation@PAD
    if registered_spam:goal=g.copy()
    return {'Gcanonical_object':goal.tolist(),'registered_canonical_reused':registered_spam,
        'reference_type':'registered_Spam_canonical' if registered_spam else 'new_OBB_surface_midpoint_reference',
        'OBB_extents_mm':(extents*1000).tolist(),'OBB_axis':close_index,
        'two_surface_hits':valid,'contact_normal_to_closing_degrees':normals,
        'normal_alignment_within_5deg':bool(valid and max(normals)<=5),
        'contact_width_mm':float(np.linalg.norm(hits[1]['point']-hits[0]['point'])*1000) if valid else None,
        'hits':[{'point_object':h['point'].tolist(),'normal_object':h['normal'].tolist(),'face':h['face']} if h else None for h in hits],
        'reference_rotation_change_degrees':float(Rotation.from_matrix(g[:3,:3].T@goal[:3,:3]).magnitude()*180/np.pi),
        'reference_position_change_mm':float(np.linalg.norm(g[:3,3]-goal[:3,3])*1000),
        'new_ik_collision_check':False,'new_physical_validation':False,
        'scope':'Offline geometric reference only; two intersections and normal checks do not establish force closure or a successful grasp.'}


def error(current,predicted,goal):
    e,p,g=map(np.asarray,(current,predicted,goal))
    delta=np.linalg.inv(e)@g
    pos=e[:3,:3].T@(g[:3,3]-p[:3,3])*1000
    rot=Rotation.from_matrix(p[:3,:3].T@g[:3,:3]).as_rotvec(degrees=True)
    return {'truth_translation_current_EE_mm':(delta[:3,3]*1000).tolist(),
        'truth_rotation_current_EE_deg':Rotation.from_matrix(delta[:3,:3]).as_rotvec(degrees=True).tolist(),
        'residual_translation_current_EE_mm':pos.tolist(),'residual_rotation_predicted_EE_deg':rot.tolist(),
        'position_error_mm':float(np.linalg.norm(pos)),'orientation_error_deg':float(np.linalg.norm(rot))}


def main(out):
    from scripts.curobo_arx.sdg_stage1.sampler import composed_mesh
    out=Path(out).resolve()
    if (out/'raw_output_review.json').exists():raise FileExistsError('Preserve finished pure-output review')
    ev=Evidence();ev.pin(__file__);ev.read(out/'registration.json');ev.pin(out/'advisor35.md')
    reg=ev.read(OLD/'registration.json');dist=z_distribution(ev,out)
    references={};rows=[];sources=[]
    for trial in reg['seed_trials']:
        if trial['method']!='D' or trial['vision_arm']=='g0':continue
        result_path=list(OLD.glob('groups/*/trials/'+trial['trial_id']+'/trial_result.json'))
        assert len(result_path)==1
        result=ev.read(result_path[0]);ep=Path(result['attempts'][0]['episode'])
        snap=ev.read(ep/'capture_snapshot.json');O0=np.asarray(snap['objects']['test_object']['world'])
        obj=trial['object_id']
        if obj not in references:
            asset=next(o for o in reg['catalog']['objects'] if o['object_id']==obj)
            mesh,proof=composed_mesh(asset['usd_path'])
            for p,h in proof['used_layers'].items():ev.pin(p,h)
            spam=obj=='household_ycb_spam_can'
            source=np.asarray(trial['G0_world'] if spam else trial['source_Gstar_world'])
            references[obj]={'object_id':obj,'source_trial_id':trial['trial_id'],
                'mesh_evidence':proof,**canonical_reference(mesh,np.linalg.inv(O0)@source,spam)}
        for path,req,rep in requests(ev,ep):
            if not rep['model_inference']:continue
            row=ev.read(req['observation']);current=np.asarray(req['inputs']['eef_world']);O=np.asarray(row['snapshot']['objects']['test_object']['world'])
            action=raw_action(current,rep['prediction']);assert action['kind']=='model_REFINE'
            predicted=np.asarray(action['goal_world'])
            historical=O@np.linalg.inv(O0)@np.asarray(trial['source_Gstar_world'])
            canon=O@np.asarray(references[obj]['Gcanonical_object'])
            values=vector(rep['prediction'])
            rows.append({'trial_id':trial['trial_id'],'object_id':obj,'vision_arm':trial['vision_arm'],
                'request_id':req['request_id'],'request_file':str(path),'raw_index':req['raw_index'],
                'motions_completed':req['motions_completed'],'prediction':rep['prediction'],
                'raw_translation_mm':values[:3].tolist(),'raw_rotation_deg':values[3:].tolist(),
                'unmodified_target_world':predicted.tolist(),
                'historical_reference_error':error(current,predicted,historical),
                'canonical_reference_error':error(current,predicted,canon),
                'canonical_quality':references[obj]['normal_alignment_within_5deg'],
                'canonical_reference_type':references[obj]['reference_type'],
                'old_65mm_25deg_admission_for_reporting_only':local_admission(current,predicted),
                'original_controller_action_kind':rep['action']['kind'],
                'recorded_observation_source':'Historical mixed-controller rollout, not the rollout of this unmodified output',
                'counterfactual_execution':False})
    assert len(rows)==36
    summaries=[]
    for arm in ('fresh','full_stale'):
        for obj in sorted(references):
            for phase in ('entry','intermediate'):
                sub=[r for r in rows if r['vision_arm']==arm and r['object_id']==obj
                     and (r['motions_completed']==0)==(phase=='entry')]
                if not sub:continue
                item={'object_id':obj,'vision_arm':arm,'phase':phase,'N':len(sub),
                    'predicted_z_median_mm':float(np.median([r['raw_translation_mm'][2] for r in sub]))}
                for reference in ('historical','canonical'):
                    errors=[r[reference+'_reference_error'] for r in sub]
                    item[reference+'_target_z_median_mm']=float(np.median([v['truth_translation_current_EE_mm'][2] for v in errors]))
                    item[reference+'_position_error_median_mm']=float(np.median([v['position_error_mm'] for v in errors]))
                    item[reference+'_angle_error_median_deg']=float(np.median([v['orientation_error_deg'] for v in errors]))
                summaries.append(item)
    dump(out/'canonical_references.json',list(references.values()))
    dump(out/'raw_output_rows.json',rows);csv_file(out/'raw_output_rows.csv',rows)
    dump(out/'raw_output_review.json',{'model_calls_reinterpreted':36,'fresh_calls':sum(r['vision_arm']=='fresh' for r in rows),
        'full_stale_calls':sum(r['vision_arm']=='full_stale' for r in rows),'explicit_model_GRASP':0,
        'new_physical_trials':0,'new_neural_inferences':0,'new_training_updates':0,
        'raw_six_values_unchanged':True,'summaries':summaries,'evidence_files':ev.files,
        'causal_scope':'Per-observation kinematic counterfactual only. Subsequent frames came from the old mixed controller; cannot splice them into a pure-model rollout.',
        'canonical_scope':'Spam registered reference; other objects get new, explicitly qualified OBB/surface-midpoint references, without new IK/collision/hold validation.'})
    print({'rows':len(rows),'references':len(references),'physical_trials':0})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args().out)
