"""Deterministically merge same-link balls while covering the original union."""
from copy import deepcopy
import hashlib
import json

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def allocate(counts,total):
    names=list(counts)
    if not len(names)<=total<=sum(counts.values()):raise ValueError('Invalid sphere-count budget')
    exact=np.asarray([counts[n] for n in names],dtype=float)*total/sum(counts.values())
    targets=np.maximum(1,np.floor(exact).astype(int))
    while targets.sum()<total:
        scores=exact-targets;scores[targets>=np.asarray(list(counts.values()))]=-np.inf
        targets[int(np.argmax(scores))]+=1
    while targets.sum()>total:
        scores=targets-exact;scores[targets<=1]=-np.inf
        targets[int(np.argmax(scores))]-=1
    return dict(zip(names,map(int,targets)))


def enclose(a,b):
    delta=b[:3]-a[:3];distance=float(np.linalg.norm(delta))
    if a[3]>=distance+b[3]:return a.copy()
    if b[3]>=distance+a[3]:return b.copy()
    radius=(distance+a[3]+b[3])/2
    return np.r_[a[:3]+delta*((radius-a[3])/distance),radius]


def merge_link(values,target):
    original=np.asarray([[*s['center'],s['radius']] for s in values],dtype=float)
    if original.ndim!=2 or original.shape[1]!=4 or not np.isfinite(original).all() or (original[:,3]<=0).any():
        raise ValueError('Invalid source sphere geometry')
    balls=original.copy();members=[[i] for i in range(len(balls))]
    while len(balls)>target:
        distances=np.linalg.norm(balls[:,None,:3]-balls[None,:,:3],axis=-1)
        radii=np.maximum(np.maximum(balls[:,None,3],balls[None,:,3]),
                         (distances+balls[:,None,3]+balls[None,:,3])/2)
        costs=radii**3-np.maximum(balls[:,None,3],balls[None,:,3])**3
        costs[np.tril_indices(len(balls))]=np.inf
        i,j=np.unravel_index(np.argmin(costs),costs.shape)
        balls[i]=enclose(balls[i],balls[j]);members[i]+=members[j]
        balls=np.delete(balls,j,axis=0);members.pop(j)
    result=[];margins=[]
    for ball,indices in zip(balls,members):
        center=ball[:3].astype(np.float32).astype(float)
        required=np.max(np.linalg.norm(original[indices,:3]-center,axis=-1)+original[indices,3])
        radius=float(np.nextafter(np.float32(required+1e-6),np.float32(np.inf)))
        margins.extend((radius-np.linalg.norm(original[indices,:3]-center,axis=-1)-original[indices,3]).tolist())
        result.append({'center':center.tolist(),'radius':radius})
    return result,{'original_count':len(original),'merged_count':len(result),
        'minimum_original_ball_cover_margin_m':float(min(margins)),
        'maximum_radius_m':max(v['radius'] for v in result),'members':members}


def create_profile(source,total):
    source=deepcopy(source);spheres=source['collision_spheres'];counts={n:len(v) for n,v in spheres.items()}
    if total==sum(counts.values()):
        result=deepcopy(spheres)
        proof={n:{'original_count':len(v),'merged_count':len(v),
            'minimum_original_ball_cover_margin_m':0.,'maximum_radius_m':max(s['radius'] for s in v),
            'members':[[i] for i in range(len(v))]} for n,v in spheres.items()}
    else:
        targets=allocate(counts,total);result={};proof={}
        for name,values in spheres.items():result[name],proof[name]=merge_link(values,targets[name])
    if sum(map(len,result.values()))!=total:raise RuntimeError('Sphere allocation mismatch')
    return {'profile_id':f'diagnostic_conservative_{total}_from_871','total_spheres':total,
        'source_parameter_sha256':source['sphere_parameter_sha256'],'sphere_parameter_sha256':digest(result),
        'collision_spheres':result,'coverage':proof,'planning_diagnostic_only':True,
        'same_link_conservative_enclosure':True,'runtime_default_activated':False}
