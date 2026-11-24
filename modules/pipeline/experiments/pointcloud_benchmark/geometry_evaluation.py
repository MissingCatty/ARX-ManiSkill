"""Common surface metrics; observation-only placement and oracle shape scores."""
import itertools
import time
import numpy as np
from scipy.spatial import cKDTree


def subsample(points,count,seed=12345):
    p=np.asarray(points,dtype=np.float64)
    if len(p)<=count:return p
    return p[np.random.default_rng(seed).choice(len(p),count,replace=False)]


def cube_rotations():
    result=[]
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product([-1,1],repeat=3):
            r=np.eye(3)[list(permutation)]*np.array(signs)[:,None]
            if np.linalg.det(r)>.5:result.append(r)
    return result


def fit_similarity(source,target,limits):
    a=np.asarray(source);b=np.asarray(target)
    ma=a.mean(0);mb=b.mean(0);ac=a-ma;bc=b-mb
    covariance=ac.T@bc/len(a)
    u,s,vt=np.linalg.svd(covariance)
    correction=np.ones(3);correction[-1]=np.sign(np.linalg.det(vt.T@u.T))
    r=vt.T@np.diag(correction)@u.T
    variance=np.sum(ac*ac)/len(a)
    scale=float(np.clip(np.sum(s*correction)/max(variance,1e-15),*limits))
    translation=mb-scale*r@ma
    return scale,r,translation


def calibrate_from_observation(predicted,observed,bidirectional=False,normalized_shape=False):
    """Never accepts asset poses or GT; Sim(3) uses observed depth alone.

    Oracle shape evaluation calls the same solver separately with normalized GT
    and bidirectional correspondences. Its result never enters metric outputs.
    """
    started=time.perf_counter()
    source=subsample(predicted,4096);target=subsample(observed,2048)
    pcenter=(np.quantile(source,.01,axis=0)+np.quantile(source,.99,axis=0))/2
    ocenter=(np.quantile(target,.01,axis=0)+np.quantile(target,.99,axis=0))/2
    pd=np.max(np.quantile(source,.99,axis=0)-np.quantile(source,.01,axis=0))
    od=np.max(np.quantile(target,.99,axis=0)-np.quantile(target,.01,axis=0))
    initial=1. if normalized_shape else od/max(pd,1e-12)
    limits=(1.,1.) if normalized_shape else (.5*initial,2.*initial)
    best=None;target_tree=cKDTree(target)
    for initial_rotation in cube_rotations():
        scale=initial;r=initial_rotation;t=ocenter-scale*r@pcenter
        for _ in range(15):
            transformed=source@r.T*scale+t
            distances,indices=cKDTree(transformed).query(target,workers=1)
            keep=np.argsort(distances)[:max(12,int(.85*len(target)))]
            a=source[indices[keep]];b=target[keep]
            if bidirectional:
                ds,js=target_tree.query(transformed,workers=1)
                ks=np.argsort(ds)[:max(12,int(.85*len(source)))]
                a=np.concatenate([a,source[ks]]);b=np.concatenate([b,target[js[ks]]])
            scale,r,t=fit_similarity(a,b,limits)
        transformed=source@r.T*scale+t
        d=cKDTree(transformed).query(target,workers=1)[0]
        score=float(np.mean(np.sort(d*d)[:max(12,int(.85*len(d)))]))
        if bidirectional:
            d2=target_tree.query(transformed,workers=1)[0]
            score+=float(np.mean(np.sort(d2*d2)[:max(12,int(.85*len(d2)))]))
        if best is None or score<best[0]:best=(score,scale,r,t)
    score,scale,r,t=best
    transformed=np.asarray(predicted)@r.T*scale+t
    return transformed,{'scale':scale,'rotation':r.tolist(),'translation':t.tolist(),
                        'trimmed_observed_fit_rmse_m':float(np.sqrt(score)),
                        'source':'evaluation_only_GT' if bidirectional else 'observed_RGBD_and_shared_predicted_mask_only',
                        'seconds':time.perf_counter()-started,'rotation_starts':24,'iterations':15,
                        'scale_bounds':list(limits),'correspondence_keep_fraction':.85}


def distances(predicted,reference):
    a=cKDTree(reference).query(predicted,workers=1)[0]
    b=cKDTree(predicted).query(reference,workers=1)[0]
    return a,b


def surface_metrics(predicted,reference,visible_reference):
    forward,reverse=distances(predicted,reference)
    visible=cKDTree(predicted).query(visible_reference,workers=1)[0]
    result={'chamfer_l1_mm':float((forward.mean()+reverse.mean())*.5*1000),
            'prediction_to_surface_mean_mm':float(forward.mean()*1000),
            'prediction_to_surface_p50_mm':float(np.median(forward)*1000),
            'prediction_to_surface_p95_mm':float(np.percentile(forward,95)*1000),
            'full_surface_to_prediction_mean_mm':float(reverse.mean()*1000),
            'visible_surface_to_prediction_mean_mm':float(visible.mean()*1000),
            'visible_surface_to_prediction_p95_mm':float(np.percentile(visible,95)*1000)}
    for threshold_mm in [2,5,10]:
        threshold=threshold_mm/1000.
        precision=float(np.mean(forward<=threshold));recall=float(np.mean(reverse<=threshold))
        result[f'precision_{threshold_mm}mm']=precision
        result[f'coverage_{threshold_mm}mm']=recall
        result[f'fscore_{threshold_mm}mm']=2*precision*recall/(precision+recall) if precision+recall else 0.
    return result


def oracle_shape_metrics(predicted,reference):
    predicted=subsample(predicted,20000);reference=subsample(reference,20000)
    pc=(predicted.min(0)+predicted.max(0))/2;ps=max(np.ptp(predicted,axis=0).max(),1e-12)
    rc=(reference.min(0)+reference.max(0))/2;rs=max(np.ptp(reference,axis=0).max(),1e-12)
    pn=(predicted-pc)/ps;rn=(reference-rc)/rs
    aligned,registration=calibrate_from_observation(pn,rn,bidirectional=True,normalized_shape=True)
    a,b=distances(aligned,rn)
    cd=float((a.mean()+b.mean())*.5)
    return {'chamfer_percent_object_extent':cd*100,'equivalent_chamfer_mm':cd*rs*1000,
            'uses_GT_for_evaluation_alignment':True,'registration':registration}


def unproject(depth,mask,k):
    y,x=np.nonzero(mask&np.isfinite(depth)&(depth>0))
    z=depth[y,x]
    return np.stack([(x-k[0,2])/k[0,0]*z,(y-k[1,2])/k[1,1]*z,z],axis=-1)
