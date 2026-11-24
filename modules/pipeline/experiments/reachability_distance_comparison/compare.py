"""Compare frozen distance experiments with the independent front FK atlas."""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
WORKSPACE = PROJECT.parent
MS = WORKSPACE / 'maniskill'
ATLAS = WORKSPACE / 'reachability/产物/前方精细_20261121_01'
PRIMARY = PROJECT / 'experiments/full_failure_distance_probe/runs/all53_20261123_01'
PILOT = PROJECT / 'experiments/edge_distance_probe/runs/three_objects_20261123_01'
ROOT = HERE / 'runs/map_distance_20261123_01'
SAMPLE_SIZE = 64
NEIGHBORS = 8192
POSITION_SCALE = .02
ANGLE_SCALE = .35


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024**2): h.update(b)
    return h.hexdigest()


def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    def encode(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    t=p.with_suffix(p.suffix+'.pending');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=encode)+'\n');t.replace(p)


def voxel_id(p,resolution):
    c=np.floor(np.asarray(p,dtype=np.float64)/resolution).astype(np.int64)+2048
    return (c[:,0]*4096+c[:,1])*4096+c[:,2]


def lookup(table,points,resolution):
    ids=voxel_id(points,resolution);i=np.searchsorted(table['voxel_id'],ids)
    found=(i<len(table['voxel_id']));i=np.minimum(i,len(table['voxel_id'])-1)
    found &= table['voxel_id'][i]==ids
    result={'covered':found}
    for key in ['sample_count','orientation_count','condition_mean','margin_mean_rad']:
        result[key]=np.where(found,table[key][i],0.)
    return result


def prepare():
    assert not ROOT.exists()
    ROOT.mkdir(parents=True)
    assert read(PRIMARY/'final_verification.json')['passed']
    assert read(PILOT/'final_verification.json')['passed']
    atlas_done=read(ATLAS/'完成.json')
    assert atlas_done['passed']
    inputs={str(ATLAS/n):h for n,h in atlas_done['file_sha256'].items()}
    for root in [PRIMARY,PILOT]:
        for n in ['registration.json','summary.json','final_verification.json']:inputs[str(root/n)]=sha(root/n)
    reg=read(PRIMARY/'registration.json')
    for source in reg['sources']:
        d=Path(source['source_trial'])
        for n in ['state.json','candidates/candidate_pool.json']:inputs[str(d/n)]=sha(d/n)
    for root in [PRIMARY,PILOT]:
        for row in read(root/'summary.json')['cases']:
            d=root/'cases'/row['id']
            for n in ['state.json','endpoint_ik.npz','result.json']:
                if (d/n).exists():inputs[str(d/n)]=sha(d/n)
    urdf=WORKSPACE/'arx_core/scripts/curobo_arx/static_pick/reference/model/arx.urdf'
    inputs[str(urdf)]=sha(urdf)
    save(ROOT/'registration.json',{'created_at':datetime.now().astimezone().isoformat(),
        'atlas':str(ATLAS),'primary':str(PRIMARY),'pilot':str(PILOT),
        'primary_source_scenes':53,'primary_distance_positions':318,'pilot_positions':18,
        'primary_pilot_pooled':False,'query_pose':'actual EEF @ frozen atlas T_eef_action, transformed by each recorded body4 inverse',
        'position_metrics':'all candidate poses: original 5mm cells, original 5cm cells, observed orientation coverage',
        'continuous_pose_sample':'64 evenly spaced native-score ranks per original scene, identical indices at all six distances',
        'continuous_pose_spatial_neighbors':NEIGHBORS,'position_scale_m':POSITION_SCALE,'orientation_scale_rad':ANGLE_SCALE,
        'pose_affinity':'exp(-min(translation/0.02 + orientation/0.35)); score for similarity, not a calibrated probability',
        'pregrasp_lift':'8cm along negative local Z and 3cm world up, matching V1',
        'primary_statistics':['Spearman with pure IK fraction','Spearman with collision-free endpoint fraction',
            'AUC for any complete initial plan','within-source paired changes'],
        'bootstrap_unit':'source scene, preserving all 6 distances','bootstrap_repetitions':500,
        'physical_attempts':0,'new_IK_queries':0,'atlas_witnesses_added':0,
        'input_sha256':inputs,'source_sha256':{str(Path(__file__)):sha(__file__)}})
    print('REGISTERED',len(inputs),'inputs; no new IK or robot execution',flush=True)


def load_cases():
    meta=read(ATLAS/'地图元数据.json');tcp=np.asarray(meta['T_eef_action'])
    primary=read(PRIMARY/'summary.json')['cases'];pilot=read(PILOT/'summary.json')['cases']
    cache={};rows=[]
    for cohort,root,records in [('all53',PRIMARY,primary),('pilot3',PILOT,pilot)]:
        for r in records:
            out=root/'cases'/r['id'];state=read(out/'state.json')
            source=Path(r['source_trial']) if cohort=='all53' else Path(read(out/'entry.json')['source_trial'])
            if str(source) not in cache:
                pool=read(source/'candidates/candidate_pool.json')['candidates']
                cache[str(source)]=np.array([c['pose'] for c in pool])
            original=cache[str(source)];contacts=original.copy();contacts[:,:3,3]+=r['translation_m']
            eef=contacts.copy();eef[:,:3,3]-=.043*eef[:,:3,2]
            action=eef@tcp
            assert np.max(np.abs(action[:,:3,3]-contacts[:,:3,3]))<1e-12
            body=np.linalg.inv(np.asarray(state['links']['body4']))
            poses=body@action
            a=np.load(out/'endpoint_ik.npz',allow_pickle=False)
            solved=a['solved'] if 'solved' in a else a['solver_success']
            ik=solved.any(-1);clear=a['scene_clear'].any(-1)
            assert len(poses)==len(ik)==len(clear)
            sample=np.unique(np.linspace(0,len(poses)-1,min(SAMPLE_SIZE,len(poses)),dtype=int))
            limits=np.array([state['joint_limits'][state['joint_names'].index(n)] for n in meta['joint_names']])
            assert np.max(np.abs(limits-np.asarray(meta['joint_limits'])))<1e-5
            rows.append({'cohort':cohort,'id':r['id'],'source_id':r['source_id'],'level':r.get('level',int(r['id'][-1])),
                'directory':out,'record':r,'state':state,'poses':poses,'ik':ik,'clear':clear,'sample':sample,
                'positions':poses[:,:3,3], 'tcp_mean':poses[:,:3,3].mean(0),
                'object_center':(body@np.r_[state['object_center'],1.])[:3],
                'body4_world':np.asarray(state['links']['body4'])})
    return meta,rows


def verify_alignment(meta,rows,bank):
    sys.path[:0]=[str(MS),str(MS/'arx_pipeline')]
    from scene_tools import ArmFK
    fk=ArmFK();tcp=np.asarray(meta['T_eef_action'])
    rng=np.random.default_rng(20261123);pe=[];re=[]
    for i in rng.choice(len(bank),512,replace=False):
        b=np.asarray(bank[i]);t=fk.fk(b[:6])@tcp
        r=Rotation.from_quat(b[9:13][[1,2,3,0]]).as_matrix()
        pe.append(np.linalg.norm(t[:3,3]-b[6:9]));re.append(Rotation.from_matrix(t[:3,:3]@r.T).magnitude())
    assert max(pe)<2e-5 and max(re)<2e-4
    trial_errors=[]
    for row in rows:
        r=row['record'];e=r['endpoint']
        choices=r.get('witnesses',[]) or e.get('witnesses',[]) or e.get('collision_witnesses',[])
        if not choices:continue
        w=choices[0];goal=row['poses'][w['candidate']];actual=fk.fk(w['q'])@tcp
        ep=float(np.linalg.norm(actual[:3,3]-goal[:3,3]));er=float(Rotation.from_matrix(actual[:3,:3]@goal[:3,:3].T).magnitude())
        assert ep<.00202 and er<.03002,(row['id'],ep,er)
        trial_errors.append({'cohort':row['cohort'],'id':row['id'],'position_error_m':ep,'rotation_error_rad':er})
    result={'passed':True,'atlas_fk_samples':512,'maximum_map_position_error_m':float(max(pe)),
        'maximum_map_rotation_error_rad':float(max(re)),'trial_pose_checks':len(trial_errors),
        'maximum_trial_position_error_m':max(r['position_error_m'] for r in trial_errors),
        'maximum_trial_rotation_error_rad':max(r['rotation_error_rad'] for r in trial_errors),
        'joint_limits_matched':True,'actual_tcp43_position_matched':True,
        'atlas_rotation_relative_to_stored_contact':tcp[:3,:3],
        'per_trial_body4_transform_used':True,'reference_body4_world_z_m':meta['reference_base_world'][2][3],
        'current_body4_world_z_range_m':[min(r['body4_world'][2,3] for r in rows),max(r['body4_world'][2,3] for r in rows)],
        'physical_attempts':0,'new_IK_queries':0}
    save(ROOT/'alignment_verification.json',result)
    return result


def pose_queries(tree,quats,positions,rotations):
    affinity=[];position_error=[];angle_error=[]
    target=Rotation.from_matrix(rotations).as_quat()[:,[3,0,1,2]].astype(np.float32)
    for start in range(0,len(positions),64):
        points=positions[start:start+64];q=target[start:start+64]
        distance,index=tree.query(points,k=min(NEIGHBORS,len(quats)),workers=4)
        neighbors=quats[index]
        dot=np.abs(np.einsum('bki,bi->bk',neighbors,q)).clip(0,1)
        angles=2*np.arccos(dot.astype(np.float64))
        score=distance/POSITION_SCALE+angles/ANGLE_SCALE
        best=score.argmin(-1);b=np.arange(len(best))
        affinity.extend(np.exp(-score[b,best]));position_error.extend(distance[b,best]);angle_error.extend(angles[b,best])
    return np.asarray(affinity),np.asarray(position_error),np.asarray(angle_error)


def auc(score,label):
    score=np.asarray(score);label=np.asarray(label,dtype=bool);n=int(label.sum());m=len(label)-n
    if not n or not m:return None
    return float((rankdata(score)[label].sum()-n*(n+1)/2)/(n*m))


def correlation(score,target):
    return float(spearmanr(score,target).statistic)


def summarize(rows,candidate_data):
    summaries={}
    metrics=['fine_density_mean','fine_coverage_fraction','coarse_density_mean','coarse_orientation_fraction_mean',
        'pose_g_affinity_mean','pose_hgl_best_affinity','pose_hgl_mean_affinity']
    for cohort in ['all53','pilot3']:
        values=[r for r in rows if r['cohort']==cohort]
        groups=defaultdict(list)
        for i,r in enumerate(values):groups[r['source_id']].append(i)
        scores={key:np.array([r[key] for r in values]) for key in metrics}
        ik=np.array([r['ik_fraction'] for r in values]);clear=np.array([r['clear_fraction'] for r in values]);plan=np.array([r['planning_success'] for r in values])
        correlations={}
        for key,x in scores.items():
            correlations[key]={'spearman_pure_ik':correlation(x,ik),'spearman_scene_clear':correlation(x,clear),'plan_auc':auc(x,plan)}
        if cohort=='all53':
            rng=np.random.default_rng(20261123);group_ids=list(groups);cluster_indices=[np.array(groups[g]) for g in group_ids]
            for key in ['coarse_density_mean','pose_g_affinity_mean','pose_hgl_best_affinity']:
                x=scores[key];rr=[];aa=[]
                for _ in range(500):
                    idx=np.concatenate([cluster_indices[i] for i in rng.integers(0,len(cluster_indices),len(cluster_indices))])
                    rr.append(correlation(x[idx],ik[idx]));aa.append(auc(x[idx],plan[idx]))
                correlations[key]['scene_bootstrap_95ci_spearman_pure_ik']=np.quantile(rr,[.025,.975])
                correlations[key]['scene_bootstrap_95ci_plan_auc']=np.quantile([a for a in aa if a is not None],[.025,.975])
        paired=[]
        for source,idx in groups.items():
            series=sorted((values[i] for i in idx),key=lambda r:r['level']);a,b=series[0],series[-1]
            paired.append({'source_id':source,'coarse_density_increased':b['coarse_density_mean']>a['coarse_density_mean'],
                'pose_g_affinity_increased':b['pose_g_affinity_mean']>a['pose_g_affinity_mean'],
                'pure_ik_fraction_increased':b['ik_fraction']>a['ik_fraction'],
                'scene_clear_fraction_increased':b['clear_fraction']>a['clear_fraction'],
                'planning_recovered':not a['planning_success'] and any(r['planning_success'] for r in series),
                'still_unplanned':not any(r['planning_success'] for r in series)})
        quantiles={}
        for key in ['coarse_density_mean','pose_g_affinity_mean','pose_hgl_best_affinity']:
            x=scores[key];cuts=np.quantile(x,[.25,.5,.75]);bins=np.searchsorted(cuts,x,side='right')
            quantiles[key]={'cutpoints':cuts,'groups':[{'bin':i,'positions':int((bins==i).sum()),
                'mean_score':float(x[bins==i].mean()),'mean_pure_ik_fraction':float(ik[bins==i].mean()),
                'mean_scene_clear_fraction':float(clear[bins==i].mean()),'planning_passes':int(plan[bins==i].sum()),
                'planning_rate':float(plan[bins==i].mean())} for i in range(4)]}
        by_level=[]
        for level in range(6):
            subset=[r for r in values if r['level']==level]
            by_level.append({'level':level,'positions':len(subset),'planning_passes':sum(r['planning_success'] for r in subset),
                'mean_ik_fraction':np.mean([r['ik_fraction'] for r in subset]),
                'mean_clear_fraction':np.mean([r['clear_fraction'] for r in subset]),
                'mean_coarse_density':np.mean([r['coarse_density_mean'] for r in subset]),
                'mean_pose_g_affinity':np.mean([r['pose_g_affinity_mean'] for r in subset])})
        summaries[cohort]={'positions':len(values),'source_scenes':len(groups),'correlations':correlations,
            'paired_counts':{key:sum(r[key] for r in paired) for key in paired[0] if key!='source_id'},
            'paired':paired,'quartiles':quantiles,'by_level':by_level}
    primary_candidates=candidate_data
    bins=np.asarray(primary_candidates['coarse_bin']);ik=np.asarray(primary_candidates['ik']);clear=np.asarray(primary_candidates['clear'])
    summaries['all53']['all_candidate_density_bins']=[{'bin':i,'candidates':int((bins==i).sum()),
        'ik_candidates':int(ik[bins==i].sum()),'ik_fraction':float(ik[bins==i].mean()),
        'clear_candidates':int(clear[bins==i].sum()),'clear_fraction':float(clear[bins==i].mean())} for i in range(4)]
    return summaries


def run():
    reg=read(ROOT/'registration.json')
    for p,h in {**reg['input_sha256'],**reg['source_sha256']}.items():assert sha(p)==h,p
    assert not (ROOT/'started.json').exists()
    save(ROOT/'started.json',{'at':datetime.now().astimezone().isoformat(),'mode':'CPU-only analysis'})
    began=time.perf_counter();meta,rows=load_cases()
    bank=np.memmap(ATLAS/meta['witness_file'],dtype=meta['dtype'],mode='r',shape=(meta['witnesses'],len(meta['fields'])))
    alignment=verify_alignment(meta,rows,bank)
    fine=dict(np.load(ATLAS/meta['fine']['file'],allow_pickle=False));coarse=dict(np.load(ATLAS/meta['coarse']['file'],allow_pickle=False))
    sorted_counts=np.sort(coarse['sample_count']);percentiles=np.searchsorted(sorted_counts,coarse['sample_count'],side='right')/len(sorted_counts)
    coarse['density_rank']=percentiles
    query_points=[];query_rotations=[];offsets=[];serial=[];candidate_data={'coarse_bin':[],'ik':[],'clear':[]}
    for row in rows:
        p=row['positions'];f=lookup(fine,p,.005);c=lookup(coarse,p,.05)
        ids=voxel_id(p,.05);index=np.searchsorted(coarse['voxel_id'],ids);index=np.minimum(index,len(coarse['voxel_id'])-1)
        ranks=np.where(c['covered'],percentiles[index],0.)
        record={'cohort':row['cohort'],'id':row['id'],'source_id':row['source_id'],'level':row['level'],
            'candidates':len(p),'fine_density_mean':float(f['sample_count'].mean()),'fine_coverage_fraction':float(f['covered'].mean()),
            'coarse_density_mean':float(c['sample_count'].mean()),'coarse_density_percentile_mean':float(ranks.mean()),
            'coarse_orientation_fraction_mean':float((c['orientation_count']/1152).mean()),
            'tcp_mean_body4':row['tcp_mean'],'object_center_body4':row['object_center'],
            'ik_fraction':float(row['ik'].mean()),'clear_fraction':float(row['clear'].mean()),
            'planning_success':bool(row['record']['planning_success']),
            'edge_gap_m':row['record'].get('target_edge_gap_m',row['record'].get('edge_gap_m')),
            'pose_sample_indices':row['sample']}
        selected=row['poses'][row['sample']];variants=[]
        for variant in ['g','h','l']:
            t=selected.copy()
            if variant=='h':t[:,:3,3]-=.08*t[:,:3,2]
            elif variant=='l':t[:,:3,3]+=.03*row['body4_world'][:3,:3].T[:,2]
            begin=len(query_points);query_points.extend(t[:,:3,3]);query_rotations.extend(t[:,:3,:3]);end=len(query_points)
            variants.append((variant,begin,end))
        offsets.append(variants);serial.append(record)
        if row['cohort']=='all53':
            candidate_data['coarse_bin'].extend(np.minimum(3,(ranks*4).astype(int)))
            candidate_data['ik'].extend(row['ik']);candidate_data['clear'].extend(row['clear'])
    save(ROOT/'progress.json',{'phase':'building_map_spatial_index','distance_positions':len(rows),'continuous_pose_queries':len(query_points)})
    positions=np.array(bank[:,6:9],dtype=np.float64);tree=cKDTree(positions,balanced_tree=True)
    quats=np.asarray(bank[:,9:13]).copy();quats/=np.linalg.norm(quats,axis=1,keepdims=True)
    queries=np.array(query_points);rotations=np.array(query_rotations)
    affinity=np.zeros(len(queries));ep=affinity.copy();er=affinity.copy()
    for start in range(0,len(queries),192):
        stop=min(start+192,len(queries));a,p,r=pose_queries(tree,quats,queries[start:stop],rotations[start:stop])
        affinity[start:stop]=a;ep[start:stop]=p;er[start:stop]=r
        if start%1920==0 or stop==len(queries):
            save(ROOT/'progress.json',{'phase':'continuous_position_orientation_queries','completed':stop,'registered':len(queries),
                'wall_seconds':time.perf_counter()-began})
            print('POSE_QUERY',stop,'/',len(queries),flush=True)
    samples=[]
    for row,out,variants in zip(rows,serial,offsets):
        q={kind:affinity[start:stop] for kind,start,stop in variants}
        joint=np.minimum.reduce([q['g'],q['h'],q['l']])
        out.update(pose_g_affinity_mean=float(q['g'].mean()),pose_g_affinity_best=float(q['g'].max()),
            pose_hgl_best_affinity=float(joint.max()),pose_hgl_mean_affinity=float(joint.mean()))
        for kind,start,stop in variants:
            if kind=='g':
                out.update(best_map_position_error_mm_median=float(np.median(ep[start:stop])*1000),
                    best_map_angle_error_deg_median=float(np.rad2deg(np.median(er[start:stop]))))
                for j,index in enumerate(row['sample']):
                    samples.append({'cohort':row['cohort'],'id':row['id'],'source_id':row['source_id'],'level':row['level'],
                        'candidate_index':int(index),'affinity':float(affinity[start+j]),
                        'translation_error_m':float(ep[start+j]),'orientation_error_rad':float(er[start+j]),
                        'ik':bool(row['ik'][index]),'clear':bool(row['clear'][index])})
    summary=summarize(serial,candidate_data)
    for cohort in ['all53','pilot3']:
        sub=[r for r in samples if r['cohort']==cohort];x=np.array([r['affinity'] for r in sub]);cuts=np.quantile(x,[.25,.5,.75]);bins=np.searchsorted(cuts,x,side='right')
        summary[cohort]['continuous_sample_size']=len(sub)
        summary[cohort]['sample_pose_affinity_ik_auc']=auc(x,[r['ik'] for r in sub])
        summary[cohort]['sample_pose_affinity_clear_auc']=auc(x,[r['clear'] for r in sub])
        summary[cohort]['sample_affinity_quartiles']=[{'bin':i,'candidates':int((bins==i).sum()),
            'mean_ik_fraction':float(np.mean([r['ik'] for r,b in zip(sub,bins) if b==i])),
            'mean_clear_fraction':float(np.mean([r['clear'] for r,b in zip(sub,bins) if b==i]))} for i in range(4)]
    save(ROOT/'rows.json',serial);save(ROOT/'sample_pose_queries.json',samples);save(ROOT/'summary.json',summary)
    np.savez_compressed(ROOT/'pose_queries.npz',position_body4=queries,affinity=affinity,position_error_m=ep,orientation_error_rad=er)
    for p,h in {**reg['input_sha256'],**reg['source_sha256']}.items():assert sha(p)==h,p
    save(ROOT/'completion_verification.json',{'passed':True,'distance_positions':336,'primary_positions':318,'pilot_positions':18,
        'all_candidate_position_queries':sum(r['candidates'] for r in serial),'continuous_pose_queries':len(queries),
        'source_inputs_unchanged':True,'map_modified':False,'new_IK_queries':0,'physical_attempts':0,'wall_seconds':time.perf_counter()-began})
    save(ROOT/'progress.json',{'phase':'complete','primary_positions':318,'pilot_positions':18,'wall_seconds':time.perf_counter()-began})
    print('COMPLETE',json.dumps(summary['all53']['correlations'],ensure_ascii=False,default=lambda x:x.tolist()),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);a=p.parse_args()
    try:
        prepare() if a.stage=='prepare' else run()
    except BaseException:
        import traceback
        if ROOT.exists():save(ROOT/(a.stage+'_error.json'),{'error':traceback.format_exc()})
        raise
