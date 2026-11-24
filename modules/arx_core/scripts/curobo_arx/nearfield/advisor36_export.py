"""Independent native-mask and actual-FK admission of paired static captures."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scripts.curobo_arx.data_factory.common import read,write,sha,verify_files
from .advisor36_contract import example,residual,token
from .capture import validate_saved_capture
from training.nearfield_arx import native_mask


def load_lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def pair_quality(rows):
    if len(rows)!=2 or {r['candidate']['pair_member'] for r in rows}!={-1,1}:
        return {'pass':False,'reason':'missing_mirror_member'}
    if not all(r.get('capture_admitted') for r in rows):
        return {'pass':False,'reason':'one_or_both_members_rejected'}
    if len({r['split'] for r in rows})!=1:raise ValueError('Mirror pair crosses split')
    a,b=rows
    x=np.asarray(a['actual_translation_m']);y=np.asarray(b['actual_translation_m'])
    rotation_sum=np.asarray(a['actual_rotation_degrees'])+np.asarray(b['actual_rotation_degrees'])
    error=np.r_[x[:2]+y[:2],x[2]-y[2]]
    passed=bool(np.max(np.abs(error))<=.001+1e-9 and np.linalg.norm(rotation_sum)<=.4+1e-8)
    return {'pass':passed,'reason':None if passed else 'actual_FK_pair_asymmetry',
        'actual_pair_translation_residual_m':error.tolist(),'actual_pair_rotation_sum_degrees':rotation_sum.tolist(),
        'actual_labels_preserved_without_balancing_or_rounding_adjustment':True}


def export(campaign,output,group_ids=None):
    campaign=Path(campaign).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    reg=read(campaign/'registration.json');static=reg['static_observations']
    verify_files(reg['dependencies'])
    candidates={r['sample_id']:r for r in load_lines(static['candidate_table'])}
    registry=read(static['canonical_registry'])
    projector=native_mask.NativeProjector(reg['learned_policy']['robot_usd'])
    scale=reg['learned_policy']['visual_contract']['depth_scale']
    groups=([Path(p) for p in static.get('prior_capture_groups',[])]+sorted((campaign/'groups').iterdir())
            if group_ids is None else [campaign/'groups'/g for g in group_ids])
    records=[];seen=set();pairs=defaultdict(list);surface_geometry_cache={}
    def surface_geometry(path):
        # One geometry file is shared by every observation in a static World.
        # Preserve the digest check while avoiding repeatedly parsing hundreds
        # of megabytes. A changed size/mtime is a new read, never a stale hit.
        from scripts.curobo_arx.data_factory.common import digest
        p=Path(path);stat=p.stat();key=(str(p),stat.st_size,stat.st_mtime_ns)
        if key not in surface_geometry_cache:
            value=read(p)
            if digest({k:v for k,v in value.items() if k!='geometry_sha256'})!=value['geometry_sha256']:
                raise ValueError('Static surface geometry content digest failed')
            surface_geometry_cache[key]=value['geometry_sha256']
        return surface_geometry_cache[key]
    for group in groups:
        complete=read(group/'completion.json')
        if not complete['pass'] or complete['processed']!=complete['registered']:
            raise RuntimeError('Static group incomplete')
        for capture in read(group/'results.json')['observations']:
            ident=capture['sample_id'];candidate=candidates[ident]
            if ident in seen:raise RuntimeError('Static candidate replayed')
            seen.add(ident)
            if capture['candidate']!=candidate:raise ValueError('Captured candidate differs from registered table')
            record={**capture,'capture_admitted':False,'admitted':False}
            records.append(record);pairs[candidate['pair_id']].append(record)
            if ident in static.get('prior_admission_exclusions',{}):
                record['export_rejected']=static['prior_admission_exclusions'][ident];continue
            if not capture['pass_']:continue
            verify_files(capture['files'])
            folder=Path(capture['observation']).parent;obs=read(capture['observation'])
            validate_saved_capture(folder,obs)
            setter=read(capture['static_setter'])
            if static.get('actual_target_surface_check_required'):
                from scripts.curobo_arx.data_factory.common import digest
                surface=read(folder/'static_surface_receipt.json')
                request=read(folder/'surface_request.json')
                if (not surface['pass'] or surface!=capture.get('target_surface_check') or
                    surface['sample_snapshot_sha256']!=digest(setter['after']) or
                    request['snapshot']!=setter['after'] or request['sample_id']!=ident or
                    surface_geometry(request['geometry'])!=surface['geometry_sha256']):
                    raise ValueError('Static target surface/request/actual state binding failed')
            if obs['capture']['acquisition_mode']!='independent_static_joint_state_v1' or obs['capture']['sample_id']!=ident:
                raise ValueError('Static observation identity mismatch')
            from scripts.curobo_arx.data_factory.common import digest
            if obs['capture']['static_setter_sha256']!=digest(setter):raise ValueError('Static setter not bound to image')
            if np.max(np.abs(obs['snapshot']['joint_velocities']))>1e-7:raise ValueError('Nonzero static velocity')
            rgb=folder/obs['rgb_relative'];depth=np.load(folder/obs['depth_relative'],allow_pickle=False)
            bound=native_mask.bind(obs,rgb)
            asset=registry['objects'][candidate['object_id']]['source']['asset']['usd_path']
            masks,visibility,_=projector.project(asset,bound,depth)
            record['native_visibility']=visibility
            if not visibility['target_visibility_pass']:
                record['export_rejected']='target_native_visibility';continue
            # Recompute labels independently from stored same-frame poses.
            state=obs['snapshot'];g=np.asarray(state['objects']['test_object']['world'])@np.asarray(candidate['canonical_object'])
            dp,dr=residual(state['link_world']['right_eef_link'],g)
            core=bool(np.linalg.norm(dp)<=.004 and np.linalg.norm(dr)<=np.deg2rad(3))
            if core!=(candidate['category']=='grasp'):
                record['export_rejected']='actual_core_differs_from_candidate_class';continue
            output=token(state['link_world']['right_eef_link'],g,grasp=core)
            if output!=capture['output']:raise ValueError('Saved action label disagrees with actual same-frame FK')
            np.testing.assert_allclose(dp,capture['actual_translation_m'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(np.rad2deg(dr),capture['actual_rotation_degrees'],atol=1e-10,rtol=0)
            views=native_mask.model_views(Image.open(rgb),depth,masks,scale)
            dest=out/'views'/ident;dest.mkdir(parents=True)
            for i,img in enumerate(views):img.save(dest/f'image{i+1}.png')
            item=example([dest/'image1.png',dest/'image2.png'],scale,output)
            item.update(sample_id=ident,object_id=candidate['object_id'],pair_id=candidate['pair_id'],
                split=candidate['split'],neighborhood_group=candidate['neighborhood_group'],category=candidate['category'],
                axial_layer_mm=candidate['axial_layer_mm'],actual_translation_m=dp.tolist(),
                actual_rotation_degrees=np.rad2deg(dr).tolist(),source_observation=capture['observation'])
            record.update(capture_admitted=True,model_item=item)
    pair_checks=[]
    for pair_id,members in pairs.items():
        check=pair_quality(members);pair_checks.append({'pair_id':pair_id,**check})
        for r in members:r.update(admitted=check['pass'],pair_check=check)
    admitted=[r for r in records if r['admitted']]
    for split in ('train','validation','test','unseen_evaluation'):
        with (out/(split+'.jsonl')).open('x') as stream:
            for row in admitted:
                if row['split']==split:stream.write(json.dumps(row['model_item'],separators=(',',':'))+'\n')
    with (out/'admission.jsonl').open('x') as stream:
        for r in records:stream.write(json.dumps(r,separators=(',',':'))+'\n')
    write(out/'pair_checks.json',pair_checks)
    objects=[]
    for obj in registry['objects']:
        sub=[r for r in records if r['object_id']==obj];ok=[r for r in sub if r['admitted']]
        objects.append({'object_id':obj,'captured_candidates':len(sub),'admitted':len(ok),
            'split_counts':dict(Counter(r['split'] for r in ok)),
            'category_counts':dict(Counter(r['category'] for r in ok)),
            'funnel_axial_counts':dict(Counter(r['candidate']['axial_layer_mm'] for r in ok if r['category']=='funnel')),
            'rejections':dict(Counter(r.get('export_rejected') or r.get('rejected') or r['pair_check']['reason'] for r in sub if not r['admitted']))})
    summary={'complete':True,'observed_candidates':len(records),'admitted_observations':len(admitted),
        'registered_candidate_count':static['candidate_count'],
        'unexecuted_after_infrastructure_interruption':static.get('excluded_without_replay',[]),
        'prior_admission_exclusions':static.get('prior_admission_exclusions',{}),
        'admitted_pairs':sum(p['pass'] for p in pair_checks),'objects':objects,
        'split_counts':dict(Counter(r['split'] for r in admitted)),
        'native_instance_mask':True,'mesh_depth_3mm_filter':False,'training_started':False,
        'physical_success_claim_per_static_observation':False,
        'files':{str(p):sha(p) for p in out.rglob('*') if p.is_file()}}
    write(out/'summary.json',summary)
    return {k:summary[k] for k in ('observed_candidates','admitted_observations','admitted_pairs','split_counts')}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('campaign');p.add_argument('output');p.add_argument('--groups',nargs='+')
    a=p.parse_args();print(export(a.campaign,a.output,a.groups))
