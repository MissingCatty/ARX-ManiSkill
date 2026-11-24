"""Streaming v1 export: split by shape/object before constructing action chunks."""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import VERSION,read,sha,verify_files,write
from .ledger import Ledger


def histogram_quantiles(roots, minimum, maximum, count, bins=4096):
    """Second streaming pass; quantile intervals have bounded value error."""
    minimum,maximum=np.asarray(minimum),np.asarray(maximum)
    width=(maximum-minimum)/bins
    hist=np.zeros((15,bins),dtype=np.int64)
    measured=0
    for root in roots:
        labels=read(Path(root)/'labels.json')
        values=np.array([item['state']+item['action'] for item in labels['items']],dtype=float)
        if values.ndim!=2 or values.shape[1]!=15 or not np.isfinite(values).all():
            raise ValueError('Invalid normalization rows')
        if (values<minimum).any() or (values>maximum).any():
            raise ValueError('Normalization source changed between passes')
        indices=np.floor((values-minimum)/np.where(width>0,width,1)).astype(np.int64)
        indices=np.minimum(indices,bins-1)
        for dim in range(15):hist[dim]+=np.bincount(indices[:,dim],minlength=bins)
        measured+=len(values)
    if measured!=count:raise ValueError('Training frame count changed between passes')
    result={'method':'Two-pass fixed-range histogram, nearest-rank empirical quantile; midpoint of containing bin',
        'bins_per_dimension':bins,'maximum_absolute_value_error':(width/2).tolist(),
        'scope':'Each empirical quantile is within its reported containing interval; no distributional/generalization bound claimed'}
    cumulative=hist.cumsum(axis=1)
    for name,p in (('q01',.01),('q99',.99)):
        rank=max(1,int(np.ceil(p*count)))
        index=np.array([np.searchsorted(row,rank) for row in cumulative])
        lo=minimum+index*width;hi=np.minimum(maximum,lo+width)
        result[name]=((lo+hi)/2).tolist()
        result[name+'_interval']=np.stack([lo,hi],axis=1).tolist()
    return result


def export(campaign,out):
    campaign,out=Path(campaign),Path(out)
    out.mkdir(parents=True,exist_ok=False)
    reg=read(campaign/'registration.json')
    ledger=Ledger(campaign/'ledger.sqlite3')
    episodes=[r for r in ledger.rows() if r['state']=='committed'];ledger.close()
    group_roles={};objects={o['object_id']:o for o in reg['catalog']['objects']}
    for obj in reg['catalog']['objects']:
        group=obj.get('geometry_group',obj['object_id'])
        if group in group_roles and group_roles[group]!=obj['split']:raise ValueError('Shape leakage across splits')
        group_roles[group]=obj['split']
    count=0;mean=np.zeros(15);m2=np.zeros(15);minimum=np.full(15,np.inf);maximum=np.full(15,-np.inf)
    chunks=0;training_roots=[]
    streams={split:(out/(split+'.jsonl')).open('w') for split in ('train','validation','development')}
    try:
        for episode in episodes:
            root=Path(episode['episode']);receipt=read(root/'quality_receipt.json');meta=read(root/'episode.json')
            committed=json.loads(episode['receipt'])
            if committed['quality_sha256']!=sha(root/'quality_receipt.json') or committed['physical_sha256']!=sha(root/'physical_receipt.json'):
                raise ValueError('Committed acceptance receipt changed before export')
            if receipt['physical_receipt_sha256']!=committed['physical_sha256']:
                raise ValueError('Export acceptance-chain mismatch')
            verify_files({str(root/p):s for p,s in receipt['files'].items()})
            if not receipt['pass'] or not read(root/'physical_receipt.json')['pass']:raise ValueError('Unaccepted episode in success index')
            obj=objects[episode['object_id']]
            if meta['attempt_id']!=episode['id'] or meta['object_id']!=episode['object_id'] or meta['split']!=obj['split'] or meta['geometry_group']!=obj['geometry_group']:
                raise ValueError('Committed episode identity/split differs from registered catalog')
            labels=read(root/'labels.json');items=labels['items'];role=meta['split']
            for start in labels['chunk_starts']:
                block=items[start:start+8]
                record={'version':VERSION,'episode_id':episode['id'],'object_id':meta['object_id'],
                    'geometry_group':meta['geometry_group'],'split':role,'episode_path':str(root),
                    'raw_indices':[r['raw_index'] for r in block],'state':block[0]['state'],
                    'actions':[r['action'] for r in block],'instruction':labels['instruction'],
                    'preprocessing':'released_jpeg_lanczos3_crop90_arx_v1','raw_images_preserved':True}
                streams[role].write(json.dumps(record,separators=(',',':'))+'\n');chunks+=1
            if role=='train':
                training_roots.append(root)
                for item in items:
                    value=np.r_[item['state'],item['action']]
                    count+=1;delta=value-mean;mean+=delta/count;m2+=delta*(value-mean)
                    minimum=np.minimum(minimum,value);maximum=np.maximum(maximum,value)
    finally:
        for stream in streams.values():stream.close()
    stats={'version':VERSION,'train_frames':count,'mean':mean.tolist() if count else None,
        'std':np.sqrt(m2/max(count-1,1)).tolist() if count else None,'minimum':minimum.tolist() if count else None,
        'maximum':maximum.tolist() if count else None,'layout':'8 state + 7 action',
        'quantiles':histogram_quantiles(training_roots,minimum,maximum,count) if count else None,
        'method':'Exact streaming Welford mean/sample variance and exact min/max; bounded-value-error histogram quantiles',
        'normalization_scope':'Training objects only; no VLM feature caching or training started'}
    write(out/'statistics.json',stats)
    write(out/'manifest.json',{'version':VERSION,'source_campaign':str(campaign),'episodes':len(episodes),'chunks':chunks,
        'lineage':reg.get('lineage',[]),
        'split_before_chunking':True,'registry_sha256':sha(campaign/'registration.json'),
        'exporter_files':{str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('common.py'),Path(__file__).with_name('ledger.py'))},
        'format':'Streaming JSONL shard indexes referencing original full RGB-D/state episodes',
        'image_codec':'Use unchanged observations.policy_observation then released service image preprocessing; raw image remains authoritative'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();export(a.campaign,a.out)
