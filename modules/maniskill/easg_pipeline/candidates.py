"""Authoritative ZeroGrasp postprocessed candidates, with no raw fallback."""
import hashlib
from pathlib import Path

import numpy as np

from common import read,save,sha

STAGE='zerograsp_collision_refinement_nms'


def rows(path):
    value=np.load(path,allow_pickle=False)
    if value.ndim!=2 or value.shape[1]!=17 or value.dtype.kind not in 'fi' or not np.isfinite(value).all():
        raise ValueError('Invalid ZeroGrasp candidate array: '+str(path))
    return value


def digest_rows(value):
    return hashlib.sha256(np.ascontiguousarray(value,dtype=np.float64).tobytes()).hexdigest()


def finalize_postprocessing(directory,detections):
    """Audit unchanged native postprocessing; remove its all-rejected fallback."""
    directory=Path(directory);raw_files=sorted(directory.glob('*.raw_grasps.npy'))
    if len(raw_files)!=len(detections):raise RuntimeError('ZeroGrasp collision audit/object count differs')
    unused=list(detections);objects=[]
    for raw in raw_files:
        original=rows(raw);digest=digest_rows(original)
        matches=[d for d in unused if d['raw_rows_sha256']==digest]
        if len(matches)!=1:raise RuntimeError('Cannot match native collision call to original proposals')
        detection=matches[0];unused.remove(detection)
        accepted=np.asarray(detection['accepted_rows'],dtype=np.float64).reshape(-1,17)
        processed=raw.with_name(raw.name.removesuffix('.raw_grasps.npy')+'.grasp.npy')
        if not processed.is_file():raise RuntimeError('Missing native postprocessed file: '+str(processed))
        corrected=False
        if not len(accepted):
            # Native demo retains the old group when every collision test rejects.
            # A zero-survivor set must stay empty in the corrected adapter.
            np.save(processed,np.empty((0,17),dtype=np.float64));corrected=True
        final=rows(processed)
        if len(final)>len(accepted):raise RuntimeError('Postprocessing retained rejected candidates')
        for row in final:
            if not np.any(np.all(np.isclose(accepted,row,rtol=0.,atol=1e-7),axis=1)):
                raise RuntimeError('Postprocessed candidate lacks collision-pass evidence')
        proof=raw.with_name(raw.name.removesuffix('.raw_grasps.npy')+'.collision_pass.npy')
        np.save(proof,accepted)
        objects.append(dict(raw_file=raw.name,raw_sha256=sha(raw),raw_count=len(original),
            postprocessed_file=processed.name,postprocessed_sha256=sha(processed),
            collision_pass_count=len(accepted),final_count=len(final),
            collision_pass_file=proof.name,collision_pass_sha256=sha(proof),
            all_rejected_fallback_removed=corrected))
    expected={o['postprocessed_file'] for o in objects}
    if {p.name for p in directory.glob('*.grasp.npy')}!=expected:
        raise RuntimeError('Unexpected native postprocessed files')
    receipt=dict(candidate_stage=STAGE,objects=objects,collision_calls=len(detections),
        collision_filter=True,width_depth_refinement=True,nms=True,
        nms_distance_m=.03,nms_angle_deg=30.,raw_fallback=False,
        empty_reconstruction=not objects)
    save(directory/'postprocessing_receipt.json',receipt)
    return receipt


def export_candidates(network,out,*,fresh_inference=True):
    network=Path(network);out=Path(out);raw=network/'raw_outputs'
    audit_path=raw/'postprocessing_receipt.json'
    audit=read(audit_path)
    if audit.get('candidate_stage')!=STAGE or audit.get('raw_fallback') is not False:
        raise RuntimeError('Expected audited ZeroGrasp postprocessed output')
    arrays=[];sources=[]
    for obj in audit['objects']:
        name=obj['postprocessed_file'];source=raw/name
        if Path(name).name!=name or not name.endswith('.grasp.npy'):
            raise RuntimeError('Candidate source is not a native postprocessed file')
        if sha(source)!=obj['postprocessed_sha256']:raise RuntimeError('Native postprocessed file changed')
        value=rows(source)
        if len(value)!=obj['final_count']:raise RuntimeError('Native candidate count differs')
        proof=raw/obj['collision_pass_file']
        if sha(proof)!=obj['collision_pass_sha256']:raise RuntimeError('Collision-pass evidence changed')
        arrays.append(value);sources.append(dict(path=str(source.resolve()),sha256=sha(source),count=len(value)))
    if {p.resolve() for p in raw.glob('*.grasp.npy')}!={Path(s['path']) for s in sources}:
        raise RuntimeError('Postprocessed file inventory differs from collision audit')
    combined=np.concatenate(arrays) if arrays else np.empty((0,17),dtype=np.float64)
    artifact=out/'postprocessed_grasps.npy'
    if artifact.exists() or (out/'network_candidates.json').exists():raise RuntimeError('Candidate export already exists')
    np.save(artifact,combined)
    metadata=dict(fresh_inference=fresh_inference,candidate_stage=STAGE,count=len(combined),
        raw_count=sum(o['raw_count'] for o in audit['objects']),
        collision_pass_count=sum(o['collision_pass_count'] for o in audit['objects']),
        sources=sources,source_files=[s['path'] for s in sources],
        artifact=str(artifact.resolve()),artifact_sha256=sha(artifact),
        postprocessing_receipt=str(audit_path.resolve()),postprocessing_receipt_sha256=sha(audit_path),
        raw_fallback=False,mask='ManiSkill target instance segmentation')
    save(out/'network_candidates.json',metadata)
    return metadata


def load_candidates(directory,*,require_fresh=True):
    directory=Path(directory);metadata=read(directory/'network_candidates.json')
    if metadata.get('candidate_stage')!=STAGE or metadata.get('raw_fallback') is not False:
        raise RuntimeError('Planner requires ZeroGrasp postprocessed candidates')
    if require_fresh and metadata.get('fresh_inference') is not True:
        raise RuntimeError('Planner requires fresh inference for this episode')
    artifact=directory/'postprocessed_grasps.npy'
    if Path(metadata['artifact']).resolve()!=artifact.resolve() or sha(artifact)!=metadata['artifact_sha256']:
        raise RuntimeError('Processed planner input changed')
    audit_path=Path(metadata['postprocessing_receipt'])
    if sha(audit_path)!=metadata['postprocessing_receipt_sha256']:
        raise RuntimeError('Postprocessing provenance changed')
    audit=read(audit_path)
    for obj in audit['objects']:
        proof=audit_path.parent/obj['collision_pass_file']
        if sha(proof)!=obj['collision_pass_sha256']:raise RuntimeError('Collision-pass evidence changed')
    for source in metadata['sources']:
        p=Path(source['path'])
        if not p.name.endswith('.grasp.npy') or sha(p)!=source['sha256']:
            raise RuntimeError('Native postprocessed source changed')
    value=rows(artifact)
    if len(value)!=metadata['count']:raise RuntimeError('Processed planner candidate count differs')
    return value,metadata
