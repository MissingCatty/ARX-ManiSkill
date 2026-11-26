"""Build a reversible representative-first queue without changing grasps."""
import numpy as np

from .geometry import poses_from_candidates
from .nms import native_nms


def validate_order(count, representatives, deferred):
    groups=[representatives,deferred]
    if any(type(i) is not int for group in groups for i in group):
        raise ValueError('Candidate source IDs must be integers')
    if any(group!=sorted(group) or len(set(group))!=len(group) for group in groups):
        raise ValueError('Keep native score order within each priority phase')
    order=representatives+deferred
    if len(order)!=count or set(order)!=set(range(count)):
        raise ValueError('NMS queue must preserve every original candidate exactly once')
    return order


def make_queue(payload, variant):
    candidates=payload['candidates'];count=len(candidates)
    if variant=='control':
        representatives=list(range(count));deferred=[];nms=None
    elif variant=='nms15':
        poses,_=poses_from_candidates(candidates)
        scores=np.asarray([c['score'] for c in candidates],dtype=float)
        widths=np.asarray([c['width'] for c in candidates],dtype=float)
        if (np.diff(scores)>1e-12).any():raise ValueError('Original native score order changed')
        nms=native_nms(poses,scores,widths,.015,np.deg2rad(15))
        representatives,deferred=nms['representatives'],nms['deferred']
    else:raise ValueError('Unknown V7 candidate-order variant')
    order=validate_order(count,representatives,deferred)
    manifest={'variant':variant,'source_count':count,'representative_count':len(representatives),
        'representatives':representatives,'deferred':deferred,'priority_to_original_source':order,
        'all_original_candidates_retained':True,'candidate_pose_score_width_and_metadata_changes':False,
        'geometry_hard_filter_enabled':False,'NMS':nms}
    ordered={**payload,'candidates':[candidates[i] for i in order],'native_score_order':variant=='control',
        'candidate_order_policy':{'version':'V7','variant':variant,
            'phase_order':['NMS_representatives','remaining_original_candidates'],
            'representative_count':len(representatives),'original_indices':order,
            'original_native_score_order':variant=='control'}}
    return ordered,manifest


def selection_source(manifest, selected):
    order=validate_order(manifest['source_count'],manifest['representatives'],manifest['deferred'])
    if order!=manifest['priority_to_original_source']:raise ValueError('Candidate order manifest changed')
    index=int(selected['index'])
    if not 0<=index<len(order):raise ValueError('Selection is outside its priority queue')
    return {'original_source_index':order[index],'search_priority_rank':index,
        'priority_phase':'representative' if index<manifest['representative_count'] else 'deferred'}
