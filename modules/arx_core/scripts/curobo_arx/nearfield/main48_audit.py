"""A/B provenance checks; original action/physical checks remain independent."""
from pathlib import Path
import numpy as np
from scripts.curobo_arx.data_factory.common import read, write
from training.nearfield_arx.learned_policy import read_bound_reply
from .main48_spec import GEOMETRY, select_candidate


def same_selection(actual, recorded, path=''):
    """Exact decisions/ranking, numerical tolerance only for diagnostic floats.

    SciPy versions polar-project float32 GraspNet rotations slightly differently.
    No acceptance test or policy threshold is changed: candidate membership,
    rejection reasons, selected index, symmetry branch and order must be exact.
    """
    if isinstance(actual,dict) and isinstance(recorded,dict):
        return set(actual)==set(recorded) and all(same_selection(actual[k],recorded[k],path+'/'+k) for k in actual)
    if isinstance(actual,list) and isinstance(recorded,list):
        return len(actual)==len(recorded) and all(same_selection(a,b,path+'/'+str(i)) for i,(a,b) in enumerate(zip(actual,recorded)))
    if type(actual) is float and type(recorded) is float:
        tolerance=1e-7 if path.endswith('/rotation_rad') else 1e-12
        return bool(np.isfinite(actual) and np.isfinite(recorded) and abs(actual-recorded)<=tolerance)
    return type(actual) is type(recorded) and actual==recorded


def audit_baseline(episode, trial):
    episode=Path(episode); method=trial['method']
    if method not in ('A','B'): raise ValueError('A/B audit cannot accept a learned policy')
    events=read(episode/'interpreter_events.json') if (episode/'interpreter_events.json').exists() else []
    commands=read(episode/'requests.json') if (episode/'requests.json').exists() else []
    ledger=read(episode/'geometry_requests.json') if (episode/'geometry_requests.json').exists() else []
    ids=[r['request_id'] for r in ledger]; folder=episode/'geometry'
    files={p.name.removesuffix('_request.json') for p in folder.glob('*_request.json')}
    files|={p.name.removesuffix('_reply.json') for p in folder.glob('*_reply.json')}
    used={e['geometry_request_id'] for e in events if e.get('geometry_request_id')}
    if len(ids)!=len(set(ids)) or files!=set(ids) or not used<=files:
        raise RuntimeError('Geometry request/event/file chain damaged')
    if len(ids)>GEOMETRY['max_queries'] or method=='A' and ids: raise ValueError('Baseline query budget exceeded')
    motions=[e for e in events if e['kind']=='target_request']
    for e in motions:
        if e.get('baseline_method')!=method or e.get('policy_request_id'):
            raise RuntimeError('Unexpected baseline goal generator')
        if method=='A': np.testing.assert_allclose(e['goal_world'],trial['G0_world'],atol=1e-12,rtol=0)
    if method=='A' and len(motions)>1: raise RuntimeError('A issued multiple targets')
    queries=[]
    for ident in ids:
        request,reply=read_bound_reply(folder/(ident+'_request.json'),folder/(ident+'_reply.json'))
        if request['method']!='B' or request['G0_world']!=trial['G0_world'] or request['trial_id']!=trial['trial_id']:
            raise ValueError('Geometry request prior/case mismatch')
        proof=folder/(ident+'_evidence'); row=read(request['observation'])
        selected=select_candidate(np.load(proof/'filtered_predictions.npy',allow_pickle=False),
            row['camera_world_optical'],request['G0_world'],np.load(proof/'target_points.npy',allow_pickle=False))
        recorded=read(proof/'candidate_selection.json')
        if not same_selection(selected,recorded) or any(not same_selection(selected[k],reply[k],'/'+k) for k in ('selected','accepted')):
            raise ValueError('Geometry candidate ranking/conversion changed')
        linked=[e for e in motions if e.get('geometry_request_id')==ident]
        if len(linked)>1: raise ValueError('Geometry request dispatched twice')
        dispatch=folder/(ident+'_dispatch_check.json')
        if not dispatch.is_file(): raise RuntimeError('Missing geometry dispatch check')
        if linked:
            if not read(dispatch)['pass'] or not reply['projection']['target_visibility_pass'] or reply['selected'] is None:
                raise ValueError('Invalid geometry result dispatched')
            np.testing.assert_allclose(linked[0]['goal_world'],reply['selected']['goal_world'],atol=1e-12,rtol=0)
        if linked and linked[0].get('command_label') not in ('b_refine_001','b_refine_002'):
            raise ValueError('Invalid geometry command label')
        queries.append({'request_id':ident,'raw_index':request['raw_index'],'dispatched_event':bool(linked),
            'selected':reply['selected'],'reason':reply['rejected'],'seconds':reply['total_seconds']})
    for command in commands:
        label=command['label']
        if label.startswith(('a_approach','b_refine_')):
            if len([e for e in motions if e.get('command_label')==label])!=1:
                raise RuntimeError('Baseline action without unique target event')
        elif label.startswith(('c_step_','d_refine_')): raise RuntimeError('Baseline executed learned-labelled action')
        elif label=='close':
            close=[e for e in events if e['kind']=='grasp_request']
            if len(close)!=1 or close[0].get('closure_origin')!='baseline_'+method or not motions:
                raise RuntimeError('Baseline closure has no accepted method target')
    return {'pass':True,'method':method,'policy_calls':0,'requests':[],
            'geometry_queries':queries,'geometry_calls':len(queries)}
