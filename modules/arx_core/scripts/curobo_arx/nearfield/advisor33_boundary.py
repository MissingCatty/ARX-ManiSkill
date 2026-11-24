"""Reject an out-of-model measured hand without changing its state or limits."""
import math

RANGE_ERROR='Measured gripper position is outside the validated model range'
REJECTION='measured_gripper_outside_frozen_model_range'
ISOLATED_OUTCOMES=frozenset(('stationary_wait_10s_timeout','planning_rejected: '+REJECTION,
                           'closure_model_rejected: '+REJECTION))


def range_rejection(error,request,context):
    if str(error)!=RANGE_ERROR or request['action'] not in ('plan','closure_validate'):
        raise error
    mode='closed' if request.get('phase')=='lift' else 'open'
    updater=context[mode][2];snapshot=request['snapshot'];rows=[]
    for name in updater.names:
        index=snapshot['joint_names'].index(name);value=float(snapshot['joint_positions'][index])
        lo,hi=updater.limits[name];actual=snapshot['joint_limits'][index]
        if not all(math.isfinite(x) for x in (value,lo,hi,*actual)):
            raise ValueError('Nonfinite measured hand/model limits remain an infrastructure fault')
        if max(abs(lo-actual[0]),abs(hi-actual[1]))>1e-9:
            raise RuntimeError('Runtime/planning hand limits disagree')
        rows.append({'joint':name,'measured_m':value,'lower_m':lo,'upper_m':hi,
                     'existing_tolerance_m':1e-5,'passes_existing_range':lo-1e-5<=value<=hi+1e-5})
    if not any(not r['passes_existing_range'] for r in rows):raise error
    return {'pass':False,'rejected':REJECTION,'phase':request.get('phase'),
            'measured_gripper':rows,'requires_world_isolation':True,
            'state_clamped':False,'limits_or_tolerance_changed':False,'new_command_generated':False}
