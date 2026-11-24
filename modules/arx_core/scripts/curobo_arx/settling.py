"""Evaluate bounded, stationary ARX tracking error against the intended tool pose."""
import numpy as np

def assess_settling(execution, goal_world):
    records=execution['records'];last=records[-1]
    joint_error=float(np.max(np.abs(np.array(last['position'])-last['target'])))
    goal=np.asarray(goal_world);actual=np.asarray(last['ee_world'])
    position_error=float(np.linalg.norm(actual[:3,3]-goal[:3,3]))
    angle=float(np.arccos(np.clip((np.trace(actual[:3,:3]@goal[:3,:3].T)-1)/2,-1,1)))
    tail=[r for r in records if r['time']>=last['time']-1.0]
    span=tail[-1]['time']-tail[0]['time']
    drift=float(np.max(np.ptp(np.array([r['position'] for r in tail]),axis=0)))
    bounded=bool(np.isfinite([joint_error,position_error,angle,drift]).all()
                 and joint_error<=.01 and position_error<=.004 and angle<=.02
                 and span>=.75 and drift<=.0005)
    return {'accepted':bool(execution['settled'] or bounded),
            'strict_joint_settled':bool(execution['settled']),
            'bounded_stationary_pose':bounded,
            'endpoint_max_error_rad':joint_error,'tool_position_error_m':position_error,
            'tool_orientation_error_rad':angle,'last_second_joint_drift_rad':drift}
