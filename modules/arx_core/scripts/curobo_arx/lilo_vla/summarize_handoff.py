"""Summarize measured A1 handoffs without treating plans as physical success."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def summarize(folder):
    folder=Path(folder)
    root=Path(__file__).resolve().parents[3]
    stats=json.loads((root/'runs/lilo_dataset_20261026_train03_05_val06_v2/dataset_statistics.json').read_text())['arx_bottle_local']['proprio']
    lower,upper=np.asarray(stats['min']),np.asarray(stats['max'])
    labels=('x','y','z','rx','ry','rz','finger27','finger28')
    rows=[]
    for trial in sorted(p for p in folder.iterdir() if p.is_dir() and (p/'scenario.json').exists()):
        row={'trial':trial.name,'complete':(trial/'result.json').exists()}
        if row['complete']:row['evaluation']=json.loads((trial/'result.json').read_text())
        predictions=sorted(trial.glob('local_*_prediction.json'))
        row['policy_calls']=len(predictions)
        if predictions:
            first=json.loads(predictions[0].read_text());state=np.asarray(first['raw_state'])
            row.update(first_state=state.tolist(),first_normalized_state=first['normalized_state'],
                outside_training_minmax=[n for n,b in zip(labels,(state<lower-1e-6)|(state>upper+1e-6)) if b],
                first_action=first['actions'][0],first_dz_m=first['actions'][0][2]*.05,
                first_gripper='open' if first['actions'][0][6]>=.5 else 'close')
            target=trial/'handoff_target.json'
            if target.exists():
                reference=np.asarray(json.loads(target.read_text())['reference_state'])
                row['reference_position_error_m']=float(np.linalg.norm(state[:3]-reference[:3]))
                row['reference_orientation_error_rad']=float((Rotation.from_rotvec(state[3:6])*Rotation.from_rotvec(reference[3:6]).inv()).magnitude())
            initial=json.loads((trial/'capture_snapshot.json').read_text())['objects']['test_object']['world'][2][3]
            heights=[]
            for path in trial.glob('*_execution.json'):
                execution=json.loads(path.read_text())
                heights.extend(r['objects']['test_object'][2][3]-initial for r in execution.get('records',[]) if 'test_object' in r.get('objects',{}))
            row['maximum_recorded_object_rise_m']=max(heights) if heights else None
        rows.append(row)
    result={'scope':'Only A1, same adapter/action execution; goal-pose interventions, not a full original-LiLo reproduction',
        'training_min':stats['min'],'training_max':stats['max'],'rows':rows,
        'interpretation':'A failed baseline and successful repeated intervention would support a start-pose contribution. Failure after a measured in-range handoff shows start correction alone is insufficient; it does not prove start pose is irrelevant.'}
    (folder/'handoff_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    print(json.dumps(summarize(parser.parse_args().folder),indent=2))
