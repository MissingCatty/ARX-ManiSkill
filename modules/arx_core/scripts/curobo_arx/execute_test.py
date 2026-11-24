"""Execute blocked baseline and cuRobo detour in the ARX Isaac simulation."""
import argparse,json,time
from pathlib import Path
import numpy as np
from client import Client

root=Path(__file__).resolve().parents[2];out=root/'runs/arx_curobo';client=Client(out/'mailbox')
report=json.loads((out/'plan_report.json').read_text());names=report['joint_names']
base=np.asarray(json.loads((out/'snapshot.json').read_text())['link_world']['body4'])

def current():
    s=client.request('snapshot');d=dict(zip(s['joint_names'],s['joint_positions']))
    return np.array([d[n] for n in names])

def obstacle(remove=False):
    return client.request('obstacle',remove=remove,center=report['obstacle_center_world'],dims=report['obstacle_dims'])

def external_contacts(result):
    return [c for c in result['contacts'] if c['num_contacts']>0 and
            not ('/World/Robot/' in c['a'] and '/World/Robot/' in c['b'])]

parser=argparse.ArgumentParser()
parser.add_argument('--resume-planned',action='store_true',help='Reuse the recorded contact baseline for the same saved plan')
args=parser.parse_args()
results=json.loads((out/'execution_report.json').read_text()) if args.resume_planned else {}
if args.resume_planned:
    assert results['baseline']['obstacle_contact_events']>0
for label in (['planned'] if args.resume_planned else ['baseline','planned']):
    obstacle(remove=True)
    # Removing a blocking collider releases the drives toward their held target.
    # Wait for that motion to settle before defining the return trajectory.
    previous=current();stable=0;deadline=time.monotonic()+90
    while stable<3:
        if time.monotonic()>deadline:raise RuntimeError('ARX did not settle after obstacle removal')
        time.sleep(1)
        q=current();stable=stable+1 if np.max(abs(q-previous))<.003 else 0;previous=q
    start=np.array(report['start'])
    if np.max(abs(q-start))>.015:
        # Return along the preceding commanded path, with the test obstacle removed.
        prev=np.asarray(json.loads((out/'baseline.json').read_text())['position'])
        path=np.vstack([q,prev[::-1]])
        limits=np.asarray(client.request('snapshot')['joint_limits'])
        s=client.request('snapshot'); ids=[s['joint_names'].index(n) for n in names]
        path=np.clip(path,limits[ids,0],limits[ids,1])
        reset=client.request('trajectory',timeout=240,joint_names=names,position=path.tolist(),time_from_start=np.linspace(0,8,len(path)).tolist())
        (out/'reset_execution.json').write_text(json.dumps(reset))
        assert reset['settled'],'ARX failed to return to test start'
    obstacle()
    command=json.loads((out/f'{label}.json').read_text())
    result=client.request('trajectory',timeout=240,**command)
    (out/f'{label}_execution.json').write_text(json.dumps(result))
    positions=np.asarray([r['position'] for r in result['records']]);targets=np.asarray([r['target'] for r in result['records']])
    contacts=external_contacts(result)
    results[label]={'settled':result['settled'],'physics_samples':len(result['records']),
        'max_tracking_error_rad':float(np.max(abs(positions-targets))),
        'end_joint_error_rad':float(np.max(abs(positions[-1]-report['goal']))),
        'external_contact_events':len(contacts),'obstacle_contact_events':sum('CuroboValidationObstacle' in c['a']+c['b'] for c in contacts),
        'contact_pairs':sorted({tuple(sorted([c['a'],c['b']])) for c in contacts})}
    print(label,json.dumps(results[label]),flush=True)
    (out/'execution_report.json').write_text(json.dumps(results,indent=2))
assert results['baseline']['obstacle_contact_events']>0,'Contact detector was not positively validated by blocked baseline'
assert results['planned']['settled'],'Planned ARX motion did not reach its goal'
assert results['planned']['external_contact_events']==0,'Planned ARX execution made external contact'
print('PASS: measured ARX obstacle avoidance, with positive contact baseline',flush=True)
