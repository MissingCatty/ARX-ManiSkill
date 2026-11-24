"""Keep P1 planning admission separate from actual ARX VLA grasp results."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image,ImageDraw

from summarize_chunks import encode,sha,read,write,ROOT


def summarize(folder,encode_videos):
    registration=read(folder/'registration.json')
    if registration.get('experiment') not in ('handoff','teacher'):raise ValueError('Expected a registered handoff/teacher experiment')
    results=read(folder/'results.json');rows=[]
    for result in results:
        trial=folder/result['label']
        report=read(trial/'grasp_report.json') if (trial/'grasp_report.json').exists() else {}
        selection=read(trial/'handoff_candidates.json') if (trial/'handoff_candidates.json').exists() else None
        entry=read(trial/'handoff_actual_entry.json') if (trial/'handoff_actual_entry.json').exists() else None
        provenance=[]
        sources=report.get('action_sources',[])
        for source in sources:
            saved=read(trial/f"local_{source['step']:04d}_prediction.json")
            original=read(trial/source['prediction_file'])
            if source['executed_policy_action_index']!=0 or saved['actions']!=original['actions']:
                provenance.append({'step':source['step'],'error':'P1 must consume action 0 with feedback each step'})
        if report.get('local_policy_calls',0)!=len(list(trial.glob('policy_*_prediction.json'))):
            provenance.append({'error':'Policy call count mismatch'})
        actual=[p for p in trial.glob('local_*_execution.json')]
        if report.get('local_action_count',0)!=len(actual):provenance.append({'error':'Executed action count mismatch'})
        frozen={}
        for name in ('model','model_closed'):
            dest=trial/name;base=ROOT/'scripts/curobo_arx/static_pick/reference'/name
            if not (dest/'arx.yml').exists():continue
            current=yaml.safe_load((dest/'arx.yml').read_text());reference=yaml.safe_load((base/'arx.yml').read_text())
            a=current['robot_cfg']['kinematics'];b=reference['robot_cfg']['kinematics']
            a['urdf_path']=b['urdf_path'];a['lock_joints']=b['lock_joints']
            a['cspace']['default_joint_position']=b['cspace']['default_joint_position']
            frozen[name]=current==reference and sha(dest/'arx.urdf')==sha(base/'arx.urdf')
        initial=read(trial/'capture_snapshot.json')['objects']['test_object']['world'][2][3] if (trial/'capture_snapshot.json').exists() else None
        heights=[]
        if initial is not None:
            for path in trial.glob('*_execution.json'):
                heights.extend(r['objects']['test_object'][2][3]-initial for r in read(path).get('records',[]))
        predictions=sorted(trial.glob('policy_*_prediction.json'))
        first=read(predictions[0]) if predictions else None
        video=encode(trial) if encode_videos else (read(trial/'video_receipt.json') if (trial/'video_receipt.json').exists() else None)
        rows.append({**result,'candidate_selection':selection,'actual_entry':entry,
            'action_provenance_pass':not provenance,'action_provenance_errors':provenance,
            'frozen_planning_models':frozen,'maximum_recorded_object_rise_m':max(heights) if heights else None,
            'first_normalized_state':first.get('normalized_state') if first else None,
            'video':video,'planning_pass_is_not_physical_grasp_success':True})
    groups=[]
    for scene in ('A1','bottle'):
        for variant in ('baseline','grasp_frame'):
            group=[r for r in rows if r['scene']==scene and r['variant']==variant]
            groups.append({'scene':scene,'variant':variant,'attempts':len(group),'successes':sum(r['pass'] for r in group),
                'failures':[{'label':r['label'],'error':r['evaluation'].get('reported_error')} for r in group if not r['pass']]})
    protected={**registration['physical_source_sha256'],**registration['policy_artifact_sha256']}
    sources={**registration['algorithm_sha256'],**registration['registered_input_sha256']}
    eligible=all(any(g['scene']==scene and g['variant']=='grasp_frame' and g['attempts']==3 and g['successes']==3 for g in groups) for scene in ('A1','bottle'))
    summary={'state':read(folder/'state.json'),'rule':registration['handoff_rule'],
        'attempts':len(rows),'successes':sum(r['pass'] for r in rows),'groups':groups,'rows':rows,
        'action_provenance_pass':all(r['action_provenance_pass'] for r in rows),
        'frozen_planning_models_equal':all(len(r['frozen_planning_models'])==2 and all(r['frozen_planning_models'].values()) for r in rows),
        'protected_physics_and_policy_unchanged':all(sha(p)==digest for p,digest in protected.items()),
        'registered_algorithm_and_inputs_unchanged':all(sha(p)==digest for p,digest in sources.items()),
        'executed_source_snapshot_matches_registration':all(sha(folder/'registered_source_snapshot'/Path(p).name)==digest for p,digest in registration['algorithm_sha256'].items()),
        'code_comparison_scope':'Current-code equality is measured when summarizing; the archived source snapshot remains the execution reference after later development.',
        'geometry_protocol_checks':read(folder/'geometry_checks.json') if (folder/'geometry_checks.json').exists() else None,
        'eligible_for_further_validation':eligible,'default_handoff_rule':None,'default_execute_steps':1,
        'scope':('P2 actual teacher target qualification; no VLA inference. ' if registration['experiment']=='teacher' else 'P1 ARX development diagnosis. ')+
            'Visibility checks cover the image frustum, not guaranteed occlusion-free RGB; corridor certificates are planning-only.'}
    write(folder/'summary.json',summary)
    sheet=Image.new('RGB',(640,540),(20,20,20));draw=ImageDraw.Draw(sheet)
    for index,scene in enumerate(('A1','bottle')):
        for col,variant in enumerate(('baseline','grasp_frame')):
            trial=folder/f'{scene}_{variant}_r1'
            actual=trial/'handoff_actual_observation/wrist.png'
            images=[actual] if actual.exists() else sorted((trial/'lilo_observations').glob('*/wrist.png'))[:1]
            if images:
                with Image.open(images[0]) as im:sheet.paste(im.convert('RGB').resize((320,240)),(col*320,index*270+30))
            draw.text((col*320+6,index*270+8),f'{scene} {variant}: actual entry' if images else f'{scene} {variant}: no executed entry',fill=(240,240,240))
    sheet.save(folder/'actual_entry_comparison.jpg',quality=92)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','geometry_protocol_checks')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    parser.add_argument('--encode-videos',action='store_true');args=parser.parse_args()
    summarize(args.folder.resolve(),args.encode_videos)
