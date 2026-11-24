"""Summarize P0 ARX action provenance and optionally encode actual recordings."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import yaml

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/curobo_arx'))
from lilo_vla.observations import relative_state


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(folder):
    import imageio_ffmpeg
    source=folder/'frames/frames.json'
    if not source.exists():return None
    info=read(source);items=info['frames']
    if info.get('error') or len(items)<2:raise RuntimeError('Incomplete recorded frames: '+str(source))
    deadline=time.monotonic()+30
    while not all((folder/'frames'/item['file']).is_file() for item in items):
        if time.monotonic()>deadline:raise TimeoutError('Recorded frame writes did not finish: '+str(source))
        time.sleep(.2)
    viewport=read(folder/'viewport_initial.json')
    if info['camera_path']!=viewport['camera_path'] or not np.allclose(info['camera_matrix'],viewport['world_matrix'],atol=1e-8):
        raise RuntimeError('Recorded viewport differs from registered historical view')
    video=folder/(folder.name+'.mp4');receipt=folder/'video_receipt.json'
    if video.exists():
        old=read(receipt)
        if old['sha256']!=sha(video):raise RuntimeError('Existing video hash differs')
        return old
    times=np.asarray([f['wall_time'] for f in items]);last=float(np.median(np.diff(times)[-10:]))
    lines=[]
    for index,item in enumerate(items):
        path=(folder/'frames'/item['file']).resolve()
        if not path.is_file():raise FileNotFoundError(path)
        if "'" in str(path):raise ValueError('Unsupported quote in recording path')
        duration=times[index+1]-times[index] if index+1<len(items) else last
        lines.extend(["file '"+str(path)+"'",'duration '+str(max(.001,float(duration)))])
    lines.append("file '"+str(path)+"'")
    manifest=folder/'video_concat.txt';manifest.write_text('\n'.join(lines)+'\n')
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg,'-v','error','-f','concat','-safe','0','-i',str(manifest),
        '-an','-map_metadata','-1','-vf','fps=10','-c:v','libx264','-crf','20',
        '-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True,timeout=180)
    subprocess.run([ffmpeg,'-v','error','-i',str(video),'-f','null','-'],check=True,timeout=180)
    result={'video':str(video),'sha256':sha(video),'source_frames':len(items),
        'timing':'Actual recorded wall-time spacing, encoded at 10fps; no acceleration',
        'wall_duration_seconds':float(times[-1]-times[0]),
        'sim_duration_seconds':items[-1]['sim_time']-items[0]['sim_time'],
        'source':'Actual ARX historical viewport recording; no generated imagery',
        'viewport_verified':True,'full_decode':'PASS'}
    write(receipt,result);return result


def summarize(folder,encode_videos=False):
    registration=read(folder/'registration.json');results=read(folder/'results.json');rows=[]
    if registration.get('experiment','chunks')!='chunks':
        raise ValueError('Use summarize_handoff_rule.py for the P1 handoff experiment')
    for result in results:
        trial=folder/result['label']
        report=read(trial/'grasp_report.json') if (trial/'grasp_report.json').exists() else {}
        sources=report.get('action_sources',[]);errors=[];previous=None
        for source in sources:
            call,index=source['policy_call_index'],source['executed_policy_action_index']
            expected=0 if previous is None or call!=previous[0] else previous[1]+1
            if index!=expected or not 0<=index<result['execute_steps']:
                errors.append({'step':source['step'],'error':'Policy chunk consumed out of order'})
            if previous is not None and call!=previous[0] and previous[1]!=result['execute_steps']-1:
                errors.append({'step':source['step'],'error':'Refreshed prediction before chunk consumed'})
            stored=read(trial/f"local_{source['step']:04d}_prediction.json")
            original=read(trial/source['prediction_file'])
            if stored['actions']!=original['actions'] or stored['policy_call_index']!=call:
                errors.append({'step':source['step'],'error':'Prediction provenance mismatch'})
            if not (trial/f"local_{source['step']:04d}_state.json").exists():
                errors.append({'step':source['step'],'error':'Missing actual pre-action state'})
            previous=(call,index)
        predictions=sorted(trial.glob('policy_*_prediction.json'))
        if len(predictions)!=report.get('local_policy_calls',0):errors.append({'error':'Policy-call accounting mismatch'})
        executed=[source for source in sources if (trial/f"local_{source['step']:04d}_execution.json").exists()]
        if len(executed)!=report.get('local_action_count',0):errors.append({'error':'Executed-action accounting mismatch'})
        max_rise=0.;height_samples=0
        initial=read(trial/'capture_snapshot.json')['objects']['test_object']['world'][2][3] if (trial/'capture_snapshot.json').exists() else None
        for execution in trial.glob('*_execution.json'):
            for row in read(execution).get('records',[]):
                if initial is not None:
                    max_rise=max(max_rise,row['objects']['test_object'][2][3]-initial);height_samples+=1
        first=read(predictions[0]) if predictions else None
        first_state=relative_state(read(trial/'local_0000_state.json')) if (trial/'local_0000_state.json').exists() else None
        model_checks={}
        for name in ('model','model_closed'):
            destination=trial/name;reference=ROOT/'scripts/curobo_arx/static_pick/reference'/name
            if not (destination/'arx.yml').exists():continue
            actual=yaml.safe_load((destination/'arx.yml').read_text())
            original=yaml.safe_load((reference/'arx.yml').read_text())
            spec=actual['robot_cfg']['kinematics'];base=original['robot_cfg']['kinematics']
            spec['urdf_path']=base['urdf_path'];spec['lock_joints']=base['lock_joints']
            spec['cspace']['default_joint_position']=base['cspace']['default_joint_position']
            model_checks[name]=actual==original and sha(destination/'arx.urdf')==sha(reference/'arx.urdf')
        summary={**result,'action_provenance_pass':not errors,'action_provenance_errors':errors,
            'consumed_action_indices':[s['executed_policy_action_index'] for s in sources],
            'executed_action_indices':[s['executed_policy_action_index'] for s in executed],
            'maximum_recorded_object_rise_m':max_rise if height_samples else None,
            'first_state':first_state.tolist() if first_state is not None else None,
            'frozen_planning_model_checks':model_checks,
            'first_normalized_state':first.get('normalized_state') if first else None,
            'unconsumed_actions_on_stop':report.get('unconsumed_chunk_actions'),
            'video':encode(trial) if encode_videos else (read(trial/'video_receipt.json') if (trial/'video_receipt.json').exists() else None)}
        rows.append(summary)
    groups=[]
    for scene in ('bottle','A1'):
        for k in (1,4,8):
            subset=[r for r in rows if r['scene']==scene and r['execute_steps']==k]
            groups.append({'scene':scene,'execute_steps':k,'attempts':len(subset),
                'successes':sum(r['pass'] for r in subset),
                'failures':[{'label':r['label'],'error':r['evaluation'].get('reported_error'),
                    'status':r['status']} for r in subset if not r['pass']]})
    verified=[]
    for k in (4,8):
        by_scene=[g for g in groups if g['execute_steps']==k]
        if all(g['attempts']==3 and g['successes']==3 for g in by_scene):
            baseline=next(g for g in groups if g['scene']=='bottle' and g['execute_steps']==1)
            bottle=next(g for g in by_scene if g['scene']=='bottle')
            if baseline['attempts']==3 and bottle['successes']>=baseline['successes']:verified.append(k)
    protected={**registration['physical_source_sha256'],**registration['policy_artifact_sha256']}
    final={'status':read(folder/'state.json'),'attempts':len(rows),'successes':sum(r['pass'] for r in rows),
        'groups':groups,'rows':rows,'action_provenance_pass':all(r['action_provenance_pass'] for r in rows),
        'protected_physics_and_policy_unchanged':all(sha(path)==digest for path,digest in protected.items()),
        'algorithm_unchanged_since_registration':all(sha(path)==digest for path,digest in registration['algorithm_sha256'].items()),
        'executed_source_snapshot_matches_registration':all(sha(folder/'registered_source_snapshot'/Path(path).name)==digest for path,digest in registration['algorithm_sha256'].items()),
        'code_comparison_scope':'Current-code equality is measured when summarizing; archived source snapshots describe the code actually executed.',
        'frozen_planning_models_equal':all(all(r['frozen_planning_model_checks'].values()) for r in rows),
        'controller_checks':read(folder/'controller_checks.json') if (folder/'controller_checks.json').exists() else None,
        'qualified_after_repeats':verified,'recommended_development_execute_steps':min(verified) if verified else 1,
        'scope':'Only actual ARX bottle/A1 development trials; not a restored/full EGAD49 benchmark'}
    write(folder/'summary.json',final)
    sheet=Image.new('RGB',(960,2*270),(20,20,20));draw=ImageDraw.Draw(sheet)
    for row_index,scene in enumerate(('bottle','A1')):
        for col,k in enumerate((1,4,8)):
            trial=folder/f'{scene}_k{k}_r1'
            images=sorted((trial/'lilo_observations').glob('*/wrist.png'))
            if images:
                with Image.open(images[0]) as im:sheet.paste(im.convert('RGB').resize((320,240)),(col*320,row_index*270+30))
            draw.text((col*320+8,row_index*270+8),f'{scene} K={k}: actual first wrist frame',fill=(240,240,240))
    sheet.save(folder/'first_wrist_comparison.jpg',quality=92)
    print(json.dumps({k:v for k,v in final.items() if k not in ('rows','controller_checks')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    parser.add_argument('--encode-videos',action='store_true');args=parser.parse_args()
    summarize(args.folder.resolve(),args.encode_videos)
