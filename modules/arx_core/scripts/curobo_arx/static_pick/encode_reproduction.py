"""Encode the original viewport plus an explicitly documented action excerpt."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);args=parser.parse_args();folder=args.folder.resolve()
    # Run the historical encoder, changing only its isolated reference-view IO.
    source=ROOT/'scripts/curobo_arx/dynamic_pick/encode_walltime.py'
    text=source.read_text().replace('folder.parent/"viewport.json"','folder/"historical_reference/viewport_initial.json"')
    adapted=folder/'environment_evidence/adapted_encode_walltime.py';adapted.write_text(text)
    subprocess.run([sys.executable,str(adapted),str(folder)],check=True)
    full=folder/'bottle_grasp_no_obstacle_full.mp4'
    (folder/'dynamic_obstacle_grasp.mp4').rename(full)
    report=json.loads((folder/'video_report.json').read_text());report['video']=str(full)
    (folder/'video_report.json').write_text(json.dumps(report,indent=2)+'\n')
    frames=folder/'frames';info=json.loads((frames/'frames.json').read_text());items=info['frames']
    intervals=[]
    report_path=folder/'grasp_report.json'
    task_report=json.loads(report_path.read_text()) if report_path.exists() else {}
    labels=list(task_report.get('stage_timing',{})) if task_report.get('controller') in ('lilo_vla','easg_cloud') else ('pregrasp','approach','close','lift','hold')
    for name in labels:
        path=folder/(name+'_execution.json')
        if not path.exists():continue
        execution=json.loads(path.read_text());end=execution['final']['sim_time'];start=end-execution['sim_duration_s']
        indices=[i for i,f in enumerate(items) if start-.4<=f['sim_time']<=end+.4]
        if indices:intervals.append((indices[0],indices[-1],name))
    assert intervals,'No actual execution interval exists for an action video'
    merged=[]
    for start,end,name in intervals:
        if merged and start<=merged[-1][1]+1:
            merged[-1][1]=max(end,merged[-1][1]);merged[-1][2].append(name)
        else:merged.append([start,end,[name]])
    lines=[];segments=[];picked=[]
    for start,end,names in merged:
        subset=items[start:end+1];times=[f['wall_time'] for f in subset]
        dtlast=float(np.median(np.diff(times)[-10:])) if len(times)>1 else .2
        durations=[]
        for i,item in enumerate(subset):
            delta=times[i+1]-times[i] if i+1<len(times) else dtlast
            durations.append(max(.001,delta));path=frames/item['file'];lines.extend(["file '"+str(path)+"'",'duration '+str(durations[-1])]);picked.append(path)
        segments.append({'stages':names,'first_source_index':start,'last_source_index':end,'source_sim_time':[subset[0]['sim_time'],subset[-1]['sim_time']],'duration_seconds':sum(durations)})
    lines.append("file '"+str(picked[-1])+"'")
    manifest=folder/'action_concat.txt';manifest.write_text('\n'.join(lines)+'\n')
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();action=folder/'bottle_grasp_no_obstacle_action.mp4';assert not action.exists()
    subprocess.run([ffmpeg,'-v','error','-f','concat','-safe','0','-i',str(manifest),'-an','-map_metadata','-1','-vf','fps=5','-c:v','libx264','-crf','20','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(action)],check=True,timeout=180)
    subprocess.run([ffmpeg,'-v','error','-i',str(action),'-f','null','-'],check=True,timeout=180)
    # Six real recorded frames make a reviewable still sheet.
    sheet=Image.new('RGB',(1280,1080))
    for i,index in enumerate(np.linspace(0,len(picked)-1,6,dtype=int)):
        with Image.open(picked[index]) as image:sheet.paste(image.convert('RGB').resize((640,360)),((i%2)*640,(i//2)*360))
    sheet.save(folder/'action_contact_sheet.jpg',quality=90)
    review={'video':str(action),'source':'Actual recorded original viewport, no generated frames','text_added':False,'audio_added':False,'editing':'Only planning/compilation waiting outside recorded execution intervals was omitted; motion remains at recorded wall-time speed; full recording retained','segments':segments,'expected_duration_seconds':sum(s['duration_seconds'] for s in segments),'full_recording':str(full),'full_decode':'PASS'}
    (folder/'action_video_report.json').write_text(json.dumps(review,indent=2)+'\n');print(json.dumps(review,indent=2))


if __name__=='__main__':main()
