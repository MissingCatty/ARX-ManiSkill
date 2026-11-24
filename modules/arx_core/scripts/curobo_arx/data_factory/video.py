"""Encode only deterministic report selections from real wrist recordings."""
import argparse
import json
from pathlib import Path
import subprocess
from .common import read,sha,write


def encode(campaign):
    import imageio_ffmpeg
    campaign=Path(campaign)
    selection=read(campaign/'video_selection.json')
    out=campaign/'videos';out.mkdir(exist_ok=True)
    results=[]
    for item in selection['selected']:
        root=Path(item['episode']);rows=[json.loads(line) for line in (root/'samples.jsonl').read_text().splitlines()]
        if not rows:continue
        dest=out/(item['id']+'.mp4')
        if dest.exists():continue  # Keep prior material; never silently replace.
        intervals=[]
        for request in read(root/'requests.json'):
            if request['label'] not in ('approach','close','lift','hold'):continue
            execution=root/(request['label']+'_execution.json')
            if execution.exists():
                intervals.append((request['requested_sim_time']-.2,
                    request['requested_sim_time']+read(execution)['sim_duration_s']+.001))
        clip=[r for r in rows if any(start-.0001<=r['frame_time']<=end+.0001 for start,end in intervals)]
        if not clip:clip=rows[:150]
        playlist=out/(item['id']+'.ffconcat')
        text=['ffconcat version 1.0']
        for row in clip:
            path=Path(row['rgb_path'])
            if "'" in str(path):raise ValueError('Unsafe ffconcat path')
            text+=['file '+"'"+str(path)+"'",'duration 0.2']
        text+=['file '+"'"+str(clip[-1]['rgb_path'])+"'"]
        playlist.write_text('\n'.join(text)+'\n')
        command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-nostdin','-n','-f','concat','-safe','0','-i',str(playlist),
            '-vf','fps=5','-an','-c:v','libx264','-threads','2','-pix_fmt','yuv420p',str(dest)]
        subprocess.run(command,check=True)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-i',str(dest),'-f','null','-'],check=True,capture_output=True)
        results.append({**item,'video':str(dest),'sha256':sha(dest),'decode_pass':True,'frames':len(clip),
            'timebase':'Original simulation 0.2-second RGB-D sample cadence; production wall speed may differ',
            'cropping':'Real approach/close/lift/hold execution intervals including their pre-command frame; planning waits removed, motion/dwell frames retained',
            'source_frames':len(rows),'removed_wait_frames':len(rows)-len(clip),
            'source':'Actual unannotated ARX wrist RGB; no generated frames, overlays or audio'})
        write(out/(item['id']+'_receipt.json'),results[-1])
    write(out/('encoding_receipt_'+str(__import__('time').time_ns())+'.json'),{'clips':results,'selection':selection})
    lines=['# 自动选取的 ARX 片段','',f"缺少片段数：{selection['missing_slots']}；两种 EGAD 形状覆盖：{selection['irregular_object_requirement_met']}。",'',
        '| 物体 | 距离 mm | 原始结果 | 视频 |','|---|---:|---|---|']
    for item in selection['selected']:
        dest=out/(item['id']+'.mp4')
        link='[播放](videos/'+dest.name+')' if dest.exists() else '尚未生成'
        lines.append(f"| {item['object_id']} | {item['distance']} | {item['state']} | {link} |")
    lines+=['','采用真实腕部 RGB，保留运动、闭合与保持的原采样节奏；删去规划等待，全部原始帧另存。没有叠字、音轨或生成画面。']
    (campaign/'VIDEOS.md').write_text('\n'.join(lines)+'\n')
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',type=Path);a=p.parse_args();print(len(encode(a.campaign)))
