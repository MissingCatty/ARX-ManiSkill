"""Separate physical replay/recovery accounting and actual wrist videos."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import json
from .campaign import summarize
from scripts.curobo_arx.data_factory.common import read,write,sha
from training.nearfield_arx.export import state_time


def execution_frames(episode,rows):
    intervals=[]
    for request in read(episode/'requests.json'):
        receipt=episode/(request['label']+'_execution.json')
        if receipt.exists():intervals.append((request['requested_sim_time']-.2,
             request['requested_sim_time']+read(receipt)['sim_duration_s']+.001))
    # Command receipts use the physics clock. Synchronized captures retain the
    # renderer's separate ReferenceTime, whose rounding can omit onset frames.
    frames=[]
    for row in rows:
        clock=state_time(row)
        if any(a-1e-5<=clock<=b+1e-5 for a,b in intervals):frames.append(row)
    return frames


def video(episode,dest):
    import imageio_ffmpeg
    rows=[json.loads(l) for l in (episode/'samples.jsonl').read_text().splitlines()]
    frames=execution_frames(episode,rows)
    if not frames:return None
    if any("'" in r['rgb_path'] for r in frames):raise ValueError('Unsafe ffconcat path')
    playlist=dest.with_suffix('.ffconcat');playlist.parent.mkdir(parents=True,exist_ok=True)
    playlist.write_text('ffconcat version 1.0\n'+''.join("file '"+r['rgb_path']+"'\nduration 0.2\n" for r in frames)+"file '"+frames[-1]['rgb_path']+"'\n")
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-n','-f','concat','-safe','0','-i',str(playlist),
                   '-vf','fps=5','-an','-c:v','libx264','-threads','2','-pix_fmt','yuv420p',str(dest)],check=True)
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-i',str(dest),'-f','null','-'],capture_output=True,check=True)
    result={'path':str(dest),'sha256':sha(dest),'actual_frames':len(frames),'decode_pass':True,
            'event_clock':'Verified physics capture clock for synchronized records; legacy frame clock otherwise',
            'scope':'Recorded near-field wrist frames at5fps; planner idle waits removed. Initial-to-entry motion precedes camera capture and is not included.',
            'recording_kind':'entry_snapshot_only' if len(frames)==1 else 'nearfield_execution_clip'}
    write(dest.with_suffix('.json'),result);return result


def report(campaign,videos=False):
    campaign=Path(campaign).resolve();summary=summarize(campaign);details=[]
    for trial in summary['trials']:
        events=[];physical=None;episode=None
        if trial['attempts']:
            episode=Path(trial['attempts'][0]['episode']);physical=read(episode/'physical_receipt.json')
            if (episode/'interpreter_events.json').exists():events=read(episode/'interpreter_events.json')
            from scripts.curobo_arx.lilo_vla.egad_evaluate import evaluate
            sequence=[]
            occurred=set(physical.get('executed_stages',[]))
            for request in read(episode/'requests.json'):
                label=request['label']
                if label not in occurred:continue
                semantic=label if label in ('pregrasp','approach','close','lift','hold') else 'approach'
                sequence.append((semantic,label))
            chronological=evaluate(episode,sequence=sequence)
            if chronological['pass']!=physical['pass']:
                raise RuntimeError('Chronological original evaluator disagrees with original receipt')
            write(episode.parent.parent/'chronological_evaluation.json',{
                'original_receipt_sha256':sha(episode/'physical_receipt.json'),
                'sequence':sequence,'result':chronological,'original_task_outcome_unchanged':True,
                'reason':'Original canonical JSON sorts stage keys; extra actions are mapped by the real request chronology'})
            physical=chronological
        detail={'trial_id':trial['trial_id'],'object_id':trial['object_id'],'method':trial['method'],
            'physical_success':trial['success'],'admitted':trial['admitted'],'outcome':trial['outcome'],
            'physical':physical,'events':dict(Counter(e['kind'] for e in events)),
            'seconds':trial['finished_unix']-trial['started_unix'],
            'video':video(episode,campaign/'videos'/(trial['trial_id']+'.mp4')) if videos and episode and (episode/'samples.jsonl').exists() else None}
        details.append(detail)
    write(campaign/'derived_results.json',details)
    lines=['# 第一阶段 C/D 真值回放与恢复补采结果','',
           '| 槽位 | 方法 | 物理成功 | 图像/记录入库 | 结果 | 耗时秒 |',
           '|---|---|---:|---:|---|---:|']
    for d in details:
        lines.append(f'| {d["trial_id"]} | {d["method"]} | {int(d["physical_success"])} | {int(d["admitted"])} | {d["outcome"]} | {d["seconds"]:.1f} |')
    lines+=['',f'登记 {summary["scheduled"]}；已结束 {summary["completed"]}；实际运动尝试 {summary["actual_attempts"]}；未结束/未执行 {len(summary["unexecuted"])}。',
        '', 'C/D 使用教师动作或离线G*驱动，用于验证解释器。训练策略试验0次，48次正式对照0次。',
        '恢复的状态生成段单列；监督只从偏位停稳后的新观察开始。原成功、图像、记录标准分别报告。']
    for d in details:
        if d['video']:lines+=['',f'- [{d["trial_id"]} 实际腕部视频]({d["video"]["path"]})']
    (campaign/'ACCEPTANCE.md').write_text('\n'.join(lines)+'\n');return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('campaign');p.add_argument('--videos',action='store_true');a=p.parse_args()
    s=report(a.campaign,a.videos);print(json.dumps({k:v for k,v in s.items() if k!='trials'}))
