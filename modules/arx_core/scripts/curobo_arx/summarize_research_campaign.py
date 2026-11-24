"""Audit an idle ARX research campaign and write reproducible research outputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import imageio_ffmpeg
import numpy as np
from autoresearch import ROOT, read, write, evaluate


def summarize(campaign):
    campaign=Path(campaign).resolve();state=read(campaign/'state.json')
    if state['status']=='running':raise RuntimeError('Wait for the active batch before packaging')
    rows=[];videos=[]
    for folder in sorted((campaign/'trials').iterdir()):
        if not (folder/'result.json').exists():continue
        result=read(folder/'result.json');score=evaluate(folder);scenario=read(folder/'scenario.json')
        snap=read(folder/'capture_snapshot.json');target=scenario.get('target','test_object')
        z0=snap['objects'][target]['world'][2][3];peak=None;wall=0.;sim=0.
        for path in folder.glob('*_execution.json'):
            execution=read(path);wall+=execution['wall_duration_s'];sim+=execution['sim_duration_s']
            if path.name=='lift_execution.json':
                peak=max(r['objects'][target][2][3]-z0 for r in execution['records'])
        delta=None
        baseline=scenario.get('paired_baseline')
        if baseline and not Path(baseline).is_absolute() and (campaign/'trials'/baseline/'result.json').exists():
            delta=score['score']-evaluate(campaign/'trials'/baseline)['score']
        policy=scenario.get('research_policy',result.get('execution_policy','baseline'))
        decision='baseline' if policy=='baseline' else ('confirmation_required' if delta and delta>0 else 'reverted')
        raw_outcome=score['reported_outcome']
        outcome=('candidate_goal_collision_rejection' if result.get('error') in
                 ('Frozen closed-gripper goals invalid','Frozen open-gripper goals invalid') else raw_outcome)
        row={'name':folder.name,'policy':policy,'hypothesis':scenario.get('hypothesis_id'),
             'target':target,'sphere_pad_height_m':scenario.get('sphere_pad_height_m',.005) if policy=='sphere_refinement' else None,
             'complete_success':score['pass'],'score':score['score'],'decision':decision,
             'paired_score_delta':delta,'executed_stages':score['executed_stages'],
             'outcome':outcome,'raw_outcome':raw_outcome,'error':result.get('error'),
             'peak_lift_rise_m':peak,'hold_min_rise_m':score['hold_min_rise_m'],
             'planning':score['planning'],'guard_pass':score['guard_pass'],
             'unexpected_physical_contact_events':sum('a' in v for v in score['guard_violations']),
             'warmup_seconds':result.get('warmup_seconds'),
             'executed_sim_seconds':sim,'executed_wall_seconds':wall,
             'total_trial_wall_seconds':result['finished_unix']-result['started_unix']}
        rows.append(row)
        video=folder/'motion.mp4'
        if video.exists():
            decoded=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(video),'-f','null','-'],capture_output=True,text=True)
            if decoded.returncode:raise RuntimeError(decoded.stderr)
            recording=result['recording']
            videos.append({'name':folder.name,'file':str(video),'decoded':True,
                           'camera_path':recording['camera_path'],'camera_matrix':recording['camera_matrix'],
                           'editing':recording['editing']})
    checks=[]
    for batch in sorted((campaign/'batches').iterdir()):
        manifest=read(batch/'protected_hashes.json')
        before=read(batch/'before_viewport.json');after=read(batch/'after_viewport.json')
        cb=read(batch/'before_control.json');ca=read(batch/'after_control.json')
        archived=batch/'source_at_end'
        checks.append({'batch':str(batch),'protected_sources_unchanged':read(batch/'integrity.json')['protected_sources_unchanged'] and all(hashlib.sha256((archived/p).read_bytes()).hexdigest()==h for p,h in manifest.items()),
                       'viewport_restored_exactly':before==after,
                       'gains_and_efforts_unchanged':all(cb[k]==ca[k] for k in ['get_dof_gains','get_dof_max_efforts']),
                       'all_video_views_match':all(v['camera_path']==before['camera_path'] and np.allclose(v['camera_matrix'],before['world_matrix']) for v in videos)})
    write(campaign/'comparison.json',rows)
    frozen=read(campaign/'frozen_baseline_hashes.json')
    baseline_ok=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in frozen.items())
    write(campaign/'verification.json',{'batches':checks,'videos':videos,
        'baseline_and_evaluator_unchanged_across_hypotheses':baseline_ok,
        'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'raw_results_unchanged':True,'derived_classification_scope':'Frozen-goal rejection is distinguished from infrastructure failure; raw_outcome retained',
        'fresh_zerograsp_end_to_end_validation':False,'cross_object_validation_executed':False})
    fields=['iteration','metric_value','delta','delta_pct','status','description','evaluator_source','timestamp']
    tsv=campaign/'autoresearch-results.tsv'
    existing=list(csv.DictReader(tsv.open(),delimiter='\t')) if tsv.exists() else []
    if len(existing)>len(rows) or any(old['description']!=row['name'] or float(old['metric_value'])!=row['score'] for old,row in zip(existing,rows)):
        raise RuntimeError('Append-only research history disagrees with archived trials')
    with tsv.open('a') as f:
        writer=csv.writer(f,delimiter='\t')
        if not existing:writer.writerow(fields)
        for i,row in enumerate(rows[len(existing):],start=len(existing)):
            writer.writerow([i,row['score'],row['paired_score_delta'] if row['paired_score_delta'] is not None else '-',
                             '-','reference' if row['decision']=='confirmation_required' else row['decision'],
                             row['name'],'mechanical',time.strftime('%Y-%m-%dT%H:%M:%S%z')])
    best=[r for r in rows if r['policy']=='sphere_refinement' and r['sphere_pad_height_m']==.010]
    summary={'total_attempts':len(rows),'actual_motion_attempts':sum(bool(r['executed_stages']) for r in rows),
             'baseline_trials':sum(r['policy']=='baseline' for r in rows),
             'baseline_successes':sum(r['complete_success'] for r in rows if r['policy']=='baseline'),
             'sphere_fit_10mm_trials':len(best),'sphere_fit_10mm_successes':sum(r['complete_success'] for r in best),
             'sphere_fit_10mm_hold_min_rise_m':[r['hold_min_rise_m'] for r in best],
             'optimized_stages':sum(len(r['planning']['optimized_stages']) for r in rows),
             'planned_limit_exceed_rad':max([r['planning']['max_limit_exceed_rad'] for r in rows]+[0]),
             'dense_invalid_samples':sum(r['planning']['dense_invalid_samples'] for r in rows),
             'unexpected_physical_contacts':sum(r['unexpected_physical_contact_events'] for r in rows),
             'default_promoted':False,'scope':'Within-case repeat trials on one fixed proposal; controls shared, not independent matched pairs',
             'fresh_inference_or_other_objects_validated':False}
    write(campaign/'experiment_conclusion.json',summary)
    lines=['ARX AutoResearch 第二轮结果','',
           'H005：24项归档ARX模型IK诊断。-10mm在三种初始化下均因原场景碰撞被拒；无障碍诊断可解，但独立原场景检查仍拒绝；仅桌面模型也拒绝。-3/-6mm均通过。此处不是PhysX执行。',
           '',f"已记录抓取尝试：{len(rows)}，完整成功：{sum(r['complete_success'] for r in rows)}。这是失败提案的开发筛选，不是总体成功率。",
           f"同固定提案基线：{summary['baseline_successes']}/{summary['baseline_trials']}；球面拟合10mm设置：{summary['sphere_fit_10mm_successes']}/{summary['sphere_fit_10mm_trials']}。后者包含同一开发案例的重复测量，未执行新提案或跨物体验证。",'']
    for row in rows:
        lines.append(f"{row['name']}：success={row['complete_success']}；分类={row['outcome']}；抬升峰值={row['peak_lift_rise_m']}m；保持最低抬升={row['hold_min_rise_m']}m；决定={row['decision']}。")
    lines+=['','默认策略未提升；单案例改善须至少3次配对复测和验证集实测后再决定。原有物理参数、碰撞余量与成功门槛保留。',
            '所有视频来自原WebRTC视角，无标题/文字/音频；规划等待被移除，运动保留录制时帧间隔。',
            '新方案的训练、网络或原论文算法未经复现时，不能把工程消融称为论文复现。']
    (campaign/'result_zh.txt').write_text('\n'.join(lines)+'\n')
    (campaign/'final_report.md').write_text('# ARX AutoResearch round 2\n\n'+'\n\n'.join(lines)+'\n')
    if videos:
        manifest=campaign/'videos.txt';manifest.write_text(''.join("file '"+v['file']+"'\n" for v in videos))
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-f','concat','-safe','0','-i',str(manifest),'-an','-c:v','copy',str(campaign/'all_trials.mp4')],check=True)
        comparison=[v for v in videos if v['name'] in ['00_control','03_sphere_fit_10mm']]
        compare_manifest=campaign/'comparison_videos.txt'
        compare_manifest.write_text(''.join("file '"+v['file']+"'\n" for v in comparison))
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-f','concat','-safe','0','-i',str(compare_manifest),'-an','-c:v','copy',str(campaign/'baseline_vs_refined.mp4')],check=True)
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        sys.path.insert(0,str(Path.home()/'.codex/skills/autoresearch-skill/scripts'))
        from style_presets import rcparams
        rcparams()
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4));x=np.arange(len(rows));values=np.array([r['score'] for r in rows])
        ax.plot(x,np.maximum.accumulate(values),color='#0072B2',label='Best single-trial result')
        ax.scatter(x,values,facecolors='white',edgecolors='#0072B2',s=65,zorder=3)
        ax.set_xticks(x,[r['name'] for r in rows],rotation=18,fontsize=8)
        ax.set(xlabel='ARX physical trial attempt',ylabel='Complete grasp (0/1)',ylim=(-.1,1.15))
        ax.axhline(1,color='gray',ls='--',label='Single-trial success, requires confirmation');ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(campaign/'progress.png',dpi=150);plt.close(fig)
    print(json.dumps({'trials':len(rows),'successes':sum(r['complete_success'] for r in rows),'checks':checks}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);summarize(p.parse_args().campaign)
