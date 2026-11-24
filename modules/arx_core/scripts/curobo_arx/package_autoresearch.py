"""Package a finished bounded ARX research campaign, without touching the sim."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
from autoresearch import read, write, evaluate, ROOT


def package(campaign):
    campaign = Path(campaign).resolve()
    state = read(campaign / 'state.json')
    if state['status'] == 'running':
        raise RuntimeError('Do not package an active campaign')
    protected = read(campaign / 'protected_hashes.json')
    integrity = all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h for p,h in protected.items())
    post_fix = read(campaign/'post_run_fix.json') if (campaign/'post_run_fix.json').exists() else None
    baseline_integrity = integrity
    if post_fix:
        # Distinguish a documented restoration-only fix after the experiment
        # from changes to the frozen evaluator or the baseline during trials.
        runner = 'scripts/curobo_arx/autoresearch.py'
        current = (ROOT/runner).read_text()
        node = next(n for n in ast.parse(current).body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
        evaluator_hash = hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
        baseline_integrity = (
            all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in protected.items() if p!=runner)
            and hashlib.sha256(current.encode()).hexdigest()==post_fix['runner_sha256_after_fix']
            and protected[runner]==post_fix['runner_sha256_during_trials']
            and hashlib.sha256((campaign/'runner_during_trials.py').read_bytes()).hexdigest()==protected[runner]
            and evaluator_hash==post_fix['evaluator_AST_sha256_unchanged'])
    scores, videos, actual_limits = {}, {}, {}
    for folder in sorted((campaign / 'trials').iterdir()):
        if not (folder/'result.json').exists():
            continue
        scores[folder.name] = evaluate(folder)
        snap = read(folder/'capture_snapshot.json')
        limits = np.asarray([snap['joint_limits'][snap['joint_names'].index(n)] for n in snap['right_arm']])
        exceeded = 0.
        for execution in folder.glob('*_execution.json'):
            data = read(execution)
            positions = np.asarray([r['position'] for r in data['records']])
            exceeded = max(exceeded, float(np.maximum(limits[:,0]-positions, positions-limits[:,1]).max(initial=0.)))
        actual_limits[folder.name] = exceeded
        video = folder/'motion.mp4'
        if video.exists():
            result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(video),'-f','null','-'], capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr)
            recording = read(folder/'result.json')['recording']
            videos[folder.name] = {'path':str(video),'decoded':True,'camera_path':recording['camera_path'],
                'camera_matrix':recording['camera_matrix'], 'editing':recording['editing']}
    before = read(campaign/'before_viewport.json')
    after = read(campaign/'after_viewport.json') if (campaign/'after_viewport.json').exists() else None
    controls_before = read(campaign/'before_control.json')
    controls_after = read(campaign/'after_control.json') if (campaign/'after_control.json').exists() else None
    controls_unchanged = bool(controls_after) and all(controls_after[k] == controls_before[k] for k in ['get_dof_gains','get_dof_max_efforts'])
    fixed_input = {}
    base = campaign/'trials/00_control/selected_grasp.json'
    variant = campaign/'trials/01_depth_minus_10mm/selected_grasp.json'
    if base.exists() and variant.exists():
        a,b = read(base),read(variant)
        fixed_input = {'same_index':a['index']==b['index'],'same_network_row':a['network_row']==b['network_row']}
        for key in ['pregrasp_world','grasp_world','lift_world']:
            delta = np.asarray(b[key])-np.asarray(a[key])
            expected = np.zeros((4,4));expected[2,3]=-.010
            fixed_input[key+'_is_only_minus_10mm_z'] = bool(np.allclose(delta,expected,atol=1e-9))
    verification = {'protected_sources_unchanged':integrity,
        'protected_sources_unchanged_during_trials':state['protected_hashes_unchanged'],
        'baseline_and_evaluator_unchanged':baseline_integrity,
        'documented_post_run_runner_fix':post_fix,
        'gains_and_max_efforts_unchanged':controls_unchanged,
        'viewport_restored_exactly':before==after, 'videos':videos, 'fixed_proposal_checks':fixed_input,
        'max_actual_measured_arm_limit_exceed_rad':actual_limits,
        'actual_limit_scope':'Measured ARX arm angles; separate from planned joint bounds. Gripper limits are not arm angles.'}
    write(campaign/'verification.json',verification)
    write(campaign/'evaluation_summary.json',scores)
    hypotheses = read(campaign/'hypotheses.json')
    hypotheses[0].update(status=state['status'],decision=state.get('decision','stopped'),results='evaluation_summary.json')
    write(campaign/'hypotheses.json',hypotheses)
    review = ['本轮已使用通用 AutoResearch skill + ARX 项目 skill 执行一个有预算研究闭环。',
              '上游版本固定；当前无常驻后台研究进程。研究由活跃代理完成，执行器负责串行实测和独立评分。',
              '',f"状态：{state['status']}；决定：{state.get('decision','未完成')}；默认策略没有提升/替换。",'']
    for name,s in scores.items():
        review.append(f"{name}：完整抓取成功={s['pass']}；已执行={s['executed_stages']}；保持最小抬升={s['hold_min_rise_m']} m；原程序分类={s['reported_outcome']}。")
    review += ['', '这是同一个绿色球提案的一次配对筛选，不是多物体或总体成功率。单案例增益也要重复和验证后才能提升默认。',
               f'实验期间保护文件哈希一致={state["protected_hashes_unchanged"]}；基线/评估器保持一致={baseline_integrity}；力度/增益一致={controls_unchanged}；原视角恢复={before==after}。',
               '规划结果、实际执行和稳定保持分别记录；本轮不是 VFAS / GNP / Dex-Net 复现。']
    if post_fix:
        review.append('实验结束后仅修复 runner 的资产元数据恢复逻辑，保证下一次 RGB-D 裁剪使用正确尺寸；原运行源码已留档，原基线与评估器未改。详见 post_run_fix.json。')
    (campaign/'result_zh.txt').write_text('\n'.join(review)+'\n')
    (campaign/'final_report.md').write_text('# ARX AutoResearch pilot\n\n'+'\n\n'.join(review)+'\n\nSee evaluation_summary.json and verification.json for machine-readable evidence.\n')
    if scores:
        import matplotlib
        matplotlib.use('Agg')
        sys.path.insert(0,str(Path.home()/'.codex/skills/autoresearch-skill/scripts'))
        from style_presets import rcparams
        rcparams()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6,4))
        values = [s['score'] for s in scores.values()]
        x = np.arange(len(values))
        ax.plot(x,np.maximum.accumulate(values),color='#0072B2',label='Best screening score')
        ax.scatter(x,values,s=70,facecolors='white',edgecolors='#0072B2',zorder=3)
        ax.axhline(1,color='gray',linestyle='--',label='One-trial success')
        ax.set_xticks(x,list(scores),rotation=12,fontsize=9)
        ax.set(xlabel='Single paired trial (no repeat confirmation)',ylabel='Complete grasp (0 / 1)',ylim=(-.1,1.15))
        ax.legend(fontsize=9)
        fig.tight_layout();fig.savefig(campaign/'progress.png',dpi=150);plt.close(fig)
    print(json.dumps({'state':state,'verification':verification,'scores':scores},allow_nan=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',required=True)
    package(parser.parse_args().campaign)
