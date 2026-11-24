"""Data-only deliverables from immutable ARX preflight/capture evidence."""
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
from scripts.curobo_arx.data_factory.common import read, write, sha
from .advisor36_export import load_lines
from .advisor36_progress import collect
from .advisor36_samples import LAYERS


NAMES = {'household_hope_milk_carton':'牛奶盒', 'raisin_box':'葡萄干盒',
    'cheez_it':'饼干盒', 'mustard_bottle':'芥末瓶', 'ycb_056_tennis_ball':'网球',
    'household_hot3d_potato_masher':'压泥器', 'household_hot3d_bbq_sauce_bottle':'BBQ瓶',
    'household_ycb_spam_can':'Spam罐头'}
CASES = ('center','X_minus','X_plus','Y_minus','Y_plus','Z_minus','Z_plus')


def distribution(rows):
    """Use unrounded measured residuals, never ideal sampling coordinates."""
    result={'admitted':len(rows), 'axes':{}, 'actual_funnel_z_layers':{},
        'split_counts':dict(Counter(r['split'] for r in rows)),
        'category_counts':dict(Counter(r['category'] for r in rows))}
    if not rows:return result
    d=np.array([r['actual_translation_m'] for r in rows])*1000
    r=np.array([r['actual_rotation_degrees'] for r in rows])
    for key,values in zip(('dx_mm','dy_mm','dz_mm','rx_deg','ry_deg','rz_deg'),np.c_[d,r].T):
        result['axes'][key]={'negative':int(np.sum(values < -1e-8)),
            'positive':int(np.sum(values > 1e-8)), 'zero':int(np.sum(np.abs(values)<=1e-8)),
            'minimum':float(np.min(values)), 'median':float(np.median(values)), 'maximum':float(np.max(values))}
    funnel=d[[row['category']=='funnel' for row in rows],2]
    for lo,hi,_ in LAYERS:
        result['actual_funnel_z_layers'][f'{lo:02d}_{hi:02d}']=int(np.sum((funnel>=lo)&(funnel<hi)))
    result['funnel_z_outside_0_90mm']=int(np.sum((funnel<0)|(funnel>=90)))
    # Check the final accepted pairs independently of the summary counters.
    pairs={}
    for row in rows:pairs.setdefault(row['pair_id'],[]).append(row)
    from .advisor36_export import pair_quality
    checks=[pair_quality(members) for members in pairs.values()]
    if not all(c['pass'] for c in checks):raise RuntimeError('Accepted dataset contains a broken mirror pair')
    result['complete_pairs']=len(checks)
    result['maximum_pair_translation_residual_mm']=float(max(np.max(np.abs(c['actual_pair_translation_residual_m'])) for c in checks)*1000)
    result['maximum_pair_rotation_sum_deg']=float(max(np.linalg.norm(c['actual_pair_rotation_sum_degrees']) for c in checks))
    return result


def report(top, data=None):
    top=Path(top).resolve();data=collect(top) if data is None else data
    inventory=read(top/'source_inventory.json')
    lineage=read(top/'static_lineage.json') if (top/'static_lineage.json').exists() else {}
    rows=[];domains={'train_domain':[], 'unseen_evaluation':[]};files={}
    for source in inventory:
        obj=source['object_id'];history=[r for r in data['candidate_rounds'] if r['object_id']==obj]
        released=data['objects'].get(obj)
        latest=next((r for r in history if r['canonical_released']),history[-1] if history else None)
        geometry=latest['geometry'] if latest else read(top/'geometry'/(obj+'.json'))
        seed=[]
        for trial in latest['seed_results'] if latest else []:
            item={'case':trial['boundary_case'],'executed':trial.get('executed',False),
                'success':trial['success'],'outcome':trial['outcome']}
            if trial.get('attempts'):
                p=Path(trial['attempts'][0]['episode'])/'physical_receipt.json'
                if p.exists():
                    receipt=read(p);files[str(p)]=sha(p)
                    item.update(hold_seconds=receipt.get('recorded_hold_duration_s') or 0,
                        bilateral_closure=receipt.get('bilateral_closure',False))
                    if receipt.get('hold_min_rise_m') is not None:
                        item['minimum_hold_rise_mm']=receipt['hold_min_rise_m']*1000
                pose=p.with_name('seed_pose_verification.json')
                if pose.exists():
                    error=read(pose)['target_error'];files[str(pose)]=sha(pose)
                    item.update(actual_arrival_error_mm=error['translation_m']*1000,
                        actual_arrival_error_deg=float(np.rad2deg(error['rotation_rad'])))
            seed.append(item)
        root=Path(lineage.get(obj,{}).get('current',top/'static_objects'/obj))
        dataset=root/'dataset';admitted=[]
        item={'object_id':obj,'name':NAMES[obj],'source_split':source['source_split'],
            'normal_errors_degrees':geometry['selected']['contact_normal_errors_degrees'],
            'local_geometry_pass':geometry['selected']['pass_local_geometry'],
            'released_7_of_7':bool(released),'canonical_rounds':len(history),
            'executed_all_candidates':sum(r['executed'] for r in history),'current_seed_results':seed,
            'registered_candidates':1024,'dataset_complete':(dataset/'summary.json').exists(),
            'observed':0,'admitted':0,'excluded_after_interruption':0,'dataset_path':None}
        if item['dataset_complete']:
            summary=read(dataset/'summary.json')
            admitted=[r for r in load_lines(dataset/'admission.jsonl') if r['admitted']]
            if len(admitted)!=summary['admitted_observations']:raise RuntimeError('Data summary count disagrees')
            item.update(observed=summary['observed_candidates'],admitted=len(admitted),dataset_path=str(dataset),
                excluded_after_interruption=len(summary.get('unexecuted_after_infrastructure_interruption',[])),
                distribution=distribution(admitted),rejections=summary['objects'][0]['rejections'])
            domain='train_domain' if source['source_split']=='train' else 'unseen_evaluation'
            domains[domain].extend(admitted)
            for p in (dataset/'summary.json',dataset/'admission.jsonl'):files[str(p)]=sha(p)
        rows.append(item)
    for key,items in domains.items():
        for item in items:
            if (key=='unseen_evaluation') != (item['split']=='unseen_evaluation'):
                raise RuntimeError('Unseen-object observations leaked into a training-domain split')
    result={'updated':datetime.now().astimezone().isoformat(timespec='seconds'),
        'complete':all(r['released_7_of_7'] and r['dataset_complete'] for r in rows),
        'objects':rows,'domains':{k:distribution(v) for k,v in domains.items()},
        'training_started':False,'training_steps':0,'files':files}
    write(top/'data_acceptance.json',result)
    lines=['# 规范抓点与全漏斗数据验收', '',
        f'更新时间：{result["updated"]}。状态：'+('数据阶段已完成。' if result['complete'] else '仍在执行，以下为已完成部分。'),
        '', '本轮只生成和检验数据。训练未启动；下一轮训练须等用户确认源抓点与数据集。',
        '', '每个候选检查中心及末端局部X/Y/Z轴正负配对：同轴平移±4毫米、旋转±3度。',
        '放行要求七次实际闭合抬升均成功，保持期所有采样抬升超过60毫米且持续至少3秒，几何和物理标准保持不变。',
        '', '| 物体 | 两端法线偏差（度） | 规范候选轮次 | 当前候选成功/已执行 | 七姿态放行 |',
        '|---|---:|---:|---:|---|']
    for r in rows:
        a,b=r['normal_errors_degrees'];s=r['current_seed_results']
        lines.append(f'| {r["name"]} | {a:.3f} / {b:.3f} | {r["canonical_rounds"]} | {sum(x["success"] for x in s)}/{sum(x["executed"] for x in s)} | {"通过" if r["released_7_of_7"] else "未放行"} |')
    lines+=['', '| 物体 | 姿态 | 结果 | 保持期最低抬升（毫米） | 保持时间（秒） | 实际到位误差（毫米/度） |', '|---|---|---|---:|---:|---:|']
    for r in rows:
        by={s['case']:s for s in r['current_seed_results']}
        for case in CASES:
            s=by.get(case)
            status='尚未执行' if s is None else ('成功' if s['success'] else s['outcome'])
            height=f'{s["minimum_hold_rise_mm"]:.2f}' if s and 'minimum_hold_rise_mm' in s else '—'
            hold=f'{s["hold_seconds"]:.2f}' if s and 'hold_seconds' in s else '—'
            fit=f'{s["actual_arrival_error_mm"]:.3f} / {s["actual_arrival_error_deg"]:.3f}' if s and 'actual_arrival_error_mm' in s else '—'
            lines.append(f'| {r["name"]} | {case} | {status} | {height} | {hold} | {fit} |')
    lines+=['', '以下数量只统计完成独立校验和整对入库的静态观测，不表示成功抓取轨迹数。',
        '六个训练域物体候选总数为6144；BBQ与Spam另有2048条独立评估候选。单侧不可用时整对排除，实际数量单列。',
        '', '| 物体 | 候选数 | 完成数据验收 | 已处理候选 | 合格入库 | 中断后未重放 |', '|---|---:|---|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["name"]} | 1024 | {"是" if r["dataset_complete"] else "否"} | {r["observed"]} | {r["admitted"]} | {r["excluded_after_interruption"]} |')
    for domain,label in [('train_domain','六个训练域物体'),('unseen_evaluation','BBQ与Spam独立评估')]:
        d=result['domains'][domain]
        lines+=['',f'{label}：合格入库 {d["admitted"]} 条。切分数量：{d["split_counts"]}。',
            '', '| 同帧实测残差轴 | 负值数 | 正值数 | 零值数 | 最小值 | 中位数 | 最大值 |', '|---|---:|---:|---:|---:|---:|---:|']
        for axis,a in d['axes'].items():
            lines.append(f'| {axis} | {a["negative"]} | {a["positive"]} | {a["zero"]} | {a["minimum"]:.3f} | {a["median"]:.3f} | {a["maximum"]:.3f} |')
        if d['admitted']:
            lines+=['',f'完整正负配对 {d["complete_pairs"]} 对；最大配对平移残差 {d["maximum_pair_translation_residual_mm"]:.4f} 毫米，旋转向量和最大范数 {d["maximum_pair_rotation_sum_deg"]:.4f} 度。']
        lines+=['', '| 漏斗实际轴向距离（毫米） | 合格观测数 |', '|---|---:|']
        for lo,hi,_ in LAYERS:
            lines.append(f'| {lo}–{hi} | {d["actual_funnel_z_layers"].get(f"{lo:02d}_{hi:02d}",0)} |')
        if d['admitted']:lines+=['',f'实际漏斗轴向落在0–90毫米外的观测：{d["funnel_z_outside_0_90mm"]}条，未按理想采样层重写统计。']
    lines+=['','配对规则：横向X/Y及三个旋转分量符号相反，同对轴向Z相同；轴向漏斗保持0–90毫米接近方向。',
        '图像使用同一渲染帧的原生实例掩码。动作由该帧实际末端与规范抓点变换计算，固定提示词不含修正次数。',
        '物理预检按注册的七个目标发送正常运动命令，保留原ARX到位标准；表中另列闭合前实际到位误差，没有把请求位姿当成零误差的实测位姿。',
        '七姿态预检提供有限点的物理证据；连续核心区各点的成功率仍需独立实测。预算文本与旧输出坍缩的因果关系尚未通过单因素对照隔离。',
        '本轮未进行模型训练、离线预测验收或纯VLM物理闭环试验。']
    (top/'DATA_ACCEPTANCE.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args()
    r=report(a.root);print({'complete':r['complete'],'training_started':False,
        'admitted':{k:v['admitted'] for k,v in r['domains'].items()}})
