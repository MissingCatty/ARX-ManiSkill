"""Read actual seed results, preserve every candidate, issue physical releases."""
import argparse
import copy
from datetime import datetime
from pathlib import Path
import numpy as np
from scripts.curobo_arx.data_factory.common import read,write,sha,verify_files
from .advisor36_samples import require_release


def collect(top):
    top=Path(top).resolve();inventory={r['object_id']:r for r in read(top/'source_inventory.json')}
    records=[];released={};evidence={}
    for directory in sorted(top.glob('seed_preflight*')):
        if not (directory/'registration.json').exists():continue
        reg=read(directory/'registration.json')
        geometry_file=Path(reg.get('canonical_candidate_file',top/'canonical_candidates.json'))
        candidates={r['object_id']:r for r in read(geometry_file)}
        rows=[]
        for p in sorted(directory.glob('groups/*/results.json')):rows.extend(read(p)['trials'])
        for obj in dict.fromkeys(t['object_id'] for t in reg['seed_trials']):
            sub=[r for r in rows if r['object_id']==obj]
            row={'campaign':str(directory),'object_id':obj,'geometry':candidates[obj],
                'scheduled':7,'processed':len(sub),'executed':sum(bool(r.get('executed')) for r in sub),
                'successes':sum(r['success'] for r in sub),'outcomes':[r['outcome'] for r in sub],
                'canonical_released':False,'seed_results':sub}
            records.append(row)
            if len(sub)!=7:continue
            try:require_release(sub,obj)
            except RuntimeError:continue
            proof={str(directory/'registration.json'):sha(directory/'registration.json'),str(geometry_file):sha(geometry_file)}
            center=None
            for result in sub:
                if len(result['attempts'])!=1:raise RuntimeError('Released seed lacks exactly one physical attempt')
                episode=Path(result['attempts'][0]['episode'])
                from .advisor36_audit import physical as evaluate_physical
                physical=evaluate_physical(episode,persist=False)
                if (not physical['pass'] or not physical['bilateral_closure'] or
                    not physical['all_recorded_hold_samples_above_6cm'] or
                    physical['hold_min_rise_m']<=.06 or physical['recorded_hold_duration_s']<3.):
                    raise RuntimeError('Seed success receipt disagrees with frozen physical standard')
                if not read(episode/'seed_pose_verification.json')['pass']:
                    raise RuntimeError('Released seed pose not verified')
                # Independent record union audit uses real occurred actions.
                from .runtime import audit_actions
                audit_actions(episode,read(episode/'requests.json'))
                expected=np.asarray(inventory[obj]['source_object_world'])@np.asarray(candidates[obj]['selected']['Gcanonical_object'])
                if not np.allclose(result['Gcanonical_world'],expected,atol=1e-10,rtol=0):
                    raise RuntimeError('Physical seed tested a different canonical')
                for path in episode.glob('*.json'):proof[str(path)]=sha(path)
                for path in episode.parent.parent.glob('frozen_*.json'):
                    if not read(path)['pass']:raise RuntimeError('Released seed frozen configuration failed')
                    proof[str(path)]=sha(path)
                if result['boundary_case']=='center':center=str(episode/'closure_request.json')
            row['canonical_released']=True
            if obj in released:raise RuntimeError('More than one released canonical for the same object')
            released[obj]={'source':inventory[obj],'geometry':candidates[obj],
                'seed_campaign':str(directory),'seed_results':sub,'physical_evidence':proof,
                'center_closure_request':center,'finite_seven_tests_not_a_continuum_guarantee':True}
            evidence.update(proof)
    return {'candidate_rounds':records,'objects':released,'released_objects':len(released),
        'executed':sum(r['executed'] for r in records),'successful_seed_tests':sum(r['successes'] for r in records),
        'training_started':False,'physical_evidence':evidence}


def write_progress(top, data=None):
    top=Path(top).resolve();data=collect(top) if data is None else data
    brief={k:v for k,v in data.items() if k not in ('objects','physical_evidence','candidate_rounds')}
    brief['object_progress']=[{k:v for k,v in r.items() if k not in ('geometry','seed_results')}
                              for r in data['candidate_rounds']]
    write(top/'seed_progress.json',brief)
    fields=('campaign','object_id','scheduled','processed','executed','successes','canonical_released')
    with (top/'experiments.tsv').open('w') as f:
        f.write('\t'.join(fields)+'\n')
        for r in data['candidate_rounds']:f.write('\t'.join(str(r[k]) for k in fields)+'\n')
    lines=['# 规范抓点与静态观测采集进度','',f'更新时间：{datetime.now().astimezone().isoformat(timespec="seconds")}。',
        '',f'已完成物理尝试 **{data["executed"]}** 次，成功 **{data["successful_seed_tests"]}** 次；',
        f'达到七姿态全部成功的物体 **{data["released_objects"]}/8**。训练未启动，等待用户验收数据后再授权。',
        '', '| 物体 | 候选轮次 | 已完成物理尝试 | 成功 | 七姿态放行 |', '|---|---|---:|---:|---|']
    for r in data['candidate_rounds']:
        lines.append(f'| {r["object_id"]} | {Path(r["campaign"]).name} | {r["executed"]} | {r["successes"]} | {"通过" if r["canonical_released"] else "未通过或尚未完成"} |')
    lines+=['','中心及X/Y/Z正负六个边界姿态分别执行；每个边界同时包含该轴±4毫米和绕同轴±3度。',
        '放行仍要求原始接触、限位、记录检查通过，物体在至少3秒保持期的所有记录采样均抬升超过60毫米。',
        '未通过的候选保留，重新搜索必须仍满足局部表面法线约束。',
        '', '静态候选计划：六个训练物体共6144条，BBQ/Spam独立评估共2048条。候选数不代表实际入库量。',
        '每个物体达到7/7后，先采表内16条接口观测，再采剩余1008条；六个训练物体的接口观测合计96条，均计入6144上限。',
        '原生实例Mask、同期实际末端位姿和动作标签通过检查后，才按完整正负配对入库。',
        '', '七姿态是有限的实际物理检查，连续核心区内每一点的抓取成功率仍需后续独立验证。']
    (top/'DATA_STATUS.md').write_text('\n'.join(lines)+'\n')
    # A standalone plot records observed counts, not planned successes.
    signature=[[r['object_id'],r['campaign'],r['executed'],r['successes']] for r in data['candidate_rounds']]
    marker=top/'progress_plot_data.json'
    if not marker.exists() or read(marker)!=signature:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        labels=[r['object_id'].replace('household_','').replace('hope_','').replace('hot3d_','') for r in data['candidate_rounds']]
        good=[r['successes'] for r in data['candidate_rounds']];bad=[r['executed']-r['successes'] for r in data['candidate_rounds']]
        fig,ax=plt.subplots(figsize=(10,4.8));x=np.arange(len(labels))
        ax.bar(x,good,label='Successful executed seeds');ax.bar(x,bad,bottom=good,label='Failed executed seeds')
        ax.set_xticks(x,labels,rotation=30,ha='right');ax.set_ylabel('Completed physical attempts')
        ax.set_title('Advisor36 canonical preflight — all candidates retained');ax.legend();ax.set_ylim(bottom=0)
        fig.tight_layout();fig.savefig(top/'progress.png',dpi=150);plt.close(fig);write(marker,signature)
    return brief


def register_final(top):
    top=Path(top).resolve();data=collect(top)
    if data['released_objects']!=8:raise RuntimeError('Eight objects must pass before full dataset registration')
    verify_files(data['physical_evidence'])
    for name in ('canonical_registry.json','released_canonical_candidates.json'):
        if (top/name).exists():raise FileExistsError('Canonical registry is immutable')
    write(top/'canonical_registry.json',data)
    write(top/'released_canonical_candidates.json',[v['geometry'] for v in data['objects'].values()])
    return data


def select_next(top, previous):
    """Pick a NEW normal-qualified candidate, never replay a failed pose."""
    top=Path(top).resolve();data=collect(top);inventory=read(top/'source_inventory.json')
    next_rows=[];exhausted=[]
    for source in inventory:
        obj=source['object_id']
        if obj in data['objects']:continue
        history=[r for r in data['candidate_rounds'] if r['object_id']==obj]
        if any(r['processed']!=7 for r in history):raise RuntimeError('Prior candidate has unresolved/unstarted slots')
        original=read(top/'geometry'/(obj+'.json'))
        choices=[c for c in original['candidates'] if c['pass_local_geometry']]
        tried=[np.asarray(r['geometry']['selected']['Gcanonical_object']) for r in history]
        from scipy.spatial.transform import Rotation
        def distinct(c):
            g=np.asarray(c['Gcanonical_object'])
            return all(np.linalg.norm(g[:3,3]-old[:3,3])>=.001 or
                       Rotation.from_matrix(old[:3,:3].T@g[:3,:3]).magnitude()>=np.deg2rad(.5) for old in tried)
        choices=[c for c in choices if distinct(c)]
        neighbor_file=top/'geometry_neighbors'/(obj+'.json')
        if not choices:
            if not neighbor_file.exists():
                from .advisor36_geometry import local_neighbors
                write(neighbor_file,local_neighbors(source,original['selected']))
            choices=[c for c in read(neighbor_file)['candidates'] if c['pass_local_geometry'] and distinct(c)]
        if not choices:exhausted.append(obj);continue
        selected=min(choices,key=lambda c:(c['cost'],max(c['contact_normal_errors_degrees']),c['index']))
        item={k:copy.deepcopy(v) for k,v in original.items() if k!='candidates'}
        item.update(selected=selected,reselected_after=[r['campaign'] for r in history],
            reselection_authority=str(top/'scope_amendment_01.json'))
        if selected['index']>=10000:item['refinement_geometry_file']=str(neighbor_file)
        next_rows.append(item)
    return next_rows,exhausted


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--register-final',action='store_true')
    a=p.parse_args();print(register_final(a.root)['released_objects'] if a.register_final else write_progress(a.root))
