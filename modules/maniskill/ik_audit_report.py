"""Export the completed ARX IK workspace and table-collision investigation."""
from collections import Counter
from pathlib import Path
import argparse
import json
import shutil
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
from PIL import Image

from graspcl_pipeline import read,save,sha
from graspcl_execute import guard


def publish_to_docs(root,text):
    docs=Path(__file__).resolve().parent / 'docs'
    resource=docs / 'resource' / root.name
    resource.mkdir(parents=True,exist_ok=True)
    files=['workspace_overview.png','table_collision_example.png','case_counts.json',
        'blocker_attribution.json','closed_margin_fullworld_countercheck.json','workspace_samples.npz']
    for name in files:shutil.copy2(root / name,resource / name)
    archive=resource / '逐例核查数据.zip'
    temporary=archive.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
        for file in sorted((root / 'cases').glob('*/*/*')):
            if file.name not in ['audit.json','solutions.npz'] or not file.is_file():continue
            compression=zipfile.ZIP_STORED if file.suffix=='.npz' else zipfile.ZIP_DEFLATED
            bundle.write(file,str(file.relative_to(root)),compress_type=compression)
    temporary.replace(archive)
    prefix='resource/'+root.name+'/'
    for name in ['workspace_overview.png','table_collision_example.png']:
        text=text.replace(']('+name+')',']('+prefix+name+')')
    old='逐例结果：`cases/<链路>/<例ID>/audit.json`、`solutions.npz`。汇总：`case_counts.json`；原网格归因：`blocker_attribution.json`；8mm完整场景核验：`closed_margin_fullworld_countercheck.json`；工作空间：`workspace_samples.npz`。'
    new=('逐例结果：[逐例核查数据.zip]('+prefix+'逐例核查数据.zip)（含各例 `audit.json` 和 `solutions.npz`）。'
        '汇总：[case_counts.json]('+prefix+'case_counts.json)；'
        '原网格归因：[blocker_attribution.json]('+prefix+'blocker_attribution.json)；'
        '8mm完整场景核验：[closed_margin_fullworld_countercheck.json]('+prefix+'closed_margin_fullworld_countercheck.json)；'
        '工作空间：[workspace_samples.npz]('+prefix+'workspace_samples.npz)。')
    text=text.replace(old,new)
    document=docs / 'IK可达性与桌面碰撞核查.md'
    document.write_text(text)
    return document


def plot_workspace(root):
    data=np.load(root / 'workspace_samples.npz');points=data['positions'];clear=data['table_clear']
    candidates=[];categories=[]
    for path in sorted((root / 'cases/zerograsp_200').glob('*/solutions.npz')):
        row=np.load(path);poses=row['goals'][:,:3,3]
        ik=row['full_pose_success'].any(-1);table=row['open_table_clear'].any(-1);world=row['open_world_clear'].any(-1)
        category=np.where(~ik,0,np.where(~table,1,np.where(~world,2,3)))
        candidates.append(poses);categories.append(category)
    candidates=np.concatenate(candidates);categories=np.concatenate(categories)
    colors=['#dc8f16','#cf3939','#8b57a8','#16855d']
    labels=['No full-pose IK found (32 seeds)','IK found, all found solutions hit table',
            'Table-clear IK, other obstacles block','World-clear grasp IK found']
    table=read(root / 'workspace.json')['table'];center=np.array(table['center']);dims=np.array(table['dims'])
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    stride=max(1,len(points)//28000)
    for ax,(first,second) in zip(axes,[(0,1),(0,2)]):
        ax.scatter(points[::stride,first],points[::stride,second],s=1,c='#b6bdc6',alpha=.18,label='Joint-limit FK samples')
        chosen=points[clear][::max(1,int(clear.sum())//16000)]
        ax.scatter(chosen[:,first],chosen[:,second],s=1,c='#619ec0',alpha=.18,label='FK samples with table-clear robot')
        lower=center-dims/2
        if second==1:ax.add_patch(Rectangle(lower[:2],dims[0],dims[1],facecolor='#b68b56',edgecolor='#876742',alpha=.15))
        else:ax.add_patch(Rectangle((lower[0],lower[2]),dims[0],dims[2],facecolor='#b68b56',edgecolor='#876742',alpha=.2))
        for index,(color,label) in enumerate(zip(colors,labels)):
            mask=categories==index
            ax.scatter(candidates[mask,first],candidates[mask,second],s=8,c=color,alpha=.45,label=label)
        ax.set_xlabel('World X (m)');ax.set_ylabel('World '+('Y' if second==1 else 'Z')+' (m)')
        ax.set_xlim(-1.72,.51);ax.set_ylim((-1.25,1.25) if second==1 else (-.55,.85))
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
        ax.set_title('Top view' if second==1 else 'Side view; table top Z = 0')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8,markerscale=2)
    fig.suptitle('ARX right-arm FK workspace and saved grasp goals | 6 joints, torso fixed at 0.22 m',fontsize=13)
    fig.subplots_adjust(bottom=.22,top=.87,wspace=.19)
    fig.savefig(root / 'workspace_overview.png',dpi=180);plt.close(fig)


def plot_example(root):
    options=[]
    for path in (root / 'cases/zerograsp_200').glob('*/audit.json'):
        row=read(path)
        if not row['all_found_ik_solutions_table_blocked']:continue
        for e in row['original_replay']['table_examples']:
            if e['stage']=='grasp' and .001<e['penetration_m']<.04:options.append((abs(e['penetration_m']-.008),row,e))
    if not options:
        for path in (root / 'cases/zerograsp_200').glob('*/audit.json'):
            row=read(path);example=row.get('table_example')
            if row['all_found_ik_solutions_table_blocked'] and example:
                options.append((example['penetration_m'],row,example))
    if not options:raise RuntimeError('No table collision example available')
    _,row,example=min(options,key=lambda x:x[0]);trial=Path(row['source_trial']);state=read(trial / 'state.json')
    body=np.asarray(state['links']['body4']);spheres=np.asarray(example['spheres_body4']);spheres[:,:3]=spheres[:,:3] @ body[:3,:3].T+body[:3,3]
    table=state['scene_cuboids']['table'];center=np.array(table['center']);dims=np.array(table['dims'])
    delta=np.abs(spheres[:,:3]-center)-dims/2
    sdf=np.linalg.norm(np.maximum(delta,0),axis=1)+np.minimum(delta.max(1),0)
    hit=(spheres[:,3]>0)&(spheres[:,3]-sdf>0)
    params=read(trial / 'camera/params.json');k=params['intrinsics']
    depth=np.asarray(Image.open(trial / 'camera/depth.png'))*params['depth_scale'];mask=np.asarray(Image.open(trial / 'camera/mask.png'))>0
    v,u=np.indices(depth.shape);cloud=np.stack([(u-k['ppx'])*depth/k['fx'],(v-k['ppy'])*depth/k['fy'],depth],-1)[mask]
    optical=np.asarray(state['camera_world_optical']);cloud=cloud @ optical[:3,:3].T+optical[:3,3]
    target=np.asarray(state['object_center']);near=np.linalg.norm(spheres[:,:3]-target,axis=1)<.22
    fig,axes=plt.subplots(1,2,figsize=(12,5.2))
    for ax,dim,name in [(axes[0],0,'X'),(axes[1],1,'Y')]:
        ax.axhspan(-.5,0,color='#b68b56',alpha=.24);ax.axhline(0,color='#876742',lw=2,label='Actual table surface')
        ax.scatter(cloud[::max(1,len(cloud)//4000),dim],cloud[::max(1,len(cloud)//4000),2],s=1,c='#6d8ab0',alpha=.5,label='Observed target RGB-D')
        for i,sphere in enumerate(spheres):
            if not near[i] or sphere[3]<=0:continue
            ax.add_patch(Circle((sphere[dim],sphere[2]),sphere[3],fill=False,color='#c63e3e' if hit[i] else '#657080',lw=1.2,alpha=.75))
        ax.set_xlim(target[dim]-.22,target[dim]+.22);ax.set_ylim(-.07,.30)
        ax.set_aspect('equal');ax.set_xlabel('World '+name+' (m)');ax.set_ylabel('World Z (m)');ax.grid(alpha=.15)
    axes[0].legend(fontsize=8,loc='upper left')
    fig.suptitle('Original ARX collision spheres: red intersects table\n'+row['id']+f" | returned pose converged; {example['link']}; penetration {example['penetration_m']*1000:.1f} mm",fontsize=10)
    fig.text(.5,.015,'Model diagnosis of a saved candidate; this pose was rejected and was not physically executed.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.87));fig.savefig(root / 'table_collision_example.png',dpi=180);plt.close(fig)
    save(root / 'table_collision_example.json',{'id':row['id'],'example':example,'source_trial':str(trial)})


def report(root):
    summary=read(root / 'summary.json');blockers=read(root / 'blocker_attribution.json')
    margin=read(root / 'closed_margin_fullworld_countercheck.json')['results'];workspace=read(root / 'workspace.json')
    profiles=['zerograsp_200','graspcl_200'];counts={}
    for profile in profiles:
        counter=Counter()
        for file in (root / 'cases' / profile).glob('*/audit.json'):
            row=read(file);counter['cases']+=1;counter['candidates']+=row['candidate_count']
            if row['baseline_physical_attempt']:continue
            first_failed=row['original_replay']['rejected'].get('grasp_ik',0)==row['candidate_count']
            if not first_failed:counter['later_gate_failed']+=1
            elif row['any_full_scene_clear_pure_solution']:counter['solver_search_miss_witness']+=1
            elif not row['any_pure_ik']:counter['full_pose_no_ik_found']+=1
            elif row['all_found_ik_solutions_table_blocked']:counter['all_found_ik_table_blocked']+=1
            else:counter['other_scene_obstacle']+=1
        counts[profile]=dict(counter)
    physical=read(root / 'physical_contacts.json');reg=read(root / 'registration.json')
    base=np.asarray(workspace['base_body4_world']);state=read(Path(reg['warmup_trial']) / 'state.json')
    from scipy.spatial.transform import Rotation
    q1=state['qpos'][state['joint_names'].index('right_joint21')]
    local=np.array([.05994899943470955,-.25,.02449999935925007])+Rotation.from_euler('z',q1).as_matrix()@np.array([.019999999552965164,0,.03999999910593033])
    shoulder=base[:3,:3]@local+base[:3,3]
    table={}
    for profile in profiles:
        a=summary[profile]['all_inferred_cases'];f=summary[profile]['baseline_no_grasp_cases']
        table[profile]={'position_only_cases':a['position_only_ik'],'full_pose_cases':a['any_pure_ik'],
            'all_found_open_solutions_table_blocked':f['all_found_solutions_table_blocked'],
            'table_checks_ignoring_earlier_obstacle_rejections':f['all_found_open_closed_solutions_table_blocked'],
            'additional_existing_8mm_clearance_cases':margin[profile]['case_count'],
            'direct_table_rejections_by_actual_gate_order':f['all_found_solutions_table_blocked']+margin[profile]['case_count'],
            'initial_table_cases':blockers[profile]['initial_table_collision_cases'],
            'recorded_physical_table_contact_cases':physical[profile]['recorded_table_contact_cases']}
    save(root / 'case_counts.json',{'counts':counts,'table':table,'shoulder_pitch_axis_above_table_m':float(shoulder[2]),
        'scope':'Per-case attribution under finite IK search, respecting actual gate order. Does not prove no other mathematical IK branches exist.'})
    z=table['zerograsp_200'];g=table['graspcl_200'];cz=counts['zerograsp_200'];cg=counts['graspcl_200']
    text=f'''# ARX IK 可达空间与桌面碰撞核查

核查两组各 200 例的保存状态、390 个有候选场景和 16,062 个候选。原规划拒绝计数与选中索引重放一致：390/390。全部为离线模型分析，没有执行机器人动作，没有修改基线结果。

## IK 可达空间

| 按场景统计 | ZeroGrasp | ZeroGrasp＋GraspCL |
|---|---:|---:|
| 有候选的场景 | 195 | 195 |
| 至少一个候选位置找到 IK，暂不约束朝向和环境碰撞 | {z['position_only_cases']} | {g['position_only_cases']} |
| 至少一个候选完整抓取位姿找到 IK，暂不计环境碰撞 | {z['full_pose_cases']} | {g['full_pose_cases']} |

这些目标位置总体在当前右臂的运动范围内，原 `grasp_ik` 不能解释为“机械臂够不到”。该标记还包含完整抓取朝向、关节限制、求解器内部碰撞约束和后续碰撞检查。

工作空间图使用原 URDF 和关节限位采样 131,072 组关节角。当前规划只包含右臂6关节，升降轴固定约 {workspace['torso_joint4_fixed_at']:.2f} m，肩部俯仰轴高出桌面约 {shoulder[2]*100:.2f} cm。采样位置包络不保证任意朝向都能达到，也不能用采样空白证明不可达。

![右臂位置采样与实际抓取目标](workspace_overview.png)

## 桌面碰撞例数

| 按实际拒绝顺序归因，避免重复计数 | ZeroGrasp | ZeroGrasp＋GraspCL |
|---|---:|---:|
| 抓取 IK 有解，但本次找到的所有解均碰桌，未通过开爪检查 | {z['all_found_open_solutions_table_blocked']} | {g['all_found_open_solutions_table_blocked']} |
| 通过开爪场景检查，但被原名义闭爪8mm桌面余量额外挡住 | {z['additional_existing_8mm_clearance_cases']} | {g['additional_existing_8mm_clearance_cases']} |
| 桌面相关直接阻挡合计 | {z['direct_table_rejections_by_actual_gate_order']} | {g['direct_table_rejections_by_actual_gate_order']} |
| 初始状态碰撞中，碰桌的例数 | {z['initial_table_cases']} | {g['initial_table_cases']} |
| 已执行轨迹记录中的桌面违规接触 | {z['recorded_physical_table_contact_cases']} | {g['recorded_physical_table_contact_cases']} |

若只做开／闭爪桌面筛查，不考虑更早的其他物体拒绝，两组数字为72／56。其中 GraspCL 的 `B0057_S3_sparse_012_strawberry` 已先被邻物挡住，不能再算作桌面直接导致的整例失败，所以按实际拒绝顺序为72／55。某个解碰桌与整例被桌面挡住不同：两组分别有178／194个场景出现过某个已求得解碰桌，其中许多仍能选其他候选执行。

当前名义闭爪检查保留了桌面 +8 mm 余量。对额外6／4例，使用同一闭爪模型和全部干扰物网格，实际桌面位置下存在可通过的已求得分支，+8mm后这些分支全部被挡住。这不证明改掉余量就能完成后续运动或抓取；本次仅做原因分解，未修改门槛。

模型碰桌主要在两根手指 `right_link27/28` 和腕部／夹爪本体 `right_link26`。球体-桌面计算与原 cuRobo 桌面专用 checker 逐解对照，开爪和名义闭爪判定差异均为0。图中的球体是原模型，不是新增几何。

![一个被拒绝的桌面碰撞候选](table_collision_example.png)

## 未进入抓取执行的失败分解

| 原因，按原检查顺序统计 | ZeroGrasp | ZeroGrasp＋GraspCL |
|---|---:|---:|
| 本次纯 IK 未找到完整姿态解，但位置-only有解 | {cz['full_pose_no_ik_found']} | {cg['full_pose_no_ik_found']} |
| 有IK解，但已求得解全碰桌，开爪检查拒绝 | {cz['all_found_ik_table_blocked']} | {cg['all_found_ik_table_blocked']} |
| 有可通过桌面的解，但全部被干扰物网格挡住 | {cz['other_scene_obstacle']} | {cg['other_scene_obstacle']} |
| 首关通过，后续闭爪、预抓取／抬升IK或路径检查失败 | {cz['later_gate_failed']} | {cg['later_gate_failed']} |
| 初始状态碰干扰物 | 2 | 2 |
| 场景准入失败或遮挡率不符 | 3 | 3 |
| 未进入抓取执行合计 | 143 | 104 |

上表后续检查失败38／29例中包含前述8mm余量的6／4例，不能把两张表重复相加。其他物体阻挡18／18例已逐个隔离网格核验，全部能由干扰物碰撞解释，未见地面碰撞。两个初始碰撞场景分别碰到漂白剂瓶和高尔夫球，均不碰桌或地面。

## 证据范围

- 原 IK 使用32个优化种子、返回前4个解。本次纯IK诊断同为32种子，保存全部返回解；有限搜索未找到解不是数学上的无解证明，“所有已求得解碰桌”也不代表穷尽全部IK分支。
- 两组未发现“原首关全部失败，但本次找到满足原完整场景碰撞约束的解”的反例；390例拒绝计数和候选索引均精确复现，未发现常驻缓存改变拒绝结果。
- 实际执行中的桌面违规接触为0，不能将规划阶段的大量桌面拒绝混报为真实碰桌事故。

逐例结果：`cases/<链路>/<例ID>/audit.json`、`solutions.npz`。汇总：`case_counts.json`；原网格归因：`blocker_attribution.json`；8mm完整场景核验：`closed_margin_fullworld_countercheck.json`；工作空间：`workspace_samples.npz`。
'''
    document=publish_to_docs(root,text)
    guard(reg)
    assert sum(c['candidates'] for c in counts.values())==16062
    for profile in profiles:
        a=summary[profile]['all_inferred_cases']
        assert a['cases']==a['original_replay_same']==195
        assert a['table_oracle_mismatches']==a['closed_table_oracle_mismatches']==0
    save(root / 'verification.json',{'passed':True,'physical_attempts':0,'source_models_unchanged':True,
        'candidate_count':16062,'exactly_replayed_cases':390,'workspace_samples':131072,
        'table_checker_disagreements':0,'finite_search_limitations_disclosed':True})
    (root / 'debug/findings.md').write_text(text)
    (root / 'debug/eliminated.md').write_text('''# 已排除与边界

- 已排除“绝大多数失败由目标位置超出当前右臂范围导致”：390个有候选场景均有位置-only IK正例。
- 原规划精确重放390/390例，未发现缓存改变拒绝计数和候选索引。
- 桌面球体判定与原桌面专用checker逐解一致，差异0；未发现桌面坐标或尺寸变换错误。
- 其他场景约束逐网格验证为干扰物；两个初始碰撞也是干扰物，非桌面或地面。
- 当前有限IK结果不证明16／1例在数学上无解；原8mm余量的额外阻挡经完整场景对照为6／4例，纯桌面筛查中另有一例与更早的邻物拒绝重叠。
''')
    print('REPORT EXPORTED',document)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve()
    plot_workspace(root);plot_example(root);report(root)
