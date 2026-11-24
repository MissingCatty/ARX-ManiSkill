"""Publish the canonical configuration; never import retired grasp results."""
import json
from pathlib import Path
import shutil
import sys

SITE=Path(__file__).resolve().parents[1]
WORKSPACE=SITE.parent/'modules'
PROJECT=WORKSPACE/'pipeline'
sys.path.insert(0,str(PROJECT/'experiments/current_configuration'))
from configuration import load_current, load_profile
from arx_project.layout import ROOT, sha256


def read(p):return json.loads(p.read_text())
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def publish_contact_validation(config):
    """Publish a completed component check alongside the existing V2."""
    revision=config.get('contact_policy_revision',{})
    run=Path(revision.get('validation_run',''))
    if not run.is_absolute() or not (run/'acceptance.json').is_file():
        return
    receipt=read(run/'acceptance.json')
    if not receipt['completed'] or receipt['component_motions_completed']!=3:
        raise ValueError('Incomplete fingertip contact validation')
    if receipt['contact_profile']!=read(ROOT/'configs/policies/fingertip_table.json'):
        raise ValueError('Published contact results differ from the active policy')
    content=SITE/'内容';site=read(content/'网站.json');registry=read(content/'experiments.json')
    media=SITE/'resources/media';media.mkdir(parents=True,exist_ok=True)
    files={'contact871-acceptance.json':run/'acceptance.json',
           'contact871-policy.json':ROOT/'configs/policies/fingertip_table.json'}
    tolerance_file=ROOT/'tmp/fingertip_contact/tolerance_review_20261124_01/review.json'
    tolerance_review=read(tolerance_file) if tolerance_file.is_file() else None
    if tolerance_review:
        if tolerance_review['selected_nominal_mesh_tolerance_mm']!=config['table_penetration_allowance_m']*1000:
            raise ValueError('Tolerance review differs from the active policy')
        files['table-tolerance-review.json']=tolerance_file
    names=['teapot','phone','soap']; rows=[]; video_rows=[]; closure_rows=[]
    for label,item in zip(names,receipt['rows']):
        trial=run/'trials'/item['id']; evaluation=item['independent_contact_evaluation']
        files[f'contact871-{label}.png']=trial/'initial_overview.png'
        for view,file in [('overview','execution.mp4'),('head','head.mp4')]:
            files[f'contact871-{label}-{view}.mp4']=trial/file
        depth=item['nominal_fixture_depth_m']*1000
        request=f'留{-depth:.1f}毫米间隙' if depth<0 else f'名义接触{depth:.1f}毫米'
        rows.append(f'| {request} | {"通过" if item["strict_real_mesh_clear"] else "拦截"} | '
            f'{"通过" if item["local_real_mesh_clear"] else "拦截"} | '
            f'{evaluation["allowed_tip_table_events"]} | {evaluation["maximum_allowed_depth_m"]*1000:.3f}毫米 | 接近、闭合命令与退离均完成 |')
        video_rows.append(f'| {request} | [第三人称](/media/contact871-{label}-overview.mp4) · '
                          f'[头部](/media/contact871-{label}-head.mp4) |')
        closed=item['measured_finger_joints_at_close_m']
        final=read(trial/'final_state.json')
        final_fingers=[final['qpos'][final['joint_names'].index(name)]
                       for name in ['right_joint27','right_joint28']]
        closure_rows.append(f'| {request} | {closed["right_joint27"]*1000:.2f} / '
                            f'{closed["right_joint28"]*1000:.2f} | '
                            f'{final_fingers[0]*1000:.2f} / {final_fingers[1]*1000:.2f} |')
    for name,source in files.items():
        target=media/name
        if not target.exists() or sha256(source)!=sha256(target):shutil.copy2(source,target)
        # Build falls back to this real publication copy when the source is absent.
        site['localAssets'][name]='contact871/'+name
    current=next(v for v in site['versions'] if v['id']=='sam3-graspgenx')
    current.update(status='871球指尖接触策略已接入',metric='空桌面接触测试3/3，正式抓取待登记')
    experiment=next(e for e in registry['experiments'] if e['id']=='sam3-graspgenx')
    node=next(n for n in experiment['iterations'] if n['code']=='V2')
    node.update(status='指尖接触策略已接入并完成接口验证',
        resultSummary='空桌面接触测试3/3通过，六段录像记录了空夹和退离。正式物体抓取0次。',
        change='保留当前配置，在最后接近与闭合阶段加入指尖最前端10毫米、桌面上表面最多2毫米的局部接触规则。')
    node['settings']=[row for row in node['settings'] if row[0] not in ['当前进度','指尖与桌面']]
    node['settings'] += [['指尖与桌面','前端10毫米、桌面上表面最多2毫米；抬升只允许退出已有接触。'],
                         ['当前进度','空桌面接触测试3/3完成，正式抓取待登记。']]
    save(content/'网站.json',site);save(content/'experiments.json',registry)
    page=content/'迭代/sam3-graspgenx-v2.md';text=page.read_text()
    text=text.replace('旧球配置及对应抓取、距离和指尖余量实验产物已清理。本轮完成配置核验，实际抓取0次，新批次待登记。',
        '当前871球版本已接入局部指尖接触策略，三项空桌面接触测试已完成。正式200场景抓取待登记。')
    section='''<a id="fingertip-table"></a>

## 指尖与桌面的接触余量

先用871球照常检查。若只有夹指与桌面冲突，再用原始凸碰撞网格核实：球体碰桌、真实网格仍有间隙时可以继续；确实接触时，只放行两根夹指最前端10毫米与桌面上表面最多2毫米的接触。夹指主体、手腕、手臂、桌侧面和其他物体继续检查。

到预抓取位姿仍用cuRobo规划。最后接近使用密集检查的局部轨迹；闭合后重新读取夹指状态与实际接触，再检查抬升。抬升时接触深度不能增加。机器人球参数、几何、质量、摩擦和1毫米物理接触偏移保持当前配置。

### 空桌面接触测试（3项）

三个动作都对准预设的空桌面测试点。茶壶、手机和香皂留在原位作为背景。测试依次下发接近、空夹和退离命令，检查指尖接触桌面后能否抬离。正式物体抓取0次。

| 验证动作 | 零余量网格检查 | 2毫米局部规则 | 放行的接触记录数 | 实际最大接触深度 | 执行结果 |
| --- | --- | --- | ---: | ---: | --- |
'''+ '\n'.join(rows)+'''

名义接触深度是规划终点要求，后两项实际接触深度约为0.024和0.018毫米。

### 录像与闭合时序

视频约11.05至14.08秒为闭合阶段，14.08秒后开始抬升。下表是两根手指各自的关节位移，单位为毫米，每根44表示全开，接近0表示合拢。

| 空桌面测试动作 | 闭合阶段结束，指27 / 指28 | 抬升阶段结束，指27 / 指28 |
| --- | ---: | ---: |
'''+ '\n'.join(closure_rows)+'''

茶壶视频在闭合阶段就已合拢。手机和香皂视频在闭合阶段末仍接近全开，抬升过程中继续合拢；香皂视频在抬升后合拢到位，手机视频末尾仍留有开度。此前将触桌和闭合延迟直接归因为“桌面阻碍闭合”，现修正为上述逐阶段记录，阻力来源尚待对照验证。

三项的禁止接触均为0，原始接触点与法线的独立复核全部通过。离线检查共87项通过，六段视频全部完成解码检查。

| 空桌面测试动作 | 空夹测试录像 |
| --- | --- |
'''+ '\n'.join(video_rows)+'''

[接触规则](/media/contact871-policy.json) · [本轮验证记录](/media/contact871-acceptance.json)

## 夹爪进场开度

当前 GraspGenX 输出抓姿、评分和生成分支。88毫米是 ARX 的全开宽度，由适配层写入候选记录。到预抓取点和最后接近时，两个指关节都使用44毫米的全开命令。

后续可按每个抓姿附近的目标点云估算所需开度，在安全位置调好夹爪，并用同一个开度做接近规划与执行。
'''
    if tolerance_review:
        section+='''
## 容差为什么取2毫米

旧硬碰撞检查采用0毫米容差，碰撞球进入桌面就会拒绝该解。保留的历史汇总中，53个原位规划失败场景通过真实夹指网格零穿入复核，确认恢复8例终点；增加2毫米指尖余量后为9例。额外恢复的香皂候选真实指尖穿入约1.818毫米，向上取0.5毫米档得到2毫米。

本轮继续采用真实网格最多2毫米、指尖前端10毫米和桌面上表面的局部规则。旧批次逐例深度记录已在清理时删除，以上是保留下来的汇总。当前871球配置的全场景深度分布尚待补充。

[容差选择与来源记录](/media/table-tolerance-review.json)
'''
    page.write_text(text+'\n'+section)
    doc=content/'实验配置/夹爪碰撞球精细化.md'
    doc.write_text(doc.read_text()+'''\n### 当前指尖接触规则

871个球的球心和半径保持不变。最后接近与闭合阶段，若碰撞只涉及指尖和桌面，使用真实凸网格复核，并允许夹指最前端10毫米与桌面上表面最多2毫米的局部接触。抬升只允许退出已有接触；其他部位和其他物体继续按原检查。

[查看V2接入方法与验证结果](/iterations/sam3-graspgenx/versions/V2/#fingertip-table)
''')


def publish():
    config=load_current(verify_dataset=True);profile=load_profile()
    if config.get('version') == 'V3':
        raise RuntimeError('V3使用website/tools/publish_dual_arm_v3.py发布；此V2工具不覆盖V3页面与登记。')
    content=SITE/'内容';assets=content/'资源';site=read(content/'网站.json')
    records=PROJECT/'experiments/current_configuration/records'
    validation=records/'validation_20261124_01'
    simulation=read(validation/'simulation.json');native=read(validation/'native.json')
    native7=read(validation/'native7.json')
    assert simulation['passed'] and native['passed'] and native7['passed']
    receipt=read(records/'cleanup_20261124_01/completion.json')
    assert receipt['completed']
    clean=assets/'active-configuration';clean.mkdir(exist_ok=True)
    save(clean/'configuration.json',config)
    save(clean/'verification.json',{'simulation':simulation,'native':native,'native7':native7,'cleanup':receipt})
    shutil.copy2(validation/'head.jpg',clean/'head.jpg')
    save(assets/'精细碰撞球参数.json',profile)
    save(assets/'碰撞球配置状态核对.json',{'active':True,'profile_id':profile['profile_id'],
        'total_spheres':871,'models':config['planning_models'],
        'parameters_sha256':profile['sphere_parameter_sha256'],'native':native})
    retired=['sam3-v1-','full-distance-','edge-distance-','reach-distance-','bimanual-','fingertip-',
             'lift-height-','ik-workspace-','edge-banana-','edge-boxed-food-','edge-donut-']
    for group in ['localAssets','benchmarkAssets']:
        site[group]={k:v for k,v in site[group].items() if not any(k.startswith(x) for x in retired)
            and k not in ['gripper-spheres-comparison.png','gripper-spheres-paired-checks.json',
                         'gripper-spheres-directional-fit.json','gripper-spheres-geometry.json','action-tcp-comparison.png']}
    site['localAssets'].update({'active-configuration.json':'active-configuration/configuration.json',
        'active-configuration-verification.json':'active-configuration/verification.json',
        'active-head.jpg':'active-configuration/head.jpg','gripper-spheres-current.jpg':'active-configuration/spheres.jpg',
        'action-tcp-current.jpg':'active-configuration/tcp.jpg'})
    entry=next(v for v in site['versions'] if v['id']=='sam3-graspgenx')
    entry.update(date='2026-11-24',status='871球配置已统一并核验',metric='当前配置已核验，抓取测试待登记',
        summary='头部SAM3、RaySt3R、GraspGenX与cuRobo；统一使用当前200场景、TCP43和871球碰撞模型。')
    for record in site['configurationRecords']:
        record['date']='2026-11-24'
        if record['id']=='gripper-configuration':
            record.update(status='TCP43与871球已启用',summary='查看当前动作参考点、精细碰撞球与实际加载核验。')
            for section in record['sections']:
                section.update(date='2026-11-24',status='已启用并核验')
                section['summary']='动作TCP位于EEF前方43毫米。' if section['id']=='action-tcp' else '开爪、闭爪与七轴参考配置统一871球。'
        elif record['id']=='reachability-check':
            record.update(status='前方地图与当前高度',summary='查看前方可达性地图，以及当前45.2厘米升降配置。')
            for section in record['sections']:
                if section['id']=='ik-workspace':section.update(status='采用当前模型',summary='区分运动学可达、场景无碰撞与完整路径通过。')
                if section['id']=='lift-height':section.update(status='当前固定45.2厘米',summary='右臂六轴规划，桌面高50厘米，升降读数固定45.2厘米。')
    save(content/'网站.json',site)
    catalog=read(content/'experiments.json')
    experiment=next(e for e in catalog['experiments'] if e['id']=='sam3-graspgenx')
    experiment['current']='V2'
    for node in experiment['iterations']:
        node['batches']=[]
        if node['code']=='V1':
            node.update(archived=True,archiveReason='按用户要求清理旧碰撞球配置及对应测试产物。',
                status='旧配置产物已清理',resultSummary='旧配置的抓取结果、距离分析与录像已清理。',
                scenes='保留当前独立采样200场景，旧测试批次已删除。',
                settings=[['状态','旧配置与相应产物已清理，当前设置见V2。']])
        else:
            node.update(status='最新配置已统一并核验',resultSummary='871球实际加载、四种夹爪状态与当前场景预检通过；实际抓取0次。',
                change='统一200场景、D405头部向下45°、TCP43、升降45.2厘米、871碰撞球与1毫米物理接触偏移。',
                pipeline='头部D405 → SAM3 → RaySt3R → GraspGenX → cuRobo（871球）→ ARX抓取',
                scenes='40类、80型号、五种布局各40例，合计200例。',
                settings=[['场景','独立随机采样200例，保持冻结摆放。'],['感知','头部D405，640×480，向下45°。'],
                    ['机器人','升降固定45.2厘米，右臂六轴；动作TCP为43毫米。'],['碰撞模型','八个右臂连杆共871球，开合状态使用实测手指关节。'],
                    ['物理','120Hz物理、40Hz控制，接触偏移1毫米。'],['成功标准','目标抬升3厘米，实际至少抬高1厘米并保持3秒，满足双指接触等原检查。'],
                    ['当前进度','配置核验完成，下一批实际抓取待登记。']])
    save(content/'experiments.json',catalog)
    overview='''## 当前链路

头部D405 RGB-D → SAM3目标掩码 → RaySt3R点云补全 → GraspGenX抓取候选与评分 → cuRobo六轴规划 → ARX接近、闭合、抬升与保持。

当前设置和核验结果记录在[V2](/iterations/sam3-graspgenx/versions/V2/)。871球碰撞模型已启用，当前200场景、TCP43、升降45.2厘米和头部向下45°已经统一。

## 从观测到抓取

[[流程卡片]]

## 相关研究

[SAM3](/papers/sam3/) · [RaySt3R](/papers/rayst3r/) · [GraspGenX](/papers/graspgenx/) · [cuRobo](/papers/curobo/)
'''
    (content/'迭代/SAM3点云GraspGenX主线.md').write_text(overview)
    (content/'迭代/sam3-graspgenx-v1.md').write_text('旧球配置对应的200例抓取结果、距离分析和录像已按要求删除。当前设置及核验见[V2](/iterations/sam3-graspgenx/versions/V2/)。\n\n## 当前链路\n\n[[流程卡片]]\n')
    table='''| 配置项 | 当前设置 |
| --- | --- |
| 场景 | 200例，40类、80型号、五种布局各40例 |
| 主相机 | RealSense D405，640×480，向下45° |
| 动作TCP | EEF前方43毫米 |
| 升降与手臂 | 升降固定45.2厘米，右臂六轴规划 |
| 碰撞球 | 开爪、闭爪与七轴参考模型均为871球 |
| 进场开度 | 双指全开，指腹间距约88毫米 |
| 物理 | 接触偏移1毫米；120Hz物理、40Hz控制 |
| 抓取标准 | 目标抬升3厘米，实际抬高至少1厘米并保持3秒，双指接触与动作到位达标 |
'''
    results=f'''## 当前配置

{table}
## 从观测到抓取

[[流程卡片]]

## 实际加载核验

cuRobo完整规划器已经加载871球。另在张开、半闭、闭合和不对称开合四种状态下核对原生球参数，均与独立机器人正运动学计算一致。当前场景启动检查确认相机、升降高度、物理频率及接触偏移一致；200场景的2406个冻结文件全部通过校验。

旧球配置及对应抓取、距离和指尖余量实验产物已清理。本轮完成配置核验，实际抓取0次，新批次待登记。

## 当前主视角

[![当前头部相机主视角](/media/active-head.jpg)](/media/active-head.jpg)

[当前配置清单](/media/active-configuration.json) · [实际加载与清理记录](/media/active-configuration-verification.json) · [夹爪配置](/configuration/gripper-configuration/)
'''
    (content/'迭代/sam3-graspgenx-v2.md').write_text(results)
    docs=content/'实验配置'
    counts=[('右臂连杆',f'right_link{i}',n) for i,n in zip(range(21,29),[26,115,97,36,24,89,258,226])]
    sphere_table='\n'.join(f'| {label} | {link} | {n} |' for label,link,n in counts)
    (docs/'夹爪碰撞球精细化.md').write_text(f'''### 当前使用871球

开爪、闭爪和七轴参考模型已统一使用871球。抓取时读取实际手指关节位置，球心随各连杆运动；闭合时继续使用同一套球形近似。

[![原始凸碰撞体与当前871球模型，同姿态同尺度](/media/gripper-spheres-current.jpg)](/media/gripper-spheres-current.jpg)

| 部件 | 连杆 | 当前球数 |
| --- | --- | ---: |
{sphere_table}
| 合计 | 八个连杆 | **871** |

### 如何确认加载正确

加载器逐项核对所有球心和半径。原生cuRobo在张开、半闭、闭合和不对称开合四种状态下均加载871个有效球，位置与独立URDF正运动学计算一致。旧球模型、球心或半径被改动、额外增加球膨胀时均会停止加载。

拟合使用原始SAPIEN/PhysX凸碰撞体，机器人实际碰撞网格和尺寸保持原设置。

[当前871球参数](/media/gripper-spheres-parameters.json) · [加载核验记录](/media/gripper-spheres-configuration-check.json)
''')
    (docs/'夹爪TCP位置调整.md').write_text('''### 当前动作TCP

动作TCP固定在right_eef_link前方43毫米，轴向距原始指尖平面约1.97毫米。动作X对应EEF的Y，动作Y对应EEF的-X，动作Z对应EEF的Z。

[![当前TCP43与原始夹爪凸碰撞体](/media/action-tcp-current.jpg)](/media/action-tcp-current.jpg)

GraspGenX的虚拟原点在EEF后方100毫米，输入条件和生成结果成对换算到实际EEF；动作TCP仍为43毫米。

[当前配置清单](/media/active-configuration.json)
''')
    (docs/'升降高度与可达性比较.md').write_text('''### 当前高度

当前SAM3＋RaySt3R＋GraspGenX主线将升降关节固定在45.2厘米，桌面高50厘米，右臂六个关节参与抓取规划。动作TCP为43毫米，规划采用871球碰撞模型。

| 项目 | 当前设置 |
| --- | --- |
| 升降关节读数 | 45.2厘米，执行中固定 |
| 桌面高度 | 50厘米 |
| 右臂规划关节 | 六轴 |
| 碰撞模型 | 871球 |

[当前配置与核验](/iterations/sam3-graspgenx/versions/V2/)
''')
    (docs/'末端可达范围核查.md').write_text('''### 当前检查方式

当前使用TCP43和871球右臂模型。先检查抓取位置与朝向能否找到关节限位内的解，再检查桌面和邻近物体碰撞，最后规划完整接近路径。三项结果分别记录，便于区分距离、朝向、碰桌和路径问题。

前方地图保存关节限位内的运动学见证，查询范围为机器人面朝的180°。地图使用body4局部坐标，随实际升降高度转换到世界坐标。

[![当前前方可达性地图](/media/reachability-front-overview.png)](/media/reachability-front-overview.png)

[查看前方地图](/configuration/reachability-check/#reachability-map) · [当前运行配置](/media/active-configuration.json)
''')
    map_file=docs/'朝向可达性地图.md'
    map_text=map_file.read_text().split('### 旧图与历史结果')[0]
    map_file.write_text(map_text.replace('旧轮 20°、30°结果作为历史探索保留。',''))
    workflow_file=content/'流程/SAM3点云GraspGenX流程.json'
    workflow=read(workflow_file);workflow['note']='当前统一使用头部向下45°、TCP43、固定45.2厘米和871球模型，配置核验见V2。'
    for step in workflow['steps']:
        if step['id']=='planning':step['summary']='TCP43、871球、六轴，检查IK、场景碰撞和完整轨迹。'
        if step['id']=='execution':step['status']='新配置待抓取登记'
    save(workflow_file,workflow)
    for step in workflow['steps']:
        file=content/'流程'/workflow['directory']/step['file']
        text=file.read_text().split('## 当前设置与结果',1)[0].split('## 当前设置与核验',1)[0].rstrip()
        file.write_text(text+'\n\n## 当前设置与核验\n\n[当前V2配置、871球加载核验与后续测试状态](/iterations/sam3-graspgenx/versions/V2/)\n')
    publish_contact_validation(config)
    print('Published current 871 configuration and available contact validation.')


if __name__=='__main__':publish()
