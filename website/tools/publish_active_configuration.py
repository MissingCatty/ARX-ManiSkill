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


def read(p):return json.loads(p.read_text())
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def publish():
    config=load_current(verify_dataset=True);profile=load_profile()
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
        text=file.read_text().split('## 当前设置与结果')[0].rstrip()
        file.write_text(text+'\n\n## 当前设置与核验\n\n[当前V2配置、871球加载核验与后续测试状态](/iterations/sam3-graspgenx/versions/V2/)\n')
    print('Published current 871 configuration; retired result links removed.')


if __name__=='__main__':publish()
