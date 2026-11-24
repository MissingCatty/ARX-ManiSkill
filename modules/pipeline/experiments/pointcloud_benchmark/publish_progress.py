"""Publish saved reconstruction evidence to the existing local progress site."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess

from prepare_scenes import RUN, PROJECT

SITE = PROJECT.parent/'claro/进度网站'
RESOURCES = PROJECT/'docs/pointcloud_benchmark/resources'


def comparison_table():
    categories = ['杯子', '水瓶', '罐头', '食品盒', '牛奶包装', '番茄酱瓶', '碗', '苹果', '勺子', '剪刀']
    input_picture = '[![50个场景输入](resources/inputs_contact_sheet_thumb.jpg)](resources/inputs_contact_sheet.jpg)'
    comparisons = []
    for index, category in zip(range(1, 51, 5), categories):
        case = f'C{index:03d}'
        comparisons.append(f'[![{category}重建对照 {case}](resources/comparison_{case.lower()}_thumb.jpg)](resources/comparison_{case.lower()}.jpg)')
    return '\n'.join([
        '| 50个场景输入 | 重建对照（每类一款） |',
        '| --- | --- |',
        f'| {input_picture} | '+''.join(comparisons)+' |',
    ])


def website_article(report):
    """Keep the website presentation independent of the full research report."""
    article = report.split('\n', 1)[1]
    omitted_sections = {'生成、定标与资源', '反投影参照与冷启动', '当前结论与下一步', '适用范围'}
    omitted_paragraphs = ('50个视觉资产中，', '共享SAM3掩码50/50有效，',
                          'CD为双向、非平方最近点距离的均值，', '类别表为F-score@5mm，',
                          '80%和50%仅帮助阅读分布，', '正式登记250项，', '十类常见物体，')
    sections = []
    for section in re.split(r'(?m)(?=^## )', article):
        title = section.split('\n', 1)[0].removeprefix('## ').strip()
        if title in omitted_sections:
            continue
        if title == '固定案例图':
            sections.append('''## 场景输入与重建对照

左边是50个场景的输入画面，右边每行是一张独立的重建对照图。两侧等高，点击图片可以单独查看对应的原图。

'''+comparison_table()+'''

## 指标分布

图中从左到右依次比较三维形状的误差、每次生成所需的时间，以及形状匹配得分。误差和耗时越低越好，匹配得分越高越好。

| 重建误差分布 | 重建耗时分布 | 形状匹配得分 |
| --- | --- | --- |
| [![重建误差分布](resources/reconstruction_error_thumb.jpg)](resources/reconstruction_error.jpg) | [![重建耗时分布](resources/reconstruction_time_thumb.jpg)](resources/reconstruction_time.jpg) | [![形状匹配得分](resources/shape_match_score_thumb.jpg)](resources/shape_match_score.jpg) |

## 没有生成三维模型的案例

下面这7个场景中，Mini-Turbo没有生成可用的三维模型。

[![全部空网格案例](resources/empty_outputs.jpg)](resources/empty_outputs.jpg)

''')
            continue
        paragraphs = [p for p in section.split('\n\n') if not p.strip().startswith(omitted_paragraphs)]
        paragraphs = ['C017和C020是两个形状、大小和摆放位置相同的食品盒，包装图案不同。下面比较每种方法对这两个盒子的重建结果。'
                      if p.strip().startswith('C017与C020的可视几何') else p for p in paragraphs]
        sections.append('\n\n'.join(paragraphs))
    article = '\n\n'.join(section.strip() for section in sections if section.strip())+'\n'
    replacements = {
        '## 完整性与几何表现': '## 五种方法的重建结果',
        '## 各类别款式表现': '## 不同物体的重建效果',
        '## 不同款式的波动': '## 哪些物体重建得好，哪些比较差',
        '## 相同形状、不同外观的已有案例': '## 形状一样、包装不同的两个食品盒',
        'C017 CD / F-score@5mm': '食品盒C017：误差／匹配得分',
        'C020 CD / F-score@5mm': '食品盒C020：误差／匹配得分',
        '有效输出/50': '生成可用模型的次数/50',
        '模型输出失败': '没有生成模型的次数',
        '| 故障 |': '| 程序出错 |',
        '| 未执行 |': '| 尚未处理 |',
        '米制CD中位数（mm）': '形状误差中位数（毫米，越小越好）',
        'F-score@5mm': '平均形状匹配得分（越高越好）',
        '统计分母': '参与计分的次数',
        'F-score≥80%': '得分达到80%的物体数',
        'F-score<50%': '得分低于50%的物体数',
        '最差有效输出（CD）': '误差最大的物体及其误差',
        '输出失败原因': '没有生成模型的原因',
        'empty_native_mesh×7': '7次没有生成三维表面',
        'ZeroGrasp（仅重建）': 'ZeroGrasp',
    }
    for old, new in replacements.items():
        article = article.replace(old, new)
    return re.sub(r'resources/([a-z0-9_]+\.jpg)', lambda m:'/media/pointcloud-'+m[1].replace('_','-'), article)


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n'); temp.replace(path)


def literature():
    path = SITE/'内容/论文调研/论文目录.json'
    catalog = json.loads(path.read_text())
    entries = [
        {'id':'rayst3r','name':'RaySt3R','title':'RaySt3R: Predicting Novel Depth Maps for Zero-Shot Object Completion',
         'authors':'Bardienus P. Duisterhof、Jan Oberst、Bowen Wen、Stan Birchfield、Deva Ramanan、Jeffrey Ichnowski',
         'year':2025,'venue':'NeurIPS 2025','source':'https://rayst3r.github.io/',
         'fulltext':'https://arxiv.org/html/2506.05285','file':'RaySt3R.md',
         'summary':'根据单帧目标RGB-D预测多个新视角的深度、掩码和置信度，融合生成完整物体点云。'},
        {'id':'hunyuan3d-2','name':'Hunyuan3D-2','title':'Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation',
         'authors':'Tencent Hunyuan3D Team','year':2025,'venue':'技术报告',
         'source':'https://github.com/Tencent-Hunyuan/Hunyuan3D-2','fulltext':'https://arxiv.org/html/2501.12202','file':'Hunyuan3D-2.md',
         'summary':'单图条件扩散生成形状，再生成纹理；本实验比较Mini-Turbo和Turbo的形状分支。'},
        {'id':'spar3d','name':'SPAR3D','title':'SPAR3D: Stable Point-Aware Reconstruction of 3D Objects from Single Images',
         'authors':'Zixuan Huang、Mark Boss、Aaryaman Vasishta、James M. Rehg、Varun Jampani',
         'year':2025,'venue':'技术报告','source':'https://spar3d.github.io/',
         'fulltext':'https://arxiv.org/html/2501.04689','file':'SPAR3D.md',
         'summary':'先生成稀疏点云，再结合图像和点云恢复网格，用中间几何条件辅助单图重建。'}]
    for entry in entries:
        old = next((p for p in catalog['papers'] if p['id']==entry['id']), None)
        entry.update(modules=['perception'], versions=['pointcloud-models'], status='已接入离线重建评测', verifiedAt='2026-11-22')
        if old:
            old.update(entry)
        else:
            catalog['papers'].append(entry)
    zero = next(p for p in catalog['papers'] if p['id']=='zerograsp')
    if 'pointcloud-models' not in zero['versions']: zero['versions'].append('pointcloud-models')
    save(path, catalog)
    texts = {
        'RaySt3R.md': '''## 方法

RaySt3R以单帧RGB-D和目标掩码为输入，将物体补全转化为新视角深度预测。网络针对查询射线生成深度、目标掩码及置信度，再融合多个视角的结果，得到完整形状。[项目主页](https://rayst3r.github.io/)

## 本轮实现

使用作者EvalWrapper及原生eval_scene，输入头部去畸变RGB-D、共享SAM3掩码和内参。预测5个视角，置信度阈值5，关闭TSDF；DINOv2在批次中常驻。输出保留原生相机坐标和米制尺度，不用真值配准后替换部署结果。

## 核查重点

分别检查已观测表面和隐藏补全。保存完整原生点云、最多20,000点评估点云、耗时、峰值显存及几何指标。当前仅做重建测试；下游GraspGenX及物理抓取尚未运行。
''',
        'Hunyuan3D-2.md': '''## 方法

Hunyuan3D 2.0将三维资产生成拆成形状与纹理两阶段。形状模型以图像为条件生成裸网格，纹理模型再生成表面外观。官方另提供蒸馏后的Mini-Turbo与Turbo形状模型。[官方项目与实现](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)

## 本轮实现

只测试两种Turbo形状分支，不生成纹理。使用本地对应DiT与Turbo VAE，5步、分辨率380、FlashVDM和固定随机种子。输入为共享SAM3透明背景RGB；深度不进入生成网络。

## 尺度与坐标

原生形状网格不含ARX场景的米制位姿。另用共享预测掩码内的观测深度定标，保存变换和耗时；真值只进入独立评分。归一化形状分数与部署米制误差分别报告。

公共预热盘子上Mini未生成表面，接口另用官方样例确认。正式案例若空输出，则记录为输出失败，不更改等值面阈值或重试。
''',
        'SPAR3D.md': '''## 方法

SPAR3D先通过轻量点扩散生成稀疏三维点云，再联合图像与点云生成网格。中间点云用于约束后续几何，也允许交互编辑。[项目主页](https://spar3d.github.io/)

## 本轮实现

采用32步、512点的官方条件点云生成，再生成最终网格。保存条件点云、原始网格和统一采样点。只实例化几何所需模块，跳过不参与几何的材质和光照头；所有可学习几何参数均与权重核对，未缺失。

## 输入和评估

输入共享SAM3去背景RGB；生成后仅用观测深度恢复尺度和坐标，定标耗时单列。真值不进入推理。权重来自ModelScope公开镜像并校验其SHA256，配置和许可与HF公开元数据一致；没有独立证明两个站点主权重逐字节相同。

只评估形状与成本，尚未测试抓取。
'''}
    for filename, content in texts.items():
        (SITE/'内容/论文调研'/filename).write_text(content)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--build', action='store_true'); args = parser.parse_args()
    summary = json.loads((RUN/'summary.json').read_text())
    complete = (RUN/'suite_complete.json').exists()
    stopped = (RUN/'STOP.json').exists()
    site_path = SITE/'内容/网站.json'; site = json.loads(site_path.read_text())
    version = next(v for v in site['versions'] if v['id']=='pointcloud-models')
    version.update(status='250项已完成' if complete else '故障停止，记录保留' if stopped else '五模型正式评测进行中',
                   summary='用50个桌面场景比较五种方法，看哪种方法重建物体更准确、处理更快，以及换成不同款式后表现怎样。',
                   metric=f"{summary['records']}/250项重建记录")
    destination = SITE/'内容/资源/pointcloud-benchmark'; destination.mkdir(parents=True, exist_ok=True)
    for name in ['pointcloud-comparison-column.jpg', 'pointcloud-comparison-column-thumb.jpg']:
        site['localAssets'].pop(name, None)
    for file in RESOURCES.glob('*.jpg'):
        if file.stem in {'comparison_column', 'comparison_column_thumb'}:
            continue
        shutil.copy2(file, destination/file.name)
        site['localAssets']['pointcloud-'+file.name.replace('_','-')] = 'pointcloud-benchmark/'+file.name
    for name in ['summary.json', 'case_metrics.csv', 'registration.json', 'verification.json', 'dataset_description.json']:
        source = RUN/name
        if source.exists():
            shutil.copy2(source, destination/name)
            site['localAssets']['pointcloud-'+name.replace('_','-')] = 'pointcloud-benchmark/'+name
    save(site_path, site)
    article = website_article((PROJECT/'docs/pointcloud_benchmark/评测结果.md').read_text())
    prefix = '''## 这次实验做了什么

这次实验比较五种方法从相机图像生成物体三维形状的效果。我们准备了10类常见物体，每类5款，让每种方法都处理同一组50个场景，共完成250次重建，主要看重建是否准确、处理需要多久。

这是[SAM3＋RGB-D点云＋GraspGenX主线](/iterations/sam3-graspgenx/)中的一项对比实验。

| 条件 | 说明 |
| --- | --- |
| 测试场景 | 每个场景在桌面上放一个物体，共测试50个场景。 |
| 相机 | 使用头部RealSense D405相机的参数，画面为640×480像素，相机向下倾斜45°，对着桌面。 |
| 图像准备 | 把彩色图和深度图对齐，再由SAM3标出目标物体在画面中的区域。 |
| 测试方式 | 所有方法都使用同一张RTX4090显卡。每种模型加载后连续处理50个场景，记录输出效果与处理时间。 |
| 同时使用彩色图和深度图的方法 | [RaySt3R](/papers/rayst3r/)、[ZeroGrasp](/papers/zerograsp/) |
| 使用彩色图生成形状的方法 | [两种Hunyuan3D Turbo](/papers/hunyuan3d-2/)、[SPAR3D](/papers/spar3d/)。生成形状后，再利用深度信息确定模型在场景中的实际大小、位置和朝向。 |

[下载每次测试的结果（CSV）](/media/pointcloud-case-metrics.csv) · [下载汇总数据](/media/pointcloud-summary.json) · [查看测试设置](/media/pointcloud-registration.json) · [查看记录检查结果](/media/pointcloud-verification.json)

'''
    (SITE/'内容/迭代/点云生成模型评测实验.md').write_text(prefix+article)
    main_doc = SITE/'内容/迭代/SAM3点云GraspGenX主线.md'; content = main_doc.read_text()
    old = '实验目前待选模型、尚未运行，拥有独立待办与方案目录。'
    content = content.replace(old, '实验已冻结50个头部观测与SAM3掩码，正在比较五种模型；几何结果与正式进度见子版本。尚未运行下游抓取。')
    if complete:
        content = content.replace('正在比较五种模型；几何结果与正式进度见子版本。', '已完成五种模型共250项比较；几何结果与耗时见子版本。')
    content = content.replace('| 待接入 |', '| 本轮静态50例已采集；主线同步待联调 |', 1)
    content = content.replace('| 权重与源码已有；新任务接口待联调 |', '| 本轮50例文本分割已完成；主线多实例接口待联调 |', 1)
    content = content.replace('| 数据契约已定义，待观测实测 |', '| 本轮反投影与独立几何参照完成；下游接口待联调 |', 1)
    main_doc.write_text(content)
    literature()
    if args.build:
        file = SITE/'工具/跟进45厘米实验发布.py'
        spec = importlib.util.spec_from_file_location('site_fonts', file)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module.fonts()
        subprocess.run(['/home/zhaoyechong/.nvm/versions/node/v22.23.2/bin/node', 'build.mjs'], cwd=SITE, check=True)
    print('published', summary['records'], 'records', 'complete' if complete else 'stopped' if stopped else 'running')


if __name__ == '__main__': main()
