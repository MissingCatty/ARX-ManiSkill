"""Standalone accuracy evidence for the existing V7 sphere-count diagnosis."""
import shutil

from arx_project.layout import ROOT, read_json, sha256
from modules.pipeline.experiments.sphere_count_diagnosis.geometry_report import finger_table


def section(website, site):
    pointer = ROOT/'tmp/sphere_count_diagnosis/active.json'
    if not pointer.exists() or not read_json(pointer).get('geometry_fit_complete'):
        return '', []
    root = (ROOT/read_json(pointer)['geometry_fit_run']).resolve()
    if not root.is_relative_to(ROOT/'tmp/sphere_count_diagnosis'):
        raise ValueError('Unexpected geometry-fit root')
    data = read_json(root/'publication_snapshot.json')
    if not data['task_complete'] or not data['independent_review']['passed']:
        raise ValueError('Geometry-fit review is incomplete')

    def attach(source, name):
        target = website/'resources/media'/name
        if target.is_symlink() or target.exists() and target.stat().st_nlink != 1:
            raise ValueError('Fit attachments must be independent files')
        shutil.copy2(source, target)
        if sha256(source) != sha256(target):
            raise RuntimeError('Geometry-fit attachment differs')
        site['localAssets'][name] = '../../resources/media/' + name
        return '/media/' + name

    url = attach(root/'publication_snapshot.json', 'sam3-v7-sphere-fit-results.json')
    image = attach(root/'fingertip_cross_sections.png', 'sam3-v7-sphere-fit-cross-sections.png')
    document = ROOT/'modules/pipeline/docs/mainline_versions/V7_512球与871球的几何拟合精度核查.md'
    name = 'sam3-v7-sphere-fit-report.md';target = website/'resources/media'/name
    if target.is_symlink() or target.exists() and target.stat().st_nlink != 1:
        raise ValueError('Fit report must be an independent file')
    # The published Markdown resolves its figure through the website copy.
    content = document.read_text().replace('resources/sphere-fit-512-vs-871.png', image)
    target.write_text(content)
    if target.read_text() != content:
        raise RuntimeError('Published fit report reference conversion failed')
    site['localAssets'][name] = '../../resources/media/' + name
    report = '/media/' + name
    lines = ['## 512球与871球的拟合精度', '',
        '**当前这版512球不适合直接替换871球，指尖外形膨胀过大。**', '',
        '用SAPIEN/PhysX实际烘焙后的碰撞体测量，分别检查两臂的16个连杆，并对四根夹指加倍采样。'
        '下面统计指尖最前端10毫米区域。“外凸P95”表示球包络外表面到真实碰撞体的距离，'
        '95%的外表面采样位置不超过这个值；“新增外包”表示512球比871球多鼓出多少。', '']
    lines += finger_table(data)
    lines += ['', '当前V2的指尖桌面局部接触余量为2毫米。第一根夹指新增外包P95约6.5毫米，'
        '会明显压缩贴桌、夹缝和邻物附近的可用空间。保留871球，后续减球应重新控制几何误差，优先保住指尖和指腹。', '',
        f'[![同位置、同比例的真实碰撞体与球模型截面]({image})]({image})', '',
        '灰色是真实碰撞体，橙色是球模型截面；每一行使用同一个切面。512球是保守合并，'
        '完整覆盖原871球，新增的主要问题是误拦。', '',
        '本次还发现左侧原有物理分片差异：右侧每根夹指有19个凸片，左侧每根只有1个大凸包，'
        '左侧871球也有局部漏包。完整报告列出了对应位置和误差，后续应单独统一左右物理分片。', '',
        f'[完整拟合报告]({report}) · [逐连杆结果与独立核验]({url})', '']
    batch = {'id': root.name, 'title': '512球与871球几何拟合精度核查', 'date': '2026-11-27',
        'status': '核查完成，保留871球', 'summary': '16连杆及4根夹指加密核查；指尖新增外包P95为6.49/2.65毫米，512不直接替换。',
        'outputs': [str(root.relative_to(ROOT))],
        'artifacts': [{'label': '拟合报告', 'url': report}, {'label': '逐连杆核验', 'url': url},
                      {'label': '截面对比', 'url': image}]}
    return '\n'.join(lines), [batch]
