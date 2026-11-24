"""Copy small map display assets into the existing configuration section."""
import re
import shutil
from pathlib import Path

from 公共 import ROOT,PROJECTS,read,save


def main():
    site=PROJECTS/'claro/进度网站';content=site/'内容'
    source=ROOT/'docs/可达性地图/resources/首版_01';assets=content/'资源/可达性地图'
    assets.mkdir(exist_ok=True)
    page=(source/'交互可达性地图.html').read_text()
    css=re.search(r'<style>(.*?)</style>',page,re.S);js=re.search(r'<script>(.*?)</script>',page,re.S)
    assert css and js
    style=re.sub(r'url\(data:font/woff;base64,[A-Za-z0-9+/=]+\)',
                 'url(/media/reachability-map.woff)',css.group(1))
    (assets/'可达性地图.css').write_text(style)
    for name in ['地图中文.woff','字体许可.txt']:shutil.copy2(Path(__file__).with_name('资源')/name,assets/name)
    (assets/'可达性地图.mjs').write_text(js.group(1))
    page=page.replace(css.group(0),'<link rel="stylesheet" href="/media/reachability-map.css">')
    page=page.replace(js.group(0),'<script type="module" src="/media/reachability-map.mjs"></script>')
    (assets/'交互可达性地图.html').write_text(page)
    for name in ['三维工作区概览.png','桌面高度切片.png']:shutil.copy2(source/name,assets/name)
    configuration=read(content/'网站.json')
    record={'id':'reachability-map','title':'ARX 朝向可达性地图','status':'首版建图与离线核验完成',
        'date':'2026-11-20','summary':'419 万关节采样，327 万静态有效解；展示朝向覆盖、关节余量和桌面工作区域。',
        'file':'朝向可达性地图.md'}
    existing=[i for i,r in enumerate(configuration['configurationRecords']) if r['id']==record['id']]
    if existing:configuration['configurationRecords'][existing[0]]=record
    else:configuration['configurationRecords'].append(record)
    configuration['localAssets'].update({
        'reachability-map.html':'可达性地图/交互可达性地图.html','reachability-map.mjs':'可达性地图/可达性地图.mjs',
        'reachability-map.css':'可达性地图/可达性地图.css','reachability-overview.png':'可达性地图/三维工作区概览.png',
        'reachability-tabletop.png':'可达性地图/桌面高度切片.png',
        'reachability-map.woff':'可达性地图/地图中文.woff','reachability-font-license.txt':'可达性地图/字体许可.txt'})
    save(content/'网站.json',configuration)
    (content/'实验配置/朝向可达性地图.md').write_text('''# ARX 朝向可达性地图

当前右臂六关节、升降约 0.22 米、TCP43、原限位和静态碰撞策略保持不变。独立关节采样 4,194,304 组，3,267,633 组通过原桌面与地面检查，形成 185,926 个 2 厘米位置格及 13,688 个 5 厘米概览格。

[打开交互三维地图](/media/reachability-map.html)

拖动旋转、滚轮缩放、点击体素看数值；可切换全部朝向、朝下朝向、限位余量和雅可比条件指标。交互数据约 0.85 MB，打开该页才加载，页面不连接仿真。完整关节解库不传入浏览器。

[![工作区与朝向覆盖概览](/media/reachability-overview.png)](/media/reachability-overview.png)

[![桌面上方不同高度的朝下方向覆盖](/media/reachability-tabletop.png)](/media/reachability-tabletop.png)

颜色是已采样到的朝向格比例，受采样和体素大小影响，不是实际抓取成功概率。白色空白表示未知。较好的朝下覆盖集中在靠近机器人一侧的桌面区域，具体范围可在交互图中读取。

地图保存了可检索的关节解，可作为 G／H 的 IK 初值。133 个历史困难点的离线对照中，原初始化与地图初值各通过 20 个，新增通过 0、退步 0；本轮尚未证明提高 IK 通过数。查询本身中位耗时 24.4 ms，地图加载与索引构建约 4.7 秒，可常驻复用。

通用地图只包含原桌面、地面和原开爪几何，原自碰撞策略保持关闭；当前杂物、闭爪及整段路径仍需执行端检查。机器人基座、升降、TCP、限位或静态环境变化后，应重新建图或重新验证。所有计算均离线，新增实际抓取 0 次。完整工具、数据与说明位于独立目录 `ARX-可达性地图/`。
''')
    print('SITE_ASSETS_READY',assets)


if __name__=='__main__':main()
