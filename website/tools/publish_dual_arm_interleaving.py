"""Record the shared arm-search optimization in V3 with standalone resources."""
from pathlib import Path
import shutil

from arx_project.layout import ROOT,read_json,write_json,sha256


HEADING = '## 双臂候选交替搜索优化'
SECTION = ROOT/'website/content_sections/双臂候选交替搜索.md'


def add_section(text, section):
    if HEADING in text:
        start = text.index(HEADING)
        end = text.find('\n## ',start+len(HEADING))
        text = text[:start]+(text[end:] if end >= 0 else '')
    marker = '## 参考方法'
    if marker in text:
        start = text.index(marker)
        return text[:start].rstrip()+'\n\n'+section.rstrip()+'\n\n'+text[start:]
    return text.rstrip()+'\n\n'+section.rstrip()+'\n'


def publish():
    website = ROOT/'website'
    site_path = website/'内容/网站.json'
    versions_path = website/'内容/experiments.json'
    site,versions = read_json(site_path),read_json(versions_path)
    proof_source = website/'resources/media/sam3-v6-interleaving-validation.json'
    method_source = ROOT/'modules/curobo/docs/双臂候选交替搜索.md'
    for source,name in [(proof_source,'sam3-v3-interleaving-validation.json'),
                        (method_source,'sam3-v3-interleaving-method.md')]:
        target = website/'resources/media'/name
        if target.is_symlink():raise ValueError('Website attachment must be an independent file')
        shutil.copy2(source,target)
        if target.stat().st_nlink != 1 or sha256(source) != sha256(target):
            raise ValueError('Website resource copy failed')
        site['localAssets'][name] = '../../resources/media/'+name
    document = method_source.read_text().splitlines()
    body = '\n'.join('#'+line if line.startswith('#') else line for line in document[1:])
    section = (HEADING+'\n\n'
        '这项优化属于V3建立的双臂选择模块。验证复用了V6保存的观测、机器人状态和候选，完成19次纯规划请求；输入共5组，来自4个场景。\n\n'
        +body.strip()+'\n\n[下载方法文档](/media/sam3-v3-interleaving-method.md) · '
        '[下载验证数据](/media/sam3-v3-interleaving-validation.json)\n')
    SECTION.parent.mkdir(parents=True,exist_ok=True)
    SECTION.write_text(section)
    for page in [website/'内容/迭代/sam3-graspgenx-v3.md',
                 ROOT/'modules/pipeline/docs/mainline_versions/V3_双臂选择与抓取.md']:
        page.write_text(add_section(page.read_text(),section))
    group = next(v for v in versions['experiments'] if v['id']=='sam3-graspgenx')
    version = next(v for v in group['iterations'] if v['code']=='V3')
    base_status = version['status'].split('；交替搜索')[0]
    version['status'] = base_status+'；交替搜索优化已验证'
    base_summary = version['resultSummary'].split('交替搜索')[0].rstrip()
    version['resultSummary'] = base_summary+' 交替搜索5组输入结果一致，香蕉规划428.23→8.55秒。'
    version['settings'] = [v for v in version['settings'] if v[0]!='候选搜索']+[
        ['候选搜索','首轮每臂16个、后续每批64个，保存游标交替续查，同名次比较关节位移。']]
    batch = {'id':'dual_arm_interleaving_20261126','title':'双臂候选交替搜索纯规划验证',
        'date':'2026-11-26','status':'已完成','summary':'28项离线检查、19次纯规划请求，5组输入的规划成败、抓姿和选臂一致；物理抓取0次。',
        'outputs':['tmp/dual_arm_interleaving/replay_20261126_01','tmp/dual_arm_interleaving/replay_20261126_02',
                   'tmp/dual_arm_interleaving/window_sizes_20261126_01'],
        'artifacts':[{'label':'方法文档','url':'/media/sam3-v3-interleaving-method.md'},
                     {'label':'纯规划验证数据','url':'/media/sam3-v3-interleaving-validation.json'}]}
    version['batches'] = [v for v in version['batches'] if v['id']!=batch['id']]+[batch]
    write_json(site_path,site);write_json(versions_path,versions)
    print('Recorded the arm-search optimization in V3; current node remains '+group['current'])


if __name__=='__main__':
    publish()
