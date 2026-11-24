"""等待原批次正常结束，单次核验并发布结果；不重启实验、不重抓。"""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

SITE=Path(__file__).resolve().parents[1]
RUN=SITE.parents[1]/'ManiSkill/outputs/arx_graspcl_compat160_20261120_01'
LOG=SITE/'数据/45厘米实验发布'
NODE=Path('/home/zhaoyechong/.nvm/versions/node/v22.23.2/bin/node')
UNIT='maniskill-arx-graspcl-compat160-20261120-01.service'

def save(state,**extra):
    path=LOG/'状态.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'state':state,'at':datetime.now().astimezone().isoformat(),**extra},ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
    print(state,extra,flush=True)

def command(args):subprocess.run([str(x) for x in args],cwd=SITE,check=True)

def fonts():
    from fontTools import subset
    from fontTools.ttLib import TTFont
    target=SITE/'页面/notosc.woff';previous=TTFont(target);codes=set(previous.getBestCmap());previous.close()
    for folder in ['内容','页面']:
        for file in (SITE/folder).rglob('*'):
            if file.is_file() and file.suffix in {'.md','.json','.mjs','.jsx'}:codes.update(map(ord,file.read_text()))
    plan_root=SITE.parents[1]/'ARX-可达性地图/docs'
    for file in [SITE/'build.mjs',plan_root/'未来改进计划.md',*(plan_root/'可达性地图').glob('*.md')]:
        codes.update(map(ord,file.read_text()))
    source=TTFont(SITE.parent/'缓存/对比视频字体/NotoSansCJKsc-Regular.otf')
    sub=subset.Subsetter();sub.populate(unicodes=codes.intersection(source.getBestCmap()));sub.subset(source);source.flavor='woff'
    tmp=target.with_suffix('.tmp.woff');source.save(tmp);tmp.replace(target)

def main():
    LOG.mkdir(parents=True,exist_ok=True)
    if (LOG/'状态.json').exists():raise RuntimeError('已有发布看护记录；不自动覆盖或重试')
    save('waiting_for_registered_batch')
    last=None
    try:
        while True:
            for name in ['STOP.json','控制器故障.json']:
                if (RUN/name).exists():raise RuntimeError('原批次出现故障，停止发布：'+name)
            d=json.loads((RUN/'progress.json').read_text())
            if d['test_completed']!=last:save('waiting_for_registered_batch',completed=d['test_completed'],total=320);last=d['test_completed']
            unit=subprocess.run(['systemctl','--user','show',UNIT,'-p','ActiveState','--value'],capture_output=True,text=True,check=True).stdout.strip()
            if d['phase']=='complete' and (RUN/'运行结束.json').exists() and unit=='inactive':break
            if unit in {'failed','inactive'} and d['phase']!='complete':raise RuntimeError('原批次未完整结束，停止发布')
            time.sleep(15)
        save('verifying_final_records')
        command([sys.executable,SITE/'工具/汇总45厘米高度实验.py','--final','--gallery'])
        save('registering_version')
        command([sys.executable,SITE/'工具/发布45厘米实验版本.py'])
        save('building_site');fonts();command([NODE,SITE/'build.mjs'])
        with urlopen('http://127.0.0.1:8786/iterations/lift452/',timeout=20) as response:
            html=response.read().decode();assert response.status==200 and html.count('data-scene-title=')==10
        with urlopen('http://127.0.0.1:8786/media/height452-audit.json',timeout=20) as response:
            assert json.load(response)['passed']
        save('published',url='/iterations/lift452/',browser_interaction_check='pending')
    except BaseException as error:
        save('failed',error=repr(error),automatic_retry=False);raise

if __name__=='__main__':main()
