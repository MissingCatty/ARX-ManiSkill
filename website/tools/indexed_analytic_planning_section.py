"""Keep the finer-NMS complete-planning result on the existing V8 page."""
import shutil
import json

from arx_project.layout import ROOT,read_json,sha256


def section(website,site):
    p=ROOT/'tmp/analytic_planning/active.json'
    if not p.is_file() or not read_json(p).get('complete'):return None
    active=read_json(p);root=(ROOT/active['run']).resolve()
    if not root.is_relative_to(ROOT/'tmp/analytic_planning'):raise ValueError('Unexpected analytic planning root')
    data=read_json(root/'publication_snapshot.json')
    if not data['independent_review']['passed'] or not data['services_stopped']['all_owned_exited']:
        raise ValueError('Full planning verification or teardown is incomplete')
    def attach(source,name):
        target=website/'resources/media'/name
        if target.is_symlink() or target.exists() and target.stat().st_nlink!=1:raise ValueError('Attachments require independent copies')
        if source.suffix=='.json':
            target.write_text(json.dumps(read_json(source),ensure_ascii=False,separators=(',',':'))+'\n')
            if read_json(source)!=read_json(target):raise ValueError('Compressed JSON publication differs')
        else:
            shutil.copy2(source,target)
            if sha256(source)!=sha256(target):raise ValueError('Planning publication copy differs')
        site['localAssets'][name]='../../resources/media/'+name
        return '/media/'+name
    url=attach(root/'publication_snapshot.json','sam3-v8-nms15-analytic-planning-results.json')
    report=attach(ROOT/active['report'],'sam3-v8-nms15-analytic-planning-report.md')
    document=(ROOT/active['report']).read_text()
    detail=document.split('## 完整规划结果\n\n',1)[1].split('## 验收与入口',1)[0]
    lines=['## 细化NMS与第七项完整规划接入','',
        f"NMS从30毫米／30°改为15毫米／15°。69例的候选数中位数从{data['offline']['coarse_median']:.0f}增加到{data['offline']['fine_median']:.0f}，"
        '保留更多相近但不同的抓法，原评分和宽深修正保持。','',
        '第七项在原16/64搜索窗口中批量求抓取、预抓取和抬升终点，再沿原顺序检查完整路径。'
        '每候选使用自己的开度，V2指尖网格复核及同候选原方法回退继续执行。两臂Graph在预热时创建，'
        '换场景时更新固定缓存，已验证新增障碍能被识别、还原场景后结果保持。','',
        detail.strip(),'',f'[完整接入报告]({report}) · [逐组计时与核验]({url})','']
    groups=data['acceptance']['summaries'];a,b=groups['fine_numeric'],groups['fine_analytic']
    status='细化NMS与解析端点完整规划对照完成'
    summary=f"细NMS两组规划{a['successes']}/{a['requests']}与{b['successes']}/{b['requests']}；平均{a['mean_seconds']:.2f}→{b['mean_seconds']:.2f}秒。"
    batch={'id':root.name,'title':'细化NMS与解析端点完整规划对照','date':'2026-11-27',
        'status':status,'summary':summary,'outputs':[str(root.relative_to(ROOT))],
        'artifacts':[{'label':'完整规划报告','url':report},{'label':'逐组计时及核验','url':url}]}
    return {'body':'\n'.join(lines),'status':status,'summary':summary,'batch':batch}
