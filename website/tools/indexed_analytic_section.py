"""Publish the optional GPU analytic IK microbenchmark with independent files."""
import shutil

from arx_project.layout import ROOT, read_json, sha256


def section(website,site):
    pointer=ROOT/'tmp/analytic_ik/active.json'
    if not pointer.is_file() or not read_json(pointer).get('complete'):return None
    active=read_json(pointer);root=(ROOT/active['run']).resolve()
    if not root.is_relative_to(ROOT/'tmp/analytic_ik'):raise ValueError('Unexpected analytic run path')
    data=read_json(root/'publication_snapshot.json')
    if not data['independent_review']['passed']:raise ValueError('GPU analytic review incomplete')

    def attach(source,name):
        target=website/'resources/media'/name
        if target.is_symlink() or target.exists() and target.stat().st_nlink!=1:
            raise ValueError('Analytic attachments must be independent files')
        shutil.copy2(source,target)
        if sha256(source)!=sha256(target):raise RuntimeError('Analytic publication copy differs')
        site['localAssets'][name]='../../resources/media/'+name
        return '/media/'+name

    url=attach(root/'publication_snapshot.json','sam3-v8-gpu-analytic-results.json')
    report=attach(ROOT/active['report'],'sam3-v8-gpu-analytic-report.md')
    doc=(ROOT/active['report']).read_text()
    table=doc.split('## 256目标整批耗时\n\n',1)[1].split('\n## 接回抓取链路的方式',1)[0]
    proof=data['independent_review']
    initialization=[r['initialization_and_graph_capture_seconds'] for r in data['acceptance']['cases']]
    lines=['## GPU解析IK与批量碰撞验证','','**ARX可以使用解析IK。已完成GPU实现和常驻查询接口。**','',
        '真实URDF的第22、23、24关节轴平行，第25、26关节轴相交，可使用[IK-Geo／EAIK](/papers/eaik/)的几何分解。'
        '一次并行求256个原目标、每目标最多8个普通分支，再把各分支与对应开爪宽度送进cuRobo的871球碰撞检查。','',
        '最终版本用双精度计算解析式，输出单精度关节命令；左右臂共8192个已知可达目标全部找回原分支，'
        '329个EAIK精确参考分支全部匹配。三种开度、480个状态与原锁定开度模型的FK及碰撞判断一致。','',
        table.strip(),'',
        f'首次创建场景会话并捕获Graph需{min(initialization):.2f}～{max(initialization):.2f}秒，'
        '后续候选批次复用同一个会话。接入整次规划时，需要同时计入新场景准备和查询开销。','',
        '| 验收项目 | 结果 |','| --- | --- |']
    for key,label in [('task_result','任务结果'),('record_integrity','记录完整性'),
        ('event_applicability','事件适用性'),('key_event_coverage','关键事件覆盖')]:lines.append(f'| {label} | {proof[key]} |')
    lines+=['',f'[完整方法、接口和核验报告]({report}) · [逐组计时及登记副本]({url})','']
    status='GPU解析IK与批量碰撞模块验证完成，待完整规划接入'
    summary='8192/8192已知可达目标找回原分支；固定256目标解析IK约1.25毫秒，871球世界/自碰撞分场景计时。'
    batch={'id':root.name,'title':'GPU解析IK与原生批量碰撞核查','date':'2026-11-27',
        'status':status,'summary':summary,'outputs':[str(root.relative_to(ROOT))],
        'artifacts':[{'label':'方法与接口报告','url':report},{'label':'逐组核验与计时','url':url}]}
    return {'body':'\n'.join(lines),'status':status,'summary':summary,'batch':batch}
