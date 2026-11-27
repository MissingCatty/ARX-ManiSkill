"""Publish verified in-memory state-update results as independent site files."""
import json
import shutil

from arx_project.layout import ROOT, read_json, sha256


def section(website, site):
    pointer = ROOT / 'tmp/state_updates/active.json'
    if not pointer.exists() or not read_json(pointer).get('complete'):
        return None
    active = read_json(pointer)
    root = (ROOT / active['run']).resolve()
    if not root.is_relative_to(ROOT / 'tmp/state_updates'):
        raise ValueError('Unexpected state-update result root')
    data = read_json(root / 'publication_snapshot.json')
    if not data['independent_review']['passed'] or not data['services_stopped']['all_owned_exited']:
        raise ValueError('State-update comparison review incomplete')
    def attach(source, name):
        target = website / 'resources/media' / name
        if target.is_symlink() or target.exists() and target.stat().st_nlink != 1:
            raise ValueError('State-update media must be independent copies')
        if source.suffix == '.json':
            target.write_text(json.dumps(read_json(source), ensure_ascii=False, separators=(',', ':')) + '\n')
            if read_json(target) != read_json(source):
                raise RuntimeError('State-update copied data differs')
        else:
            shutil.copy2(source, target)
            if sha256(source) != sha256(target):
                raise RuntimeError('State-update copied report differs')
        site['localAssets'][name] = '../../resources/media/' + name
        return '/media/' + name
    prefix = 'sam3-v8-' + root.name.replace('_', '-')
    url = attach(root / 'publication_snapshot.json', prefix + '-results.json')
    report = attach(ROOT / active['report'], prefix + '-report.md')
    method = attach(ROOT / 'modules/pipeline/docs/mainline_versions/V8_内存状态更新实施.md', prefix + '-method.md')
    analysis_doc = ROOT / 'modules/pipeline/docs/mainline_versions/V8_内存状态更新剩余耗时.md'
    analysis_url = attach(analysis_doc, prefix + '-analysis.md') if analysis_doc.exists() else None
    history = []
    previous = data.get('previous_pilot')
    if previous:
        prior = (ROOT / previous['run']).resolve()
        if not prior.is_relative_to(ROOT / 'tmp/state_updates'):
            raise ValueError('Unexpected prior state-update result root')
        old_prefix = 'sam3-v8-' + prior.name.replace('_', '-')
        old_report = attach(prior / 'report.md', old_prefix + '-report.md')
        old_data = attach(prior / 'publication_snapshot.json', old_prefix + '-results.json')
        old = previous['acceptance']['summaries']
        history.append({'id': prior.name, 'title': '第一版内存状态更新对照', 'date': '2026-11-27',
            'status': '已完成，未达到采用门槛', 'summary':
            f"原方法／第一版平均{old['control']['mean_seconds']:.2f}／{old['memory']['mean_seconds']:.2f}秒。",
            'outputs': [previous['run']], 'artifacts': [{'label': '完整报告', 'url': old_report},
                                                       {'label': '计时与核验', 'url': old_data}]})
    body = (ROOT / active['report']).read_text().replace('# V8 内存状态更新完整规划对照',
        '## 内存状态更新：完整规划对照', 1).replace('\n## ', '\n### ')
    body += f'\n[实施方法]({method}) · [完整报告]({report}) · [计时与核验数据]({url})\n'
    if analysis_url:
        body += '\n' + analysis_doc.read_text().replace('# V8 内存状态更新后的完整耗时与下一步',
            '## 更新后的完整耗时与下一步', 1).replace('\n## ', '\n### ')
        body += f'\n[完整耗时分析]({analysis_url})\n'
    g = data['acceptance']['summaries']
    summary = (f"共享GPU2下，原方法／内存状态更新平均{g['control']['mean_seconds']:.2f}／"
               f"{g['memory']['mean_seconds']:.2f}秒，原成功{g['control']['successes']}/{g['control']['requests']}、"
               f"新方法{g['memory']['successes']}/{g['memory']['requests']}。")
    status = '内存状态更新完整规划对照完成'
    return {'body': body, 'summary': summary, 'status': status, 'history_batches': history,
        'batch': {'id': root.name, 'title': '内存状态更新完整规划对照', 'date': '2026-11-27',
            'status': status, 'summary': summary, 'outputs': [str(root.relative_to(ROOT))],
            'artifacts': [{'label': '实施方法', 'url': method}, {'label': '完整报告', 'url': report},
                          {'label': '计时与核验', 'url': url}]}}
