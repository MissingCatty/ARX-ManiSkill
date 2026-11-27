"""Copy parallel-planning evidence into the independently runnable website."""
import json
import shutil

from arx_project.layout import ROOT, read_json, sha256


def section(website, site):
    pointer = ROOT / 'tmp/parallel_planning/active.json'
    if not pointer.exists() or not read_json(pointer).get('complete'):
        return None
    active = read_json(pointer)
    root = (ROOT / active['run']).resolve()
    if not root.is_relative_to(ROOT / 'tmp/parallel_planning'):
        raise ValueError('Unexpected parallel-planning directory')
    data = read_json(root / 'publication_snapshot.json')
    if not data['independent_review']['passed'] or not data['services_stopped']['all_owned_exited']:
        raise ValueError('Parallel planning has not passed complete verification')
    def attach(source, name):
        target = website / 'resources/media' / name
        if target.is_symlink() or target.exists() and target.stat().st_nlink != 1:
            raise ValueError('Website attachments must be independent files')
        if source.suffix == '.json':
            target.write_text(json.dumps(read_json(source), ensure_ascii=False, separators=(',', ':')) + '\n')
            if read_json(target) != read_json(source): raise RuntimeError('Copied results differ')
        else:
            shutil.copy2(source, target)
            if sha256(source) != sha256(target): raise RuntimeError('Copied report differs')
        site['localAssets'][name] = '../../resources/media/' + name
        return '/media/' + name
    url = attach(root / 'publication_snapshot.json', 'sam3-v8-parallel-path-planning-results.json')
    report = attach(ROOT / active['report'], 'sam3-v8-parallel-path-planning-report.md')
    document = (ROOT / active['report']).read_text()
    body = document.replace('# V8 预抓取并行路径规划', '## 预抓取并行路径规划', 1)
    body = body.replace('\n## ', '\n### ')
    body += f'\n[完整报告]({report}) · [逐组计时与核验]({url})\n'
    groups = data['acceptance']['summaries']
    summary = '原方法／并行4／并行8平均' + '／'.join(f"{groups[m]['mean_seconds']:.2f}" for m in
        ['control', 'parallel4', 'parallel8']) + '秒；各组' + '、'.join(
        f"{groups[m]['successes']}/{groups[m]['requests']}" for m in ['control', 'parallel4', 'parallel8']) + '通过。'
    status = '预抓取并行路径对照完成'
    return {'body': body, 'status': status, 'summary': summary,
        'batch': {'id': root.name, 'title': '预抓取并行路径完整对照', 'date': '2026-11-27',
            'status': status, 'summary': summary, 'outputs': [str(root.relative_to(ROOT))],
            'artifacts': [{'label': '完整报告', 'url': report}, {'label': '计时与核验', 'url': url}]}}
