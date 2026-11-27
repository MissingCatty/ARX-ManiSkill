"""Publish the fixed-window follow-up while preserving the prior comparison."""
import json
import shutil

from arx_project.layout import ROOT, read_json, sha256


def section(website, site):
    pointer = ROOT / 'tmp/parallel_planning/windowed_active.json'
    if not pointer.exists() or not read_json(pointer).get('complete'): return None
    active = read_json(pointer); root = (ROOT / active['run']).resolve()
    if not root.is_relative_to(ROOT / 'tmp/parallel_planning'): raise ValueError('Unexpected batch comparison root')
    data = read_json(root / 'publication_snapshot.json')
    if not data['independent_review']['passed'] or not data['services_stopped']['all_owned_exited']:
        raise ValueError('Batch comparison verification incomplete')
    def attach(source, name):
        target = website / 'resources/media' / name
        if target.is_symlink() or target.exists() and target.stat().st_nlink != 1:
            raise ValueError('Website attachments require independent copies')
        if source.suffix == '.json':
            public = read_json(source)
            environment = public.get('timing_environment', {})
            if 'foreign_observations' in environment:
                environment['foreign_compute_contexts_observed'] = len(environment.pop('foreign_observations'))
                environment.pop('known_baseline_context_pid', None)
            target.write_text(json.dumps(public, ensure_ascii=False, separators=(',', ':')) + '\n')
            if read_json(target) != public: raise RuntimeError('Copied batch results differ')
        else:
            shutil.copy2(source, target)
            if sha256(source) != sha256(target): raise RuntimeError('Copied batch report differs')
        site['localAssets'][name] = '../../resources/media/' + name
        return '/media/' + name
    prefix = 'sam3-v8-' + root.name.replace('_', '-')
    url = attach(root / 'publication_snapshot.json', prefix + '-results.json')
    report = attach(ROOT / active['report'], prefix + '-report.md')
    body = (ROOT / active['report']).read_text().replace('# V8 固定候选窗口的批量端点与并行路径',
        '## 固定候选窗口：批量端点与并行路径', 1).replace('\n## ', '\n### ')
    body += f'\n[完整报告]({report}) · [逐组计时与核验]({url})\n'
    pending = ROOT / 'tmp/parallel_planning/idle_active.json'
    waiting = False
    if pending.exists():
        queued = ROOT / read_json(pending)['run']
        if not (queued / 'acceptance.json').exists() and not (queued / 'STOP').exists():
            waiting = True
            body += '\n空闲卡的同组30次耗时对照已登记，等待没有其他计算任务的显卡；预检通过后自动启动一次。\n'
    g = data['acceptance']['summaries']
    summary = '原方法／批量端点／批量端点加并行路径平均' + '／'.join(
        f"{g[m]['mean_seconds']:.2f}" for m in ['control', 'batched_endpoints', 'batched_paths4']) + '秒。'
    if not data.get('timing_environment', {}).get('exclusive_timing', True):
        summary = '共享GPU下：' + summary
    status = '批量端点功能验证完成，等待空闲卡计时' if waiting else '固定候选窗口的批量端点与路径对照完成'
    return {'body': body, 'summary': summary, 'status': status, 'batch': {'id': root.name,
        'title': '去掉串行凑批后的端点与路径对照', 'date': '2026-11-27', 'status': status, 'summary': summary,
        'outputs': [str(root.relative_to(ROOT))],
        'artifacts': [{'label': '完整报告', 'url': report}, {'label': '计时与核验', 'url': url}]}}
