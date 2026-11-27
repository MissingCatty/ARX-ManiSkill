"""Wait read-only, then publish a completed comparison; never restart experiments."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time
import urllib.request

from arx_project.layout import ROOT, now, read_json, sha256, write_json, ensure_run_path


def run_command(command, cwd=ROOT, timeout=1800):
    return subprocess.run([str(x) for x in command], cwd=cwd, check=True, capture_output=True,
        text=True, timeout=timeout).stdout


def publish(root):
    from .registration import guard
    guard(root)
    accepted = read_json(root / 'acceptance.json')
    if not accepted['task_complete'] or not accepted['record_integrity']:
        raise RuntimeError('Do not publish an incomplete execution result as complete')
    if not read_json(root / 'services_stopped.json')['all_owned_exited']:
        raise RuntimeError('Owned GPU workers are not confirmed stopped')
    phase = {'at': now(), 'phase': 'publishing', 'experiment_retries': 0}
    write_json(root / 'publication_progress.json', phase)
    # Prior physical analysis exporters update timestamps while rendering.
    # Preserve their already-published evidence snapshots unchanged.
    names = ['sam3-v8-physical-pilot.json', 'sam3-v8-raw-order-physical-analysis.json']
    preserved = {name: (ROOT / 'website/resources/media' / name).read_bytes() for name in names}
    output = run_command([ROOT / 'environments/curobo/bin/python',
        ROOT / 'website/tools/publish_indexed_planning_plan.py'])
    (root / 'website_publish.log').write_text(output)
    for name, data in preserved.items(): (ROOT / 'website/resources/media' / name).write_bytes(data)
    node = shutil.which('node')
    if not node: raise RuntimeError('Node runtime unavailable for website build')
    output = run_command([node, 'build.mjs'], cwd=ROOT / 'website')
    (root / 'website_build.log').write_text(output)
    prefix = 'sam3-v8-' + root.name.replace('_', '-')
    verified = {}
    for url, file in [('/iterations/sam3-graspgenx/versions/V8/',
            ROOT / 'website/resources/site/iterations/sam3-graspgenx/versions/V8/index.html'),
            ('/media/' + prefix + '-report.md', ROOT / 'website/resources/media' / (prefix + '-report.md')),
            ('/media/' + prefix + '-results.json', ROOT / 'website/resources/media' / (prefix + '-results.json'))]:
        with urllib.request.urlopen('http://127.0.0.1:8786' + url, timeout=30) as response:
            digest = hashlib.sha256(response.read()).hexdigest()
        if digest != sha256(file): raise ValueError('Local publication differs: ' + url)
        if file.is_symlink() or file.stat().st_nlink != 1: raise ValueError('Website file is not an independent copy')
        verified[url] = digest
    for video in accepted['decoded_videos']:
        source = ROOT / video; case, mode = source.parent.parent.name, source.parent.name
        view = 'head' if source.name == 'head.mp4' else 'overview'
        target = ROOT / 'website/resources/media' / (prefix + '-' + case.split('_')[0].lower() + '-' + mode + '-' + view + '.mp4')
        if target.is_symlink() or target.stat().st_nlink != 1 or sha256(target) != sha256(source):
            raise ValueError('Website video is not an independent verified copy')
    write_json(root / 'local_publication.json', {'at': now(), 'passed': True, 'verified_http': verified,
        'independent_recordings': len(accepted['decoded_videos'])})
    cfg = read_json(ROOT / 'configs/deployments/public_website.json')
    with urllib.request.urlopen(cfg['public_http_url'].rstrip('/') + '/api/todos', timeout=30) as response:
        (root / 'public_todos_before.json').write_bytes(response.read())
    template = (ROOT / 'tmp/state_updates/memory5_20261127_03/deploy_site.py').read_text()
    script = template.replace('memory5_20261127_03', root.name).replace(
        'memory-state-planning-20261127-03', 'memory-state-' + root.name.replace('_', '-')).replace(
        'sam3-v8-memory5-20261127-03', prefix)
    deployment = root / 'deploy_site.py'
    if deployment.exists(): raise RuntimeError('Inspect an existing deployment instead of rerunning it')
    deployment.write_text(script)
    (root / 'deployment.log').write_text(run_command([ROOT / 'environments/curobo/bin/python', deployment]))
    for url, digest in verified.items():
        with urllib.request.urlopen(cfg['public_http_url'].rstrip('/') + url, timeout=30) as response:
            if hashlib.sha256(response.read()).hexdigest() != digest: raise ValueError('Public result differs')
    backup = read_json(root / 'public_publication.json')['backup']
    code = 'import pathlib;print(pathlib.Path(' + repr(backup + '/todos-at-switch.snapshot.json') + ').read_text())'
    snapshot = run_command(['ssh', '-S', str(Path.home() / '.ssh/arx_public_repeat_continue'),
        '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', cfg['ssh_user'] + '@' + cfg['host'],
        'python3 -c ' + shlex.quote(code)], timeout=30)
    saved_todos = json.loads(snapshot)
    if len(saved_todos['items']) != len(read_json(root / 'public_todos_after.json')):
        raise ValueError('Public todo snapshot and preserved DB differ')
    (ROOT / 'website/resources/state/todos.snapshot.json').write_text(snapshot)
    # Only these new result files and generated website sources belong to this publication.
    if run_command(['git', 'diff', '--cached', '--name-only']).strip():
        raise RuntimeError('Existing staged edits need separate review before publication commit')
    assets = [str(p.relative_to(ROOT)) for p in (ROOT / 'website/resources/media').glob(prefix + '-*')]
    tracked_assets = [p for p in assets if Path(p).suffix in ['.md', '.json']]
    paths = ['modules/pipeline/docs/mainline_versions/V8_内存状态更新完整抓取测试.md',
        'website/内容/experiments.json', 'website/内容/网站.json', 'website/内容/迭代/sam3-graspgenx-v8.md',
        'website/resources/manifest.json', 'website/resources/optimized/build-report.json',
        'website/resources/state/todos.snapshot.json', *tracked_assets]
    run_command(['git', 'add', '--', *paths]); run_command(['git', 'diff', '--cached', '--check'])
    run_command(['git', 'commit', '-m', 'docs(planning): publish full memory-state grasp comparison'])
    run_command(['git', 'push', 'origin', 'main'], timeout=180)
    commit = run_command(['git', 'rev-parse', 'HEAD']).strip()
    remote = run_command(['git', 'ls-remote', 'origin', 'refs/heads/main']).split()[0]
    if commit != remote: raise RuntimeError('Published Git commit is not on origin/main')
    write_json(root / 'publication_complete.json', {'at': now(), 'passed': True, 'git_commit': commit,
        'public_page': cfg['public_http_url'].rstrip('/') + '/iterations/sam3-graspgenx/versions/V8/',
        'todos_preserved': True, 'copied_recordings': len(accepted['decoded_videos']),
        'experiment_retries': 0})
    phase.update(at=now(), phase='complete'); write_json(root / 'publication_progress.json', phase)


def main(root):
    with (root / 'publication_watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            if (root / 'publication_complete.json').exists(): return
            if (root / 'STOP').exists():
                write_json(root / 'publication_progress.json', {'at': now(), 'phase': 'experiment_stopped',
                    'experiment_retried': False, 'completed_requests': read_json(root / 'progress.json')['completed_requests']})
                return
            if read_json(root / 'progress.json')['phase'] == 'complete': break
            write_json(root / 'publication_progress.json', {'at': now(), 'phase': 'waiting_read_only',
                'completed_requests': read_json(root / 'progress.json')['completed_requests'],
                'experiment_restarts': 0})
            time.sleep(15)
        try: publish(root)
        except BaseException:
            import traceback
            write_json(root / 'publication_error.json', {'at': now(), 'error': traceback.format_exc(),
                'experiment_retries': 0, 'actual_results_preserved': True})
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    main(ensure_run_path(p.parse_args().root))
