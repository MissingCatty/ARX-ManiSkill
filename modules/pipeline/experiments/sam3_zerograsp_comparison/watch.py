"""Read-only follow-up, independent evidence checks and website publication."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from arx_project.layout import ROOT, now, read_json, sha256, write_json
from .cohort import check_pair


def audit(root):
    import numpy as np
    from modules.zerograsp.arx_grasp_adapter import common_candidates
    from modules.curobo.fingertip_contact import load_profile
    sys.path[:0] = [str(ROOT/'modules/maniskill'), str(ROOT/'modules/maniskill/planning_common')]
    from planning_common.candidates import load_candidates
    from planning_common.contact_execution import verify_trial
    reg = read_json(root/'registration.json')
    issues, rows, videos = [], [], []
    for pair in reg['cohort']['pairs']:
        claim = root/'claims'/(pair['id']+'.json')
        if not claim.exists():
            rows.append({'id': pair['id'], 'state': 'not_started'})
            continue
        receipt = read_json(claim)
        if receipt['state'] != 'complete':
            rows.append({'id': pair['id'], 'state': 'interrupted'})
            continue
        out = root/'trials'/pair['id']
        check_pair(pair)
        try:
            if sha256(out/'result.json') != receipt['result_sha256']:
                raise RuntimeError('Result claim hash mismatch')
            if sha256(out/'paired_latency.json') != receipt['paired_latency_sha256']:
                raise RuntimeError('Paired latency receipt changed')
            result = read_json(out/'result.json')
            if not pair.get('common_precondition_failure'):
                if not read_json(out/'paired_restore_verification.json')['passed']:
                    raise RuntimeError('Paired physical state was not verified')
                for name in ['camera/color.png', 'camera/depth_m.npy', 'camera/mask.png', 'state.json']:
                    if sha256(out/name) != pair['input_hashes'][name]:
                        raise RuntimeError('Inference input differs from paired SAM observation')
                native, _ = load_candidates(out)
                actual = read_json(out/'candidates/candidate_pool.json')['candidates']
                expected = common_candidates(native, read_json(out/'state.json')['camera_world_optical'])
                if actual != expected:
                    raise RuntimeError('Native ZeroGrasp candidates were changed')
            if result.get('planning_success'):
                proof = read_json(out/'prepared_plan_execution_gate.json')
                if not proof['passed'] or proof['selected_arm'] != result['selected_arm']:
                    raise RuntimeError('Prepared-plan execution gate missing or arm mismatch')
                for name, digest in proof['artifact_hashes'].items():
                    if sha256(out/name) != digest:
                        raise RuntimeError('Prepared-plan artifact changed')
            # Verify in a separate audit directory so accepted original bytes stay immutable.
            work = root/'independent_readback'/pair['id']
            work.mkdir(parents=True, exist_ok=True)
            for name in ['result.json', 'execution.jsonl', 'contact_entities.json', 'execution_origin.json']:
                if (out/name).is_file():
                    shutil.copy2(out/name, work/name)
            verify_trial(work, load_profile())
            evidence = read_json(work/'independent_evaluation.json')
            if not evidence['passed'] or evidence.get('independently_successful') != result['success']:
                raise RuntimeError('Independent physical result differs')
            videos.extend(out.glob('*.mp4'))
            rows.append({'id': pair['id'], 'state': 'complete', 'success': result['success'],
                         'planning_success': bool(result.get('planning_success'))})
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            issues.append(pair['id']+': '+str(error))
    import imageio_ffmpeg
    binary = imageio_ffmpeg.get_ffmpeg_exe()

    def decode(path):
        done = subprocess.run([binary, '-v', 'error', '-threads', '1', '-i', str(path),
            '-threads', '1', '-f', 'null', '-'], capture_output=True, text=True, timeout=300)
        return {'file': str(path.relative_to(root)), 'passed': done.returncode == 0}

    with ThreadPoolExecutor(max_workers=4) as pool:
        decoded = list(pool.map(decode, videos))
    if not all(value['passed'] for value in decoded):
        issues.append('Video decode failed')
    stopped = read_json(root/'services_stopped.json')
    if not stopped['all_owned_exited']:
        issues.append('Owned services did not exit')
    record = {'at': now(), 'passed': not issues, 'record_integrity': not issues,
        'issues': issues, 'results': rows, 'videos': decoded,
        'task_complete': len([r for r in rows if r['state'] == 'complete']) == len(reg['cohort']['pairs'])}
    write_json(root/'acceptance.json', record)
    return record


def publish(root):
    subprocess.run([sys.executable, str(ROOT/'website/tools/publish_sam3_zerograsp_v6.py'),
                    '--root', str(root)], cwd=ROOT, check=True)
    node = ROOT/'environments/node/bin/node'
    if not node.is_file():
        node = max((Path.home()/'.nvm/versions/node').glob('v*/bin/node'),
                   key=lambda p: tuple(int(v) for v in p.parents[1].name.lstrip('v').split('.')))
    with (root/'website_build.log').open('a') as stream:
        subprocess.run([str(node), 'build.mjs'], cwd=ROOT/'website', stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    write_json(root/'website_publication.json', {'at': now(), 'published': True})


def watch(root):
    last = None
    while True:
        try:
            progress = read_json(root/'progress.json') if (root/'progress.json').exists() else None
            count = progress['completed_pairs'] if progress else 0
            phase = progress['phase'] if progress else 'queued'
            terminal = phase in ['complete', 'stopped']
            exited = (root/'services_stopped.json').exists() and read_json(root/'services_stopped.json')['all_owned_exited']
            stamp = (count//5, phase, terminal and exited)
            if stamp != last:
                if terminal and exited:
                    audit(root)
                publish(root)
                last = stamp
            if terminal and exited:
                return
        except Exception:
            write_json(root/'publication_error.json', {'at': now(), 'error': traceback.format_exc()})
            print(traceback.format_exc(), flush=True)
        time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    watch(parser.parse_args().root.resolve())
