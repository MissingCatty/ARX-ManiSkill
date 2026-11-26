"""Build two actual baseline/optimized comparison clips for the pilot report."""
import json
from pathlib import Path
import subprocess

from arx_project.layout import ROOT, sha256, write_json


def comparisons(website, run, accepted, *, prefix='sam3-v8-physical', cache_name='sam3-v8-physical-comparisons.json'):
    binary = subprocess.check_output([str(ROOT/'environments/simulation/bin/python'), '-c',
        'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())'], text=True, timeout=30).strip()
    cache_file = website/'resources/metadata'/cache_name
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    targets = accepted['baseline_success_losses'][:2]
    labels = {'success': 'SUCCESS', 'approach_not_converged': 'APPROACH STOPPED',
              'no_bilateral_target_contact': 'NO BILATERAL CONTACT',
              'insufficient_hold_height': 'OBJECT NOT HELD'}
    output = []
    for case in targets:
        folders = [run/'trials'/case/mode for mode in ['baseline', 'optimized']]
        videos = [p/'execution.mp4' for p in folders]
        if not all(p.exists() for p in videos): continue
        name = prefix+'-'+case.split('_')[0].lower()+'-comparison.mp4'
        target = website/'resources/media'/name
        if target.is_symlink() or target.exists() and target.stat().st_nlink != 1:
            raise ValueError('Comparison videos must be independent website files')
        results = [json.loads((p/'result.json').read_text())['reason'] for p in folders]
        seconds = [json.loads((p/'latency.json').read_text())['timings']['initial_planning_with_postprocessing_seconds']
                   for p in folders]
        source = {'source_sha256': [sha256(p) for p in videos], 'results': results,
                  'seconds': seconds, 'settings': '640x480_per_panel_plus80_header_20fps_crf24_v1'}
        if target.exists() and cache.get(name, {}).get('source') == source:
            output.append((case, name));continue
        durations = []
        for video in videos:
            command = [str(ROOT/'environments/simulation/bin/python'), '-c',
                'import sys,json,imageio_ffmpeg as f;g=f.read_frames(sys.argv[1]);m=next(g);g.close();print(json.dumps(m))', str(video)]
            metadata = json.loads(subprocess.check_output(command, text=True, timeout=30))
            durations.append(metadata['duration'])
        filters = []
        for i, (label, reason, elapsed) in enumerate(zip(['Baseline', 'Optimized'], results, seconds)):
            title = label+' - '+labels.get(reason, reason.upper().replace('_', ' '))
            text = f'Initial planning: {elapsed:.2f} s'
            filters.append(f'[{i}:v]fps=20,scale=640:480:force_original_aspect_ratio=decrease,'
                f'pad=640:560:(ow-iw)/2:80+(480-ih)/2:black,setsar=1,'
                f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{title}':"
                'x=16:y=12:fontsize=20:fontcolor=white,'
                f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{text.replace(':', '')}':"
                'x=16:y=44:fontsize=19:fontcolor=white,'
                f'tpad=stop_mode=clone:stop_duration=60[v{i}]')
        filters.append('[v0][v1]hstack=inputs=2:shortest=1[v]')
        command = [binary, '-nostdin', '-v', 'error', '-y', '-i', str(videos[0]), '-i', str(videos[1]),
            '-filter_complex', ';'.join(filters), '-map', '[v]', '-an', '-t', str(max(durations)),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '24', '-pix_fmt', 'yuv420p',
            '-threads', '2', '-movflags', '+faststart', str(target)]
        subprocess.run(command, check=True, timeout=300)
        cache[name] = {'source': source, 'sha256': sha256(target), 'size': target.stat().st_size,
            'ends_are_padded_with_last_actual_frame': True, 'left': 'baseline', 'right': 'optimized'}
        output.append((case, name))
    write_json(cache_file, cache)
    return output
