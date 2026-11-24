"""Publish failed executed grasps and three synchronized two-view review clips."""
from pathlib import Path
import hashlib
import html
import json
import shutil
import subprocess
import sys

import cv2
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

SITE = Path(__file__).resolve().parents[1]
PROJECT = SITE.parent / 'modules/pipeline'
RUN = PROJECT / 'experiments/diverse_grasp_benchmark/runs/geometry200_fixed452_20261122_01'
DEST = SITE / '内容/资源/grasp-failures'
FONT = SITE.parent / 'tmp/website/arx-video-assets/NotoSansCJKsc-Regular.otf'
REPRESENTATIVES = ['D0002_S1_open_bottled_water', 'D0003_S1_open_mushroom', 'D0042_S2_edge_lemon']


def sha(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def main():
    sys.path[:0] = [str(PROJECT / 'experiments/diverse_grasp_benchmark'), str(SITE / 'tools')]
    from presentation_report import render, describe
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location('gallery_labels', SITE / 'tools/build-scene-gallery.py')
    gallery = module_from_spec(spec)
    spec.loader.exec_module(gallery)
    summary = render(RUN)
    assert json.loads((RUN / 'user_stop.json').read_text())['status'] == 'stopped'
    DEST.mkdir(parents=True, exist_ok=True)
    assets, cases = {}, []

    def register(file, public):
        assets[public] = str(file.relative_to(SITE / '内容/资源'))
        return '/media/' + public

    for claim_file in sorted((RUN / 'claims').glob('*.json')):
        claim = json.loads(claim_file.read_text())
        if claim['state'] != 'complete':
            continue
        folder = RUN / 'trials' / claim_file.stem
        result_file = folder / 'result.json'
        assert sha(result_file) == claim['result_sha256']
        result = json.loads(result_file.read_text())
        if not (result['planning_success'] and result['physical_attempt'] and not result['success']):
            continue
        short = result['id'].split('_')[0].lower()
        item = {'id': result['id'], 'target': gallery.LABELS[result['semantic_class']],
                'failure': describe(result['reason'])[1], 'reason': result['reason'],
                'planning_success': True, 'success': False, 'sources': {}}
        for source_name, suffix in [('execution.mp4', 'overview'), ('head.mp4', 'head')]:
            source = folder / source_name
            assert source.stat().st_size > 0
            public = f'geo-fail-{short}-{suffix}.mp4'
            target = DEST / public
            if not target.exists() or sha(target) != sha(source):
                shutil.copy2(source, target)
            item[suffix] = register(target, public)
            item['sources'][suffix] = {'path': str(source), 'sha256': sha(source)}
        cases.append(item)
    assert len(cases) == summary['planning_passes'] == 50
    font, small = ImageFont.truetype(str(FONT), 25), ImageFont.truetype(str(FONT), 20)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for item in cases:
        if item['id'] not in REPRESENTATIVES:
            continue
        short = item['id'].split('_')[0].lower()
        header = Image.new('RGB', (1680, 72), '#10232f')
        draw = ImageDraw.Draw(header)
        draw.text((16, 6), f'{short.upper()}  {item["target"]}  |  规划通过，抓取失败：{item["failure"]}', font=font, fill='white')
        draw.text((16, 43), '第三人称视角', font=small, fill='#c4e5f1')
        draw.text((976, 43), '头部主相机视角', font=small, fill='#c4e5f1')
        header_file = DEST / (short + '-header.png')
        header.save(header_file)
        paired = DEST / f'geo-fail-{short}-paired.mp4'
        command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                   '-i', str(RUN / 'trials' / item['id'] / 'execution.mp4'),
                   '-i', str(RUN / 'trials' / item['id'] / 'head.mp4'),
                   '-loop', '1', '-framerate', '20', '-i', str(header_file),
                   '-filter_complex', '[0:v]scale=960:540,setsar=1[l];[1:v]scale=720:540,setsar=1[r];[l][r]hstack=inputs=2:shortest=1[p];[2:v][p]vstack=inputs=2:shortest=1[out]',
                   '-map', '[out]', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                   '-pix_fmt', 'yuv420p', '-threads', '2', '-movflags', '+faststart', '-shortest', str(paired)]
        with (DEST / (short + '-encode.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        item['paired'] = register(paired, paired.name)
        cap = cv2.VideoCapture(str(paired))
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * .5))
        ok, frame = cap.read()
        cap.release()
        assert ok
        poster = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        poster.thumbnail((1008, 368))
        poster_file = DEST / f'geo-fail-{short}-poster.jpg'
        poster.save(poster_file, quality=82, optimize=True)
        item['poster'] = register(poster_file, poster_file.name)
        print('PAIRED_VIDEO_READY', item['id'], flush=True)
    css = DEST / 'review.css'
    css.write_text('body{margin:0;background:#f3f6f8;color:#172a35;font:16px/1.65 system-ui,sans-serif}main{max-width:1120px;margin:32px auto;padding:0 20px}h1{font-size:30px}a{color:#086c86}article{background:white;padding:20px;margin:24px 0;border-radius:12px}h2{font-size:21px;margin-top:0}video{display:block;width:100%;background:#10232f;border-radius:8px}table{width:100%;border-collapse:collapse;background:white}th,td{padding:9px 12px;border-bottom:1px solid #dce5eb;text-align:left}.scroll{overflow:auto}.muted{color:#536978}.links{display:flex;gap:20px;flex-wrap:wrap}@media(max-width:650px){main{padding:0 10px}article{padding:12px}h1{font-size:24px}}')
    css_url = register(css, 'grasp-failure-review.css')
    esc = html.escape
    parts = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>规划通过、抓取失败的视频</title>', f'<link rel="stylesheet" href="{css_url}"></head><body><main>',
             '<p><a href="/configuration/scene-gallery/">返回200个场景</a></p><h1>规划通过，实际抓取失败</h1>',
             f'<p>测试已停止。完成{summary["completed"]}例，其中{len(cases)}例规划通过并实际执行，抓取均未成功。另有15例SAM3没有分割出目标、1例规划不出来。停止时1例中断，133例尚未开始。</p>',
             '<p class="muted">下面三段同时显示第三人称和头部主相机，两个视角按原始帧同步播放。完整50例的原始视频目录在下方。</p>']
    for case_id in REPRESENTATIVES:
        item = next(c for c in cases if c['id'] == case_id)
        parts += [f'<article><h2>{esc(case_id.split("_")[0])} · {esc(item["target"])}：{esc(item["failure"])}</h2>',
                  f'<video controls playsinline preload="none" poster="{item["poster"]}"><source src="{item["paired"]}" type="video/mp4"></video>',
                  f'<p class="links"><a href="{item["overview"]}" target="_blank" rel="noopener">第三人称原视频</a><a href="{item["head"]}" target="_blank" rel="noopener">头部原视频</a></p></article>']
    parts += ['<h2>全部50例原始视频</h2><div class="scroll"><table><thead><tr><th>编号</th><th>目标</th><th>抓取失败原因</th><th>第三人称</th><th>头部相机</th></tr></thead><tbody>']
    for item in cases:
        parts.append(f'<tr><td>{esc(item["id"].split("_")[0])}</td><td>{esc(item["target"])}</td><td>{esc(item["failure"])}</td><td><a href="{item["overview"]}" target="_blank" rel="noopener">播放</a></td><td><a href="{item["head"]}" target="_blank" rel="noopener">播放</a></td></tr>')
    parts += ['</tbody></table></div></main></body></html>']
    review = DEST / 'review.html'
    review.write_text('\n'.join(parts))
    review_url = register(review, 'grasp-failure-videos-20261123.html')
    evidence = {'run': str(RUN), 'summary': summary, 'cases': cases, 'review_url': review_url,
                'representatives': REPRESENTATIVES, 'assets': assets}
    (DEST / 'publication.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    config = SITE / '内容/网站.json'
    data = json.loads(config.read_text())
    data.setdefault('localAssets', {}).update(assets)
    temporary = config.with_suffix('.json.pending')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(config)
    print('PUBLISHED_SOURCE', review_url, 'all_failed_grasps', len(cases), flush=True)


if __name__ == '__main__':
    main()
