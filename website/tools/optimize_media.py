"""Create reproducible, independent web media copies without changing source data."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile

from PIL import Image, ImageOps, __version__ as pillow_version


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    pending.replace(path)


def valid(row, source, policy_hash, resources):
    if not row or row.get('source_sha256') != digest(source) or row.get('policy_sha256') != policy_hash:
        return False
    return all((resources / item['file']).is_file() and digest(resources / item['file']) == item['sha256']
               for item in row.get('files', []))


def image_job(job):
    source, resources, policy, policy_hash = job
    source, resources = Path(source), Path(resources)
    key = source.relative_to(resources).as_posix()
    source_hash = digest(source)
    stem = re.sub('[^a-z0-9-]', '_', source.stem.lower())[:100]
    prefix = f'{stem}-{source_hash[:12]}-{policy_hash[:8]}'
    destination = resources / 'optimized' / source.parent.name
    destination.mkdir(parents=True, exist_ok=True)
    files, variants = [], []
    with Image.open(source) as original:
        image = ImageOps.exif_transpose(original)
        image = image.convert('RGBA' if 'A' in image.getbands() or 'transparency' in image.info else 'RGB')
        width, height = image.size
        sizes = sorted({w for w in policy['image']['widths'] if w < width} | {width})
        for w in sizes:
            resized = image if w == width else image.resize((w, max(1, round(height * w / width))), Image.Resampling.LANCZOS)
            output = destination / f'{prefix}-w{w}.webp'
            pending = output.with_suffix('.pending')
            resized.save(pending, 'WEBP', quality=policy['image']['quality'], method=policy['image']['method'])
            pending.replace(output)
            if w == width and output.stat().st_size >= source.stat().st_size:
                output.unlink()
                variants.append({'file': key, 'width': w, 'height': height, 'bytes': source.stat().st_size})
                continue
            item = {'file': output.relative_to(resources).as_posix(), 'width': w, 'height': resized.height,
                    'bytes': output.stat().st_size, 'sha256': digest(output)}
            files.append(item)
            variants.append(item)
    return key, {'kind': 'image', 'source_sha256': source_hash, 'source_bytes': source.stat().st_size,
                 'policy_sha256': policy_hash, 'width': width, 'height': height, 'variants': variants,
                 'files': files, 'pillow_version': pillow_version}


def ffmpeg_path():
    if os.environ.get('FFMPEG_EXE'):
        return os.environ['FFMPEG_EXE']
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        path = shutil.which('ffmpeg')
        if path:
            return path
        raise RuntimeError('Install imageio-ffmpeg or set FFMPEG_EXE')


def command(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-5000:])
    return result.stderr


def video_metadata(ffmpeg, path):
    # Probe with FFmpeg itself so a separate ffprobe installation is unnecessary.
    result = subprocess.run([ffmpeg, '-hide_banner', '-i', str(path)], capture_output=True, text=True)
    text = result.stderr
    video = re.search(r'Video: ([^,\s]+).*?\b(\d{2,5})x(\d{2,5})\b.*?([\d.]+) fps', text)
    duration = re.search(r'Duration: (\d+):(\d+):([\d.]+)', text)
    if not video or not duration:
        raise RuntimeError('Cannot probe video: ' + str(path))
    return {'codec': video[1], 'width': int(video[2]), 'height': int(video[3]), 'fps': float(video[4]),
            'duration': int(duration[1])*3600 + int(duration[2])*60 + float(duration[3])}


def faststart(path):
    atoms = []
    with Path(path).open('rb') as stream:
        end = Path(path).stat().st_size
        while stream.tell() + 8 <= end:
            offset = stream.tell()
            length, kind = struct.unpack('>I4s', stream.read(8))
            if length == 1:
                length = struct.unpack('>Q', stream.read(8))[0]
            elif length == 0:
                length = end - offset
            if length < 8:
                break
            atoms.append(kind)
            stream.seek(offset + length)
    return b'moov' in atoms and b'mdat' in atoms and atoms.index(b'moov') < atoms.index(b'mdat')


def video_job(source, resources, policy, policy_hash):
    ffmpeg = ffmpeg_path()
    key = source.relative_to(resources).as_posix()
    source_hash = digest(source)
    folder = resources / 'optimized/video'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f'{source.stem}-{source_hash[:12]}-{policy_hash[:8]}.mp4'
    before = video_metadata(ffmpeg, source)
    attempts, accepted = [], False
    with tempfile.TemporaryDirectory(prefix='encode-', dir=folder) as work:
        candidate, stats = Path(work)/'candidate.mp4', Path(work)/'ssim.txt'
        for crf in policy['video']['crf']:
            command([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source),
                     '-map', '0:v:0', '-map', '0:a?', '-c:v', 'libx264', '-preset', policy['video']['preset'],
                     '-crf', str(crf), '-pix_fmt', 'yuv420p', '-threads', str(policy['video']['threads']),
                     '-c:a', 'aac', '-b:a', '96k', '-movflags', '+faststart', str(candidate)])
            after = video_metadata(ffmpeg, candidate)
            for field in ['width', 'height', 'fps']:
                if before[field] != after[field]:
                    raise RuntimeError('Video geometry/timing changed: ' + key)
            if abs(before['duration'] - after['duration']) > 1 / before['fps']:
                raise RuntimeError('Video duration changed: ' + key)
            log = command([ffmpeg, '-hide_banner', '-i', str(source), '-i', str(candidate),
                           '-filter_complex_threads', str(policy['video']['threads']),
                           '-lavfi', f'[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[a][b]ssim=stats_file={stats}',
                           '-an', '-f', 'null', '-'])
            summary = re.findall(r'All:([\d.]+)', log)
            frames = [float(v) for v in re.findall(r'All:([\d.]+)', stats.read_text())]
            if not summary or not frames:
                raise RuntimeError('Video SSIM was not measured: ' + key)
            attempt = {'crf': crf, 'bytes': candidate.stat().st_size, 'mean_ssim': float(summary[-1]),
                       'minimum_frame_ssim': min(frames), 'frames_checked': len(frames)}
            attempts.append(attempt)
            accepted = (attempt['bytes'] < source.stat().st_size
                        and attempt['mean_ssim'] >= policy['video']['minimum_mean_ssim']
                        and attempt['minimum_frame_ssim'] >= policy['video']['minimum_frame_ssim'])
            print('VIDEO_ATTEMPT', source.name, json.dumps(attempt), flush=True)
            if accepted:
                candidate.replace(output)
                break
            if candidate.stat().st_size >= source.stat().st_size:
                break
        if not accepted:
            # Preserve decoded pixels, but move the MP4 index before the payload.
            command([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source),
                     '-map', '0', '-c', 'copy', '-movflags', '+faststart', str(candidate)])
            candidate.replace(output)
    if not faststart(output):
        raise RuntimeError('MP4 index is not at the front: ' + key)
    item = {'file': output.relative_to(resources).as_posix(), 'bytes': output.stat().st_size, 'sha256': digest(output)}
    return key, {'kind': 'video', 'source_sha256': source_hash, 'source_bytes': source.stat().st_size,
                 'policy_sha256': policy_hash, 'files': [item], 'published': item, 'metadata': before,
                 'transcoded': accepted, 'faststart': True, 'attempts': attempts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    resources = root / 'resources'
    policy_file = root/'media-policy.json'
    policy, policy_hash = json.loads(policy_file.read_text()), digest(policy_file)
    manifest_file = resources/'optimized/manifest.json'
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {'files': {}}
    sources = sorted(p for folder in ['media', 'attachments'] for p in (resources/folder).glob('*')
                     if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.mp4'] and p.is_file())
    pending = [p for p in sources if not valid(manifest['files'].get(p.relative_to(resources).as_posix()), p, policy_hash, resources)]
    images = [p for p in pending if p.suffix.lower() != '.mp4']
    print('MEDIA', json.dumps({'sources': len(sources), 'cached': len(sources)-len(pending), 'images': len(images),
                               'videos': len(pending)-len(images)}), flush=True)
    with ProcessPoolExecutor(max_workers=policy['image']['workers']) as pool:
        jobs = [pool.submit(image_job, (str(p), str(resources), policy, policy_hash)) for p in images]
        for count, future in enumerate(as_completed(jobs), 1):
            key, row = future.result()
            manifest['files'][key] = row
            if count % 20 == 0 or count == len(jobs):
                save_json(manifest_file, manifest)
                print('IMAGES', count, '/', len(jobs), flush=True)
    for source in [p for p in pending if p.suffix.lower() == '.mp4']:
        key, row = video_job(source, resources, policy, policy_hash)
        manifest['files'][key] = row
        save_json(manifest_file, manifest)
    manifest['policy_sha256'] = policy_hash
    manifest['source_count'] = len(sources)
    save_json(manifest_file, manifest)
    print('MEDIA_COMPLETE', len(sources), flush=True)


if __name__ == '__main__':
    main()
