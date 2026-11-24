"""Bounded host-only lossless archival of NEW ARX viewport recordings.

No robot/mailbox/planner access. Each completed batch is decoded and compared
with every original RGBA pixel before releasing its temporary PNG files.
The original timing manifest is never rewritten. Existing recordings are not
eligible: the worker must be started before its new frames directory exists.
"""
from pathlib import Path
import hashlib
import json
import math
import subprocess
import threading
import time
import imageio_ffmpeg
from PIL import Image
from archive_recorded_frames import decode


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


class StreamingRecordingArchive:
    def __init__(self, folder, batch_frames=32):
        self.folder = Path(folder).resolve()
        self.frames = self.folder / 'frames'
        assert not self.frames.exists(), 'Only NEW recording directories'
        self.chunks = self.folder / 'recording_chunks'
        self.chunks.mkdir(exist_ok=False)
        self.batch_frames = batch_frames
        self.rows, self.parts = [], []
        self.dimensions = None
        self.error = None
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.done.wait(.5):
                files = sorted(self.frames.glob('[0-9][0-9][0-9][0-9][0-9][0-9].png'))
                # Capture writes asynchronously. Retain a recent tail and only
                # accept complete PNGs unchanged for at least two seconds.
                available = {f.name: f for f in files[:-8]}
                expected = [f'{i:06d}.png' for i in range(len(self.rows), len(self.rows) + self.batch_frames)]
                if all(name in available for name in expected):
                    eligible = [available[name] for name in expected]
                    if all(time.time() - f.stat().st_mtime >= 2 and f.read_bytes().endswith(b'\x00\x00\x00\x00IEND\xaeB`\x82') for f in eligible):
                        self._batch(eligible)
        except BaseException as exc:
            self.error = repr(exc)
            write(self.folder / 'streaming_archive_error.json', {'error': self.error})

    def _batch(self, files):
        start = len(self.rows)
        assert [f.name for f in files] == [f'{i:06d}.png' for i in range(start, start + len(files))]
        rows = []
        for file in files:
            raw = file.read_bytes()
            assert raw.endswith(b'\x00\x00\x00\x00IEND\xaeB`\x82'), 'Incomplete PNG; retain originals'
            with Image.open(file) as im:
                im.load()
                assert im.mode in ('RGB', 'RGBA') and not im.info
                if self.dimensions is None:
                    self.dimensions = im.size
                assert self.dimensions == im.size
                rgba = im.convert('RGBA')
                assert rgba.getchannel('A').getextrema() == (255, 255)
                rows.append(dict(file=file.name, source_mode=im.mode,
                                 source_png_sha256=sha(raw), rgba_pixels_sha256=sha(rgba.tobytes())))
        part = self.chunks / f'{len(self.parts):06d}.mkv'
        assert not part.exists()
        temporary = part.with_suffix('.partial.mkv')
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-framerate', '5',
                        '-start_number', str(start), '-i', str(self.frames / '%06d.png'),
                        '-frames:v', str(len(files)), '-map_metadata', '-1', '-an',
                        '-c:v', 'libx264rgb', '-crf', '0', '-preset', 'ultrafast',
                        '-threads', '2', '-pix_fmt', 'rgb24', str(temporary)], check=True, timeout=180)
        for index, pixels in enumerate(decode(temporary, *self.dimensions, len(rows))):
            assert sha(pixels) == rows[index]['rgba_pixels_sha256'], 'Pixel mismatch; retain PNGs'
        assert all(sha(f.read_bytes()) == row['source_png_sha256'] for f, row in zip(files, rows))
        temporary.replace(part)
        self.parts.append(dict(path=str(part), frames=len(rows), start=start,
                               sha256=sha(part.read_bytes()), every_rgba_pixel_verified=True))
        self.rows.extend(rows)
        write(self.folder / 'streaming_archive_journal.json',
              dict(parts=self.parts, frames=self.rows, dimensions=self.dimensions,
                   scope='Only verified NEW capture PNGs released; complete timing recorded by original simulator'))
        for file, row in zip(files, rows):
            assert sha(file.read_bytes()) == row['source_png_sha256']
            file.unlink()

    def finish(self):
        self.done.set()
        self.thread.join(timeout=180)
        assert not self.thread.is_alive(), 'Archive worker did not stop'
        assert self.error is None, self.error
        timing = self.frames / 'frames.json'
        info = json.loads(timing.read_text())
        assert not info.get('error') and info['frames']
        expected = len(info['frames'])
        # The recorder retains asynchronous capture futures. Wait only for its
        # remaining files to be fully materialized; do not issue mailbox calls.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            remaining = sorted(self.frames.glob('*.png'))
            if len(self.rows) + len(remaining) == expected:
                if all(f.read_bytes().endswith(b'\x00\x00\x00\x00IEND\xaeB`\x82') for f in remaining):
                    break
            time.sleep(.25)
        assert len(self.rows) + len(remaining) == expected, 'Missing capture files'
        while remaining:
            self._batch(remaining[:self.batch_frames])
            remaining = remaining[self.batch_frames:]
        assert [r['file'] for r in self.rows] == [r['file'] for r in info['frames']]
        sources = self.folder / 'raw_chunk_inputs.txt'
        sources.write_text(''.join(f"file '{part['path']}'\n" for part in self.parts))
        video = self.folder / 'all_frames_lossless.mkv'
        assert not video.exists()
        temporary = self.folder / 'all_frames_lossless.partial.mkv'
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-f', 'concat', '-safe', '0',
                        '-i', str(sources), '-map_metadata', '-1', '-an', '-c:v', 'copy', str(temporary)],
                       check=True, timeout=180)
        for index, pixels in enumerate(decode(temporary, *self.dimensions, len(self.rows))):
            assert sha(pixels) == self.rows[index]['rgba_pixels_sha256'], 'Combined frame order/pixel mismatch'
        temporary.replace(video)
        report = dict(version=1, status='verified_sources_archived', container=video.name,
                      container_sha256=sha(video.read_bytes()), dimensions=list(self.dimensions), frames=self.rows,
                      codec='libx264rgb CRF0; each batch and final combined RGBA sequence verified',
                      original_alpha=255, timing_manifest='frames.json', timing_manifest_sha256=sha(timing.read_bytes()),
                      all_original_frames_included=True, source_cleanup_requested=True,
                      original_pngs_removed=len(self.rows), verified_unix=time.time(),
                      scope='Host-only streaming representation; original complete wall/sim timing unchanged; no spatial/temporal frame loss')
        write(self.frames / 'lossless_archive.json', report)
        # The verified combined archive now preserves every frame. Release only
        # its temporary new chunk copies, retaining their complete journal.
        for part in self.parts:
            file = Path(part['path'])
            assert sha(file.read_bytes()) == part['sha256']
            file.unlink()
        return dict(container=str(video), manifest=str(self.frames / 'lossless_archive.json'),
                    frames=len(self.rows), pixel_exact_verified=True, source_pngs_archived=len(self.rows),
                    bytes=video.stat().st_size,
                    restore_command=f'python scripts/curobo_arx/archive_recorded_frames.py restore --folder {self.folder}')


def encode_motion(folder):
    """Same frame/range selection as original preview, streamed without PNGs."""
    folder = Path(folder)
    info = json.loads((folder / 'frames/frames.json').read_text())
    archive = json.loads((folder / 'frames/lossless_archive.json').read_text())
    assert not info.get('error') and archive['all_original_frames_included']
    ranges = []
    for path in folder.glob('*_execution.json'):
        e = json.loads(path.read_text()); end = e['final']['sim_time']
        ranges.append((end - e['sim_duration_s'] - .4, end + .5))
    selected = [(i, f) for i, f in enumerate(info['frames']) if not ranges or any(a <= f['sim_time'] <= b for a, b in ranges)]
    if not ranges:
        selected = selected[:15]
    assert len(selected) >= 3
    counts, elapsed, written = {}, 0., 0
    for j, (index, frame) in enumerate(selected):
        duration = selected[j+1][1]['wall_time'] - frame['wall_time'] if j+1 < len(selected) and selected[j+1][0] == index+1 else .2
        elapsed += max(.001, duration)
        desired = int(math.floor(elapsed * 5 + .5))
        counts[index] = max(0, desired - written)
        written += counts[index]
    counts[selected[-1][0]] += 1
    width, height = archive['dimensions']
    video = folder / 'motion.mp4'
    assert not video.exists()
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba',
               '-s', f'{width}x{height}', '-r', '5', '-i', 'pipe:0', '-an', '-c:v', 'libx264',
               '-threads', '2', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    timer = threading.Timer(180, process.kill); timer.daemon = True; timer.start()
    try:
        for index, pixels in enumerate(decode(folder / archive['container'], width, height, len(archive['frames']))):
            assert sha(pixels) == archive['frames'][index]['rgba_pixels_sha256']
            for _ in range(counts.get(index, 0)):
                process.stdin.write(pixels)
        process.stdin.close()
        assert process.wait(timeout=30) == 0, process.stderr.read().decode(errors='replace')
    finally:
        timer.cancel()
        if process.poll() is None:
            process.kill(); process.wait()
        process.stderr.close()
    return dict(video=str(video), camera_path=info['camera_path'], camera_matrix=info['camera_matrix'],
                frames=len(selected), editing='Original stage ranges; original wall durations quantized to5fps; no overlays/audio; every source pixel retained in separate lossless archive',
                preview_output_frames=written+1)
