"""Lossless, pixel-verified archive of NEW ARX recordings; old frames untouched."""
import argparse,hashlib,json,subprocess,threading,time
from pathlib import Path

import imageio_ffmpeg
from PIL import Image


def sha(data):return hashlib.sha256(data).hexdigest()


def decode(path,width,height,count):
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-threads','2','-i',str(path),'-map','0:v:0','-f','rawvideo','-pix_fmt','rgba','pipe:1']
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    timer=threading.Timer(300,process.kill);timer.daemon=True;timer.start()
    try:
        size=width*height*4
        for index in range(count):
            data=bytearray()
            while len(data)<size:
                chunk=process.stdout.read(size-len(data))
                if not chunk:raise RuntimeError(f'Incomplete lossless decode at frame{index}')
                data.extend(chunk)
            yield bytes(data)
        if process.stdout.read(1):raise RuntimeError('Unexpected extra decoded frames')
        if process.wait(timeout=30):raise RuntimeError(process.stderr.read().decode(errors='replace'))
    finally:
        timer.cancel()
        if process.poll() is None:process.kill();process.wait()
        process.stdout.close();process.stderr.close()


def archive(folder,remove_source=False):
    folder=Path(folder);frames=folder/'frames';manifest=frames/'lossless_archive.json';video=folder/'all_frames_lossless.mkv'
    assert not manifest.exists() and not video.exists(),'Never overwrite a frame archive'
    timing=frames/'frames.json';info=json.loads(timing.read_text());count=len(info['frames'])
    assert count>0 and not info.get('error')
    assert len(list(frames.glob('*.png')))==count,'Frame count differs from recording metadata'
    rows=[];dimensions=None
    for index in range(count):
        file=frames/f'{index:06d}.png'
        with Image.open(file) as im:
            assert im.mode in ('RGB','RGBA') and not im.info,'Preserve unsupported image metadata/modes as PNG'
            if dimensions is None:dimensions=im.size
            assert im.size==dimensions
            rgba=im.convert('RGBA');assert rgba.getchannel('A').getextrema()==(255,255),'Keep nonopaque original frames'
            rows.append({'file':file.name,'source_mode':im.mode,'source_png_sha256':sha(file.read_bytes()),'rgba_pixels_sha256':sha(rgba.tobytes())})
    temporary=folder/'all_frames_lossless.partial.mkv'
    assert not temporary.exists()
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-framerate','5','-i',str(frames/'%06d.png'),'-frames:v',str(count),
             '-map_metadata','-1','-an','-c:v','libx264rgb','-crf','0','-preset','ultrafast','-threads','2','-pix_fmt','rgb24',str(temporary)]
    subprocess.run(command,check=True,timeout=300)
    for index,pixels in enumerate(decode(temporary,*dimensions,count)):
        assert sha(pixels)==rows[index]['rgba_pixels_sha256'],f'Lossless pixel mismatch frame{index}; original PNGs retained'
    temporary.rename(video)
    report={'version':1,'status':'verified','container':video.name,'container_sha256':sha(video.read_bytes()),'dimensions':list(dimensions),'frames':rows,
            'codec':'libx264rgb CRF0; verified every decoded RGBA pixel against original PNG','original_alpha':255,
            'timing_manifest':'frames.json','timing_manifest_sha256':sha(timing.read_bytes()),'all_original_frames_included':True,
            'source_cleanup_requested':remove_source,'original_pngs_removed':0,'verified_unix':time.time(),
            'scope':'All original pixels/alpha and complete original wall/sim-time metadata preserved. PNG container bytes may differ on restoration. Display MP4/view/editing unchanged.'}
    manifest.write_text(json.dumps(report,indent=2)+'\n')
    if remove_source:
        assert sha(timing.read_bytes())==report['timing_manifest_sha256']
        for row in rows:
            file=frames/row['file'];assert sha(file.read_bytes())==row['source_png_sha256'],'Source changed; preserve remaining PNGs'
            file.unlink();report['original_pngs_removed']+=1
        report['status']='verified_sources_archived'
        manifest.write_text(json.dumps(report,indent=2)+'\n')
    return {'container':str(video),'manifest':str(manifest),'frames':count,'pixel_exact_verified':True,'source_pngs_archived':report['original_pngs_removed'],
            'bytes':video.stat().st_size,'restore_command':f'python scripts/curobo_arx/archive_recorded_frames.py restore --folder {folder}'}


def restore(folder,index=None):
    folder=Path(folder);frames=folder/'frames';manifest=json.loads((frames/'lossless_archive.json').read_text());video=folder/manifest['container']
    assert sha(video.read_bytes())==manifest['container_sha256']
    assert sha((frames/manifest['timing_manifest']).read_bytes())==manifest['timing_manifest_sha256']
    rows=manifest['frames'];assert index is None or 0<=index<len(rows)
    restored=0
    for i,pixels in enumerate(decode(video,*manifest['dimensions'],len(rows))):
        assert sha(pixels)==rows[i]['rgba_pixels_sha256']
        if index is not None and index!=i:continue
        file=frames/rows[i]['file']
        if file.exists():
            with Image.open(file) as im:assert sha(im.convert('RGBA').tobytes())==rows[i]['rgba_pixels_sha256'],'Never overwrite changed frames'
            continue
        image=Image.frombytes('RGBA',tuple(manifest['dimensions']),pixels).convert(rows[i]['source_mode'])
        image.save(file)
        with Image.open(file) as im:assert sha(im.convert('RGBA').tobytes())==rows[i]['rgba_pixels_sha256']
        restored+=1
    return {'restored_frames':restored,'all_decoded_pixels_verified':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['archive','restore']);parser.add_argument('--folder',required=True);parser.add_argument('--remove-source',action='store_true');parser.add_argument('--index',type=int)
    args=parser.parse_args();print(json.dumps(archive(args.folder,args.remove_source) if args.action=='archive' else restore(args.folder,args.index)))
