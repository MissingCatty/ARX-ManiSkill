"""Produce plain viewport motion clips for the two weight groups."""
from pathlib import Path
import json,subprocess
import imageio_ffmpeg
root=Path(__file__).resolve().parents[2];out=root/'runs/weight_comparison_20261015'
rows=json.loads((out/'comparison.json').read_text());ff=imageio_ffmpeg.get_ffmpeg_exe();videos=out/'videos';videos.mkdir(exist_ok=True)
for weight in (1,100):
    clips=[]
    for r in rows:
        if r['weight_multiplier']!=weight:continue
        folder=Path(r['folder']);info=json.loads((folder/'frames/frames.json').read_text());ranges=[]
        for f in folder.glob('*_execution.json'):
            e=json.loads(f.read_text());end=e['final']['sim_time'];ranges.append((end-e['sim_duration_s']-.4,end+.5))
        frames=[(i,f) for i,f in enumerate(info['frames']) if not ranges or any(a<=f['sim_time']<=b for a,b in ranges)]
        if not ranges:frames=frames[:15]
        lines=[]
        for j,(i,f) in enumerate(frames):
            path=folder/'frames'/f['file'];assert path.exists()
            duration=frames[j+1][1]['wall_time']-f['wall_time'] if j+1<len(frames) and frames[j+1][0]==i+1 else .2
            lines.extend([f"file '{path}'",f'duration {max(.001,duration)}'])
        lines.append(f"file '{path}'")
        manifest=videos/f"w{weight}_{r['name']}.txt";manifest.write_text('\n'.join(lines)+'\n');clip=manifest.with_suffix('.mp4')
        subprocess.run([ff,'-y','-v','error','-f','concat','-safe','0','-i',str(manifest),'-an','-vf','fps=5','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(clip)],check=True);clips.append(clip)
    manifest=videos/f'w{weight}_concat.txt';manifest.write_text(''.join(f"file '{p}'\n" for p in clips));dest=videos/f'weight_{weight}_all.mp4'
    subprocess.run([ff,'-y','-v','error','-f','concat','-safe','0','-i',str(manifest),'-c','copy','-an','-movflags','+faststart',str(dest)],check=True)
    subprocess.run([ff,'-v','error','-i',str(dest),'-f','null','-'],check=True)
    print(dest,flush=True)
