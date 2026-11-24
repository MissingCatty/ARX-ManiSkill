"""Encode the original ARX viewport, with no text, audio, or synthetic imagery."""
import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args()
    folder=a.folder.resolve();frames=folder/'frames';info=json.loads((frames/'frames.json').read_text())
    assert not info.get('error') and len(info['frames'])>1
    for f in info['frames']:assert (frames/f['file']).exists()
    reference=json.loads((folder.parent/'viewport.json').read_text())
    assert info['camera_path']==reference['camera_path']
    assert np.allclose(info['camera_matrix'],reference['world_matrix'],atol=1e-8)
    ff=imageio_ffmpeg.get_ffmpeg_exe();video=folder/'dynamic_obstacle_grasp.mp4'
    assert not video.exists()
    subprocess.run([ff,'-v','error','-framerate','5','-i',str(frames/'%06d.png'),
        '-frames:v',str(len(info['frames'])),'-map_metadata','-1','-an','-c:v','libx264',
        '-threads','2','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True,timeout=180)
    subprocess.run([ff,'-v','error','-i',str(video),'-f','null','-'],check=True,timeout=180)
    last=frames/info['frames'][-1]['file']
    Image.open(last).save(folder/'final_frame.png')
    report={'video':str(video),'frames':len(info['frames']),'fps':5,'duration_s':len(info['frames'])/5,
        'sim_elapsed_s':info['frames'][-1]['sim_time']-info['frames'][0]['sim_time'],
        'wall_elapsed_s':info['frames'][-1]['wall_time']-info['frames'][0]['wall_time'],
        'original_user_view_verified':True,'text_added':False,'audio_added':False,'full_decode':'PASS'}
    (folder/'video_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
