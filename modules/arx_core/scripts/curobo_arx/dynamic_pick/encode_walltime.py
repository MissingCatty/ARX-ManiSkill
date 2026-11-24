import argparse,json,time,subprocess
from pathlib import Path
import numpy as np
import imageio_ffmpeg
from PIL import Image
parser=argparse.ArgumentParser();parser.add_argument("folder",type=Path);args=parser.parse_args()
folder=args.folder.resolve();frames=folder/"frames";info=json.loads((frames/"frames.json").read_text());assert not info.get("error") and len(info["frames"])>1
view_path=folder/"viewport_initial.json"
if not view_path.exists():view_path=folder.parent/"viewport.json"
reference=json.loads(view_path.read_text());assert info["camera_path"]==reference["camera_path"];assert np.allclose(info["camera_matrix"],reference["world_matrix"],atol=1e-8)
end=time.monotonic()+30
while not all((frames/f["file"]).exists() for f in info["frames"]):
 if time.monotonic()>end:raise TimeoutError("Recorded capture futures incomplete")
 time.sleep(.2)
rows=[];items=info["frames"];dtlast=float(np.median(np.diff([f["wall_time"] for f in items])[-10:]))
for i,item in enumerate(items):
 path=frames/item["file"];dt=items[i+1]["wall_time"]-item["wall_time"] if i+1<len(items) else dtlast
 rows.extend(["file '"+str(path)+"'","duration "+str(max(.001,dt))])
rows.append("file '"+str(path)+"'");manifest=folder/"walltime_concat.txt";manifest.write_text("\n".join(rows)+"\n")
ff=imageio_ffmpeg.get_ffmpeg_exe();video=folder/"dynamic_obstacle_grasp.mp4";assert not video.exists()
subprocess.run([ff,"-v","error","-f","concat","-safe","0","-i",str(manifest),"-an","-map_metadata","-1","-vf","fps=5","-c:v","libx264","-crf","20","-threads","2","-pix_fmt","yuv420p","-movflags","+faststart",str(video)],check=True,timeout=180)
subprocess.run([ff,"-v","error","-i",str(video),"-f","null","-"],check=True,timeout=180)
Image.open(path).save(folder/"final_frame.png")
report={"video":str(video),"recording_start_view_verified":True,"original_view_verified":bool(np.allclose(reference["world_matrix"],json.loads((folder.parent/"viewport.json").read_text())["world_matrix"],atol=1e-8)),"source_frames":len(items),"wall_elapsed_s":items[-1]["wall_time"]-items[0]["wall_time"],"sim_elapsed_s":items[-1]["sim_time"]-items[0]["sim_time"],"playback_timing":"recorded wall-time; no acceleration","text_added":False,"audio_added":False,"full_decode":"PASS"};(folder/"video_report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report))
