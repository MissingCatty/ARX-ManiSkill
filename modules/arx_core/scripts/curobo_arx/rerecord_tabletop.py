"""Re-execute archived ARX commands with physics and record the current viewport."""
import argparse,json,sys,time,shutil,subprocess
from pathlib import Path
import numpy as np
import imageio_ffmpeg
from client import Client
from settling import assess_settling
root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser();parser.add_argument('output');parser.add_argument('--source',default='runs/tabletop_trials');parser.add_argument('--names',nargs='*');parser.add_argument('--gains',nargs=2,type=float);args=parser.parse_args()
source=root/args.source;out=root/args.output
c=Client(root/'runs/arx_curobo/mailbox');ff=imageio_ffmpeg.get_ffmpeg_exe()
reference=json.loads((out/'view_probe/frames.json').read_text());results=[]
if args.gains:c.request('grasp_arm_gains',stiffness=args.gains[0],damping=args.gains[1])
def encode(folder,name):
 info=json.loads((folder/'frames/frames.json').read_text())
 assert not info.get('error'),info.get('error')
 assert info['camera_path']==reference['camera_path']
 assert np.allclose(info['camera_matrix'],reference['camera_matrix'],atol=1e-8)
 frames=info['frames'];assert len(frames)>2
 manifest=[]
 for i,f in enumerate(frames):
  path=folder/'frames'/f['file'];assert path.exists(),path
  dt=frames[i+1]['wall_time']-f['wall_time'] if i+1<len(frames) else .2
  manifest.extend([f"file '{path}'",f'duration {max(.001,dt)}'])
 manifest.append(f"file '{path}'")
 m=folder/'concat.txt';m.write_text('\n'.join(manifest)+'\n')
 p=folder/(name+'.mp4')
 subprocess.run([ff,'-y','-v','error','-f','concat','-safe','0','-i',str(m),'-an','-vf','fps=5','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(p)],check=True)
 return {'video':str(p),'frames':len(frames),'camera_path':info['camera_path']}
for i,old in enumerate(json.loads((source/'results.json').read_text()),1):
 name=old['name']
 if args.names and name not in args.names:continue
 asset=old['asset'];previous=source/f'{i:02d}_{name}'
 folder=out/f'{i:02d}_{name}';folder.mkdir()
 r={'name':name,'mode':'reexecute_archived_commands_in_physx','source':str(previous),'success':False,'stages':{}}
 print('START',name,flush=True)
 recording=False
 try:
  shutil.copy2(source/'initial_setup.json',root/'runs/arx_grasp/capture_snapshot.json');c.request('grasp_reset_setup')
  for n,p in [('blue_cylinder',[4.33,1.79,.76]),('green_sphere',[4.46,1.96,.76]),('red_cube',[4.22,1.75,.73])]:c.request('place_object',name=n,position=p)
  c.request('grasp_asset_load',asset_path=asset['container_path'],size_m=asset['size_m'],position=[4.048,2.08,.679+asset['size_m'][2]/2+.012])
  time.sleep(2);c.request('grasp_asset_bind');time.sleep(2)
  snap=c.request('snapshot');(folder/'initial_snapshot.json').write_text(json.dumps(snap))
  saved=json.loads((previous/'capture_snapshot.json').read_text())
  assert np.allclose(snap['link_world']['body4'],saved['link_world']['body4'],atol=.005),'Robot base changed'
  initial=np.array(snap['objects']['test_object']['world']);expected=np.array(saved['objects']['test_object']['world'])
  assert np.linalg.norm(initial[:3,3]-expected[:3,3])<.01,'Object start differs from archived scene'
  target=snap['objects']['test_object']['path'];initial_z=initial[2,3]
  c.request('grasp_video_start',directory='/workspace/COOLAS-ARX/'+str(folder.relative_to(root)/'frames'));recording=True
  stages=old.get('stage_convergence',{})
  if not stages:
   r['error']='No archived executable trajectory; scene-only clip';time.sleep(3)
  for label in ('pregrasp','approach','close','lift','hold'):
   if label not in stages:break
   command=json.loads((previous/f'{label}_command.json').read_text())
   print('EXECUTE',name,label,flush=True)
   result=c.request('trajectory',timeout=60,**command)
   (folder/f'{label}_execution.json').write_text(json.dumps(result))
   (folder/f'{label}_command.json').write_text(json.dumps(command))
   external=[x for x in result['contacts'] if x['num_contacts']>0 and not ('/World/Robot/' in x['a'] and '/World/Robot/' in x['b'])]
   unwanted=[x for x in external if not(target in x['a']+x['b'] and any(n in x['a']+x['b'] for n in ('right_link27','right_link28')))]
   selection=json.loads((previous/'selected_grasp.json').read_text())
   key={'pregrasp':'pregrasp_world','approach':'grasp_world','close':'grasp_world','lift':'lift_world','hold':'lift_world'}[label]
   assessment=assess_settling(result,selection[key])
   r['stages'][label]={**assessment,'settled':assessment['accepted'],'unwanted_contacts':len(unwanted)}
   assert not unwanted,'Unexpected contact during '+label
   if label in ('pregrasp','approach','close') and not assessment['accepted']:
    r['error']=label+' did not converge; stopped';break
   if label=='hold':
    heights=[np.array(x['objects']['test_object'])[2,3] for x in result['records']]
    r.update(object_held=bool(min(heights)>initial_z+.06),lift_m=float(heights[-1]-initial_z))
    r['success']=r['object_held'] and all(s['settled'] for s in r['stages'].values())
 except Exception as exc:r['error']=str(exc)
 finally:
  if recording:
   c.request('grasp_video_stop');time.sleep(2);r['recording']=encode(folder,name)
  (folder/'result.json').write_text(json.dumps(r,indent=2));results.append(r);(out/'results.json').write_text(json.dumps(results,indent=2))
  print('RESULT',json.dumps({k:v for k,v in r.items() if k!='stages'}),flush=True)
  if not recording:raise RuntimeError('Setup failed: '+r.get('error','unknown'))
print('ALL_COMPLETE',flush=True)
