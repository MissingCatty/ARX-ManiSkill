"""Run each local tabletop asset in ARX and encode the unchanged viewport."""
from pathlib import Path
import argparse,json,os,sys,time,shutil,subprocess,signal
import imageio_ffmpeg
from client import Client
root=Path(__file__).resolve().parents[2];os.chdir(root)
parser=argparse.ArgumentParser();parser.add_argument('--output',default='runs/tabletop_trials');parser.add_argument('--names',nargs='*');parser.add_argument('--timeout',type=int,default=600);parser.add_argument('--flat-remote',action='store_true');parser.add_argument('--thin-grasp',action='store_true');parser.add_argument('--remote-near',action='store_true');parser.add_argument('--planner',choices=['prm','prm_trajopt'],default='prm');args=parser.parse_args()
r=root/'runs/arx_grasp';out=root/args.output;out.mkdir(parents=True,exist_ok=True);client=Client(root/'runs/arx_curobo/mailbox')
if not (out/'initial_setup.json').exists():shutil.copy2(root/'runs/tabletop_trials/initial_setup.json',out/'initial_setup.json')
catalog=json.loads((r/'tabletop_catalog.json').read_text());results=[]
if args.names:catalog=[a for a in catalog if a['name'] in args.names]
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
def encode(directory,dest):
 info=json.loads((directory/'frames.json').read_text());frames=[f for f in info['frames'] if (directory/f['file']).exists()]
 if info.get('error'):raise RuntimeError(info['error'])
 if len(frames)<3:raise RuntimeError('Too few captured viewport frames')
 lines=[]
 for i,f in enumerate(frames):
  lines.append("file '"+str(directory/f['file'])+"'")
  duration=frames[i+1]['wall_time']-f['wall_time'] if i+1<len(frames) else .2
  lines.append('duration '+str(max(.001,duration)))
 lines.append("file '"+str(directory/frames[-1]['file'])+"'")
 manifest=directory/'concat.txt';manifest.write_text('\n'.join(lines)+'\n')
 subprocess.run([ffmpeg,'-y','-hide_banner','-loglevel','error','-f','concat','-safe','0','-i',str(manifest),'-an','-vf','fps=5','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True)
 return {'frames':len(frames),'missing_frames':len(info['frames'])-len(frames),'camera_path':info['camera_path'],'video':str(dest)}
for index,asset in enumerate(catalog,1):
 folder=out/f"{index:02d}_{asset['name']}";folder.mkdir(exist_ok=False)
 result={'name':asset['name'],'asset':asset,'success':False,'requested_planner':args.planner};print('START',asset['name'],flush=True)
 try:
  shutil.copy2(out/'initial_setup.json',r/'capture_snapshot.json');client.request('grasp_reset_setup')
  for name,pos in [('blue_cylinder',[4.33,1.79,.76]),('green_sphere',[4.46,1.96,.76]),('red_cube',[4.22,1.75,.73])]:client.request('place_object',name=name,position=pos)
  (r/'active_asset.json').write_text(json.dumps(asset,indent=2))
  placement={'position':[4.048,2.08,.679+asset['size_m'][2]/2+.012]}
  if args.flat_remote and asset['name']=='remote_control':
   placement={'position':[4.048,2.08,.679+asset['size_m'][1]/2+.004],'orientation':[2**-.5,2**-.5,0.,0.]}
  if args.remote_near and asset['name']=='remote_control':
   assert args.flat_remote,'--remote-near requires --flat-remote'
   placement['position'][1]=2.14;placement['orientation']=[.5,.5,.5,.5]
  result['placement']=placement
  client.request('grasp_asset_load',asset_path=asset['container_path'],size_m=asset['size_m'],**placement)
  time.sleep(2);client.request('grasp_asset_bind');time.sleep(2)
  video_dir='/workspace/COOLAS-ARX/'+str(folder.relative_to(root)/'frames')
  env=os.environ.copy();env.update(GRASP_PLANNER=args.planner,GRASP_TARGET='test_object',GRASP_VIDEO_DIR=video_dir,GPU='1',CUROBO_GPU='4')
  if args.thin_grasp:env.update(GRASP_CONTACT_HEIGHT_OFFSET='.006',GRASP_CONTACT_RADIUS='.035')
  # Previous trials are archived; remove their stage artifacts from the live scratch directory.
  for pattern in ('*_optimization.json','*_command.json','*_execution.json','*_settling.json','*_prm_seed.json'):
   for stale in r.glob(pattern):stale.unlink()
  for name in ('selected_grasp.json','closed_snapshot.json','candidate_diagnostics.json','trajopt_effective_config.json'):
   (r/name).unlink(missing_ok=True)
  (r/'grasp_report.json').write_text(json.dumps({'success':False,'status':'input_pending'}))
  with (folder/'run.log').open('w') as log:
   process=subprocess.Popen(['bash','scripts/curobo_arx/run_grasp.sh'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=process.wait(timeout=args.timeout)
   except subprocess.TimeoutExpired:
    os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=15);raise RuntimeError(f'Trial exceeded {args.timeout} seconds')
  result.update(json.loads((r/'grasp_report.json').read_text()));result['process_exit_code']=code
 except Exception as exc:result.update(success=False,error=str(exc))
 finally:
  try:
   stopped=client.request('grasp_video_stop')
   if not stopped.get('frames'):
    client.request('grasp_video_start',directory='/workspace/COOLAS-ARX/'+str(folder.relative_to(root)/'frames'));time.sleep(3);client.request('grasp_video_stop')
   time.sleep(2);result['recording']=encode(folder/'frames',folder/(asset['name']+'.mp4'))
  except Exception as exc:result['video_error']=str(exc)
  for p in r.iterdir():
   if p.is_file() and p.suffix=='.json':shutil.copy2(p,folder/p.name)
  for name in ['camera','zerograsp']:
   if (r/name).exists():shutil.copytree(r/name,folder/name,dirs_exist_ok=True)
  (folder/'result.json').write_text(json.dumps(result,indent=2));results.append(result);(out/'results.json').write_text(json.dumps(results,indent=2))
  print('RESULT',asset['name'],json.dumps({k:result.get(k) for k in ['success','error','object_held','video_error']}),flush=True)
  if 'Mailbox' in result.get('error','') or 'Timed out waiting for ARX' in result.get('error',''):raise RuntimeError('Simulation unavailable; stop batch')
print('ALL_TRIALS_COMPLETE',flush=True)
