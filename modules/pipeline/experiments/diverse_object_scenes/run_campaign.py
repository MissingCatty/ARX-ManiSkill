"""Run the authorized asset/scene preparation once, keeping explicit stage receipts."""
from pathlib import Path
from datetime import datetime
import os
import subprocess
import sys
import time
import traceback

from asset_catalog import HERE,PROJECT,CATALOG,read,save,sha
from pipeline import SELECTED,DEFAULT

def main():
 root=DEFAULT;started=time.perf_counter()
 deadline=time.monotonic()+1200
 while not CATALOG.exists():
  if time.monotonic()>deadline:raise RuntimeError('Prepared and deduplicated asset catalog is not ready')
  save(HERE/'campaign_progress.json',{'stage':'waiting_for_asset_preparation','physical_grasps':0})
  time.sleep(5)
 stages=[('asset_preflight',[str(HERE/'pipeline.py'),'preflight'],SELECTED),
   ('register_scenes',[str(HERE/'pipeline.py'),'prepare','--root',str(root)],root/'registration.json'),
   ('build_scenes',[str(HERE/'pipeline.py'),'build','--root',str(root),'--resume'],root/'manifest.json'),
   ('verify',[str(HERE/'verify_dataset.py'),str(root)],root/'verification.json')]
 receipts=[]
 try:
  for stage,args,completion in stages:
   if completion.exists():
    receipts.append({'stage':stage,'already_complete':True,'file_sha256':sha(completion)});continue
   save(HERE/'campaign_progress.json',{'stage':stage,'at':datetime.now().astimezone().isoformat(),'physical_grasps':0,'receipts':receipts})
   print('STAGE',stage,flush=True);tick=time.perf_counter()
   result=subprocess.run([sys.executable,*args],cwd=PROJECT,env=os.environ.copy())
   receipts.append({'stage':stage,'seconds':time.perf_counter()-tick,'returncode':result.returncode})
   if result.returncode:raise RuntimeError(f'{stage} exited {result.returncode}')
  assert read(root/'verification.json')['passed']
  active={'dataset':str(root),'object_catalog':str(root/'object_catalog.json'),'environment_id':'ARXPickCatalog-v1',
    'adapter':str(HERE/'asset_environment.py'),'runtime_factory':str(HERE/'pipeline.py')+':runtime',
    'scenes':200,'categories':40,'models':80,'state':'verified','grasp_tests_authorized':False}
  save(HERE/'active_dataset.json',active)
  save(PROJECT/'configs/scene_dataset.json',active)
  mainline=read(PROJECT/'configs/mainline.json');mainline['scene_dataset_config']='configs/scene_dataset.json'
  save(PROJECT/'configs/mainline.json',mainline)
  save(HERE/'campaign_progress.json',{'stage':'complete','passed':True,'wall_seconds':time.perf_counter()-started,
    'at':datetime.now().astimezone().isoformat(),'physical_grasps':0,'receipts':receipts})
  print('COMPLETE',root,flush=True)
 except BaseException:
  save(HERE/'campaign_halt.json',{'at':datetime.now().astimezone().isoformat(),'receipts':receipts,'error':traceback.format_exc(),'physical_grasps':0});raise

if __name__=='__main__':main()
