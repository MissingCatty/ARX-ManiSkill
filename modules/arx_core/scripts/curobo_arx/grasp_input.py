"""Capture the ARX head RGB-D and prepare an existing tabletop object for ZeroGrasp."""
import json,time,os
from pathlib import Path
from PIL import Image
from scipy import ndimage
import httpx
import numpy as np
import yaml
from client import Client

target=os.environ.get('GRASP_TARGET','red_cube')
channels={'red_cube':0,'green_sphere':1,'blue_cylinder':2,'test_object':None}
assert target in channels,f'Unsupported target: {target}'
root=Path(__file__).resolve().parents[2];out=Path(os.environ.get('GRASP_OUTPUT_DIR',root/'runs/arx_grasp')).resolve();capture=out/'camera';out.mkdir(parents=True,exist_ok=True)
if capture.exists():capture.rename(out/f'camera_previous_{time.time_ns()}')
with httpx.Client(base_url=os.environ.get('ARX_OPS_URL','http://127.0.0.1:8101'),timeout=120,trust_env=False) as http:
    response=http.post('/camera/capture-rgbd',json={'camera_id':'head','output_dir':str(capture)})
    response.raise_for_status()
    assert 'color_path' in response.json(),response.json()
snap=Client(root/'runs/arx_curobo/mailbox').request('snapshot')
(out/'capture_snapshot.json').write_text(json.dumps(snap,indent=2))
params=json.loads((capture/'params.json').read_text());k=params['intrinsics']
rgb=np.array(Image.open(capture/'color.png').convert('RGB'),dtype=float);depth=np.array(Image.open(capture/'depth.png'))
cam=np.array(snap['camera_world']['head'])@np.diag([1,-1,-1,1])
obj=np.linalg.inv(cam)@np.array(snap['objects'][target]['world'])[:,3]
u=k['fx']*obj[0]/obj[2]+k['ppx'];v=k['fy']*obj[1]/obj[2]+k['ppy']
assert obj[2]>0 and 0<=u<rgb.shape[1] and 0<=v<rgb.shape[0],f'{target} outside camera view'
if target=='test_object':
    # Ground-truth pose and bounds isolate the selected asset; RGB-D still feeds ZeroGrasp.
    size=np.array(json.loads((out/'active_asset.json').read_text())['size_m'])
    yy,xx=np.indices(depth.shape);z=depth*params['depth_scale']
    points=np.stack([(xx-k['ppx'])/k['fx']*z,(yy-k['ppy'])/k['fy']*z,z,np.ones_like(z)],axis=-1)
    transform=np.linalg.inv(np.array(snap['objects'][target]['world']))@cam
    local=points@transform.T
    mask=(np.abs(local[:,:,:3])<=size/2+.004).all(axis=2)&(local[:,:,2]>-size[2]/2+.004)
elif target=='blue_cylinder':
    mask=(rgb[:,:,2]>1.5*rgb[:,:,0])&(rgb[:,:,2]>.95*rgb[:,:,1])&(rgb[:,:,2]>50)
elif target=='green_sphere':
    mask=(rgb[:,:,1]>1.3*rgb[:,:,0])&(rgb[:,:,1]>1.05*rgb[:,:,2])&(rgb[:,:,1]>50)
else:
    mask=(rgb[:,:,0]>1.6*rgb[:,:,1])&(rgb[:,:,0]>1.6*rgb[:,:,2])&(rgb[:,:,0]>50)
# Use the known simulator target position to disambiguate same-color scenery.
mask &= (depth>0)&(np.abs(depth*params['depth_scale']-obj[2])<(.20 if target=='test_object' else .10))
labels,n=ndimage.label(mask)
candidates=[]
for idx in range(1,n+1):
    ys,xs=np.where(labels==idx)
    if len(xs)>50:candidates.append((float(np.hypot(xs.mean()-u,ys.mean()-v)),idx))
assert candidates,f'No {target} mask visible in ARX camera'
err,idx=min(candidates);mask=(labels==idx).astype(np.uint8)
ys,xs=np.where(mask);centroid=np.array([xs.mean(),ys.mean()])
assert err<30,f'Camera frame/mask mismatch: {obj}, pixel error {err}'
Image.fromarray(mask).save(capture/'mask.png')
meta={'width':k['width'],'height':k['height'],'rectified_width':k['width'],'rectified_height':k['height'],
      'left_p':[k['fx'],0,k['ppx'],0,0,k['fy'],k['ppy'],0,0,0,1,0]}
(capture/'camera.yml').write_text(yaml.safe_dump(meta))
(out/'camera_world_optical.json').write_text(json.dumps(cam.tolist()))
report={'target':target,'mask_pixels':int(mask.sum()),'mask_centroid':centroid.tolist(),'projected_object_center':[float(u),float(v)],'projection_error_pixels':err,'median_depth_m':float(np.median(depth[mask>0]))*params['depth_scale'],'object_center_camera':obj[:3].tolist()}
(out/'input_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
