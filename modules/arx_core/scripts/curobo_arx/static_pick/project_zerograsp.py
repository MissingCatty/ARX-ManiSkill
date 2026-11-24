"""Project archived raw GraspNet-format poses into the original optical RGB frame."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
p=Path(sys.argv[1]).resolve();out=p/'zerograsp_projection';out.mkdir(exist_ok=True)
rows=np.load(p/'zerograsp/object_1.grasp.npy');order=np.argsort(-rows[:,0],kind='stable');k=json.loads((p/'camera/params.json').read_text())['intrinsics']
rgb=Image.open(p/'camera/color.png').convert('RGB');scale=3
records=[]
def project(points):
 assert (points[:,2]>0).all()
 return np.stack([k['fx']*points[:,0]/points[:,2]+k['ppx'],k['fy']*points[:,1]/points[:,2]+k['ppy']],axis=1)
def draw_grasp(draw,row,color):
 width,depth=float(row[1]),float(row[3]);rot=row[4:13].reshape(3,3);t=row[13:16]
 # GraspNet X: approach, Y: closing. Finger tips at translation+depth*X.
 local=np.array([[depth,-width/2,0],[-.02,-width/2,0],[-.02,width/2,0],[depth,width/2,0],[-.055,0,0],[-.02,0,0]])
 pix=project(local@rot.T+t)
 for a,b in [(0,1),(1,2),(2,3),(4,5)]:
  coords=[tuple(pix[i]*scale) for i in (a,b)]
  draw.line(coords,fill=(15,15,15),width=4*scale)
  draw.line(coords,fill=color,width=2*scale)
 center=project((t+depth*rot[:,0])[None])[0]*scale
 draw.ellipse((center[0]-3*scale,center[1]-3*scale,center[0]+3*scale,center[1]+3*scale),fill=color)
 return pix.tolist()
for n in (1,20):
 im=rgb.resize((rgb.width*scale,rgb.height*scale));draw=ImageDraw.Draw(im)
 selected=order[:n]
 for rank in reversed(range(len(selected))):
  i=int(selected[rank]);color=(0,255,110) if rank==0 else (255,int(80+140*rank/max(1,n-1)),40)
  pix=draw_grasp(draw,rows[i],color)
  if n==20:records.append({'rank':rank+1,'raw_row_index':i,'score':float(rows[i,0]),'width_m':float(rows[i,1]),'depth_m':float(rows[i,3]),'projected_gripper_vertices_uv':pix})
 im.resize(rgb.size,Image.Resampling.LANCZOS).save(out/f'top{n}_rgb.png')
(out/'projection.json').write_text(json.dumps({'input':str(p/'camera/color.png'),'poses':str(p/'zerograsp/object_1.grasp.npy'),'intrinsics':k,'scope':'Raw ZeroGrasp predictions, sorted by score; not IK/collision/execution-valid grasps. No pose refinement, no manual coordinate adjustment. Green is rank 1. X is approach and Y is jaw closing. Overlay not occlusion culled.','gripper_glyph':'U shaped centerline, 20mm rear bridge and 55mm stem behind network translation; endpoints use predicted depth and width','total_predictions':len(rows),'displayed':sorted(records,key=lambda x:x['rank'])},indent=2))
print(out)
