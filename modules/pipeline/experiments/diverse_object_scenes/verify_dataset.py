"""Independent saved-frame, geometry, category and quota checks."""
from arx_project.assets import load_object_catalog
import argparse
from collections import Counter
from pathlib import Path
import hashlib
import json
import shutil
import sys

import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from asset_catalog import read,save,sha

LABELS={'apple':'苹果','banana':'香蕉','orange':'橙子','pear':'梨','peach':'桃子','lemon':'柠檬','mango':'芒果','avocado':'牛油果',
 'bell_pepper':'甜椒','carrot':'胡萝卜','corn':'玉米','cucumber':'黄瓜','eggplant':'茄子','mushroom':'蘑菇','onion':'洋葱','potato':'土豆',
 'tomato':'番茄','broccoli':'西兰花','bread':'面包','baguette':'法棍','croissant':'羊角面包','donut':'甜甜圈','canned_food':'食品罐头',
 'boxed_food':'盒装食品','bottled_water':'瓶装水','milk':'牛奶','yogurt':'酸奶','mug':'马克杯','bowl':'碗','soap_dispenser':'洗手液瓶',
 'bar_soap':'肥皂','spatula':'锅铲','scissors':'剪刀','teapot':'茶壶','candle':'蜡烛','book':'书','remote_control':'遥控器',
 'cell_phone':'手机','sponge':'海绵','keychain':'钥匙串'}

def object_gallery(catalog):
 from PIL import ImageFont
 font_path=PROJECT.parent/'claro/缓存/对比视频字体/NotoSansCJKsc-Regular.otf'
 font=ImageFont.truetype(str(font_path),19)
 sheet=Image.new('RGB',(8*180,10*205),'#f4f6f9');draw=ImageDraw.Draw(sheet)
 for i,category in enumerate(catalog['categories']):
  models=sorted(m for m,r in catalog['models'].items() if r['semantic_class']==category)
  for j,model in enumerate(models):
   index=i*2+j;x=(index%8)*180;y=(index//8)*205
   image=Image.open(HERE/'assets/preflight'/model/'thumb.jpg');sheet.paste(image,(x+10,y+8))
   draw.text((x+10,y+172),LABELS.get(category,category)+f' {j+1}',font=font,fill='#28364b')
 path=HERE/'assets/object_gallery.jpg';sheet.save(path,quality=90);return path

def verify(root):
 reg=read(root/'registration.json');manifest=read(root/'manifest.json');catalog=load_object_catalog(root/'object_catalog.json');db=catalog['models']
 assert manifest['registration_sha256']==sha(root/'registration.json')
 assert sha(root/'object_catalog.json')==reg['object_catalog_sha256']
 for path,digest in reg['source_hashes'].items():assert sha(path)==digest,path
 revisions=[]
 for path in (root/'construction_revisions').glob('*/registration.json'):
  revision=read(path);assert revision['original_registration_sha256']==sha(root/'registration.json')
  assert sha(revision['source'])==revision['source_sha256']
  for file,digest in revision['existing_case_file_hashes'].items():assert sha(file)==digest,file
  revisions.append({'file':str(path),'sha256':sha(path),'preserved_files':len(revision['existing_case_file_hashes'])})
 assert len(reg['semantic_classes'])==40 and len(db)==80
 assert Counter(r['semantic_class'] for r in db.values())==dict.fromkeys(reg['semantic_classes'],2)
 assert len({r['raw_visual_sha256'] for r in db.values()})==80
 assert len({r['source_geometry_sha256'] for r in db.values()})==80
 for model,row in db.items():
  assert read(HERE/'assets/preflight'/model/'verification.json')['passed']
  for path,digest in row['source_hashes'].items():assert sha(path)==digest,path
 records=[];keys=set();model_counts=Counter();class_counts=Counter();layouts=Counter()
 for entry in manifest['schedule']:
  folder=root/'cases'/entry['id'];admission=read(folder/'admission.json');state=read(folder/'state.json');q=entry['admission']
  assert admission['passed'] and admission['entry']==entry
  assert entry['semantic_class']==db[entry['model_id']]['semantic_class']
  assert abs(state['qpos'][state['joint_names'].index('joint4')]-.452)<.001 and state['perception_camera']=='head'
  for name,digest in admission['camera_input_sha256'].items():assert sha(folder/'camera'/name)==digest
  mask=np.asarray(Image.open(folder/'target_gt_native.png'))>0
  assert int(mask.sum())==q['visible_pixels'] and q['visible_pixels']>=64
  rawdepth=np.asarray(Image.open(folder/'camera/depth.png'))*.0001;gt=np.asarray(Image.open(folder/'target_gt.png'))>0
  assert rawdepth.shape==gt.shape==(480,640)
  valid=(rawdepth>.10)&(rawdepth<2.)&np.isfinite(rawdepth);assert float((valid&gt).sum()/gt.sum())>=.98
  if entry['layout']=='S5_occluded':assert .20<=q['occlusion_fraction']<=.40
  else:assert q['occlusion_fraction']<=.15
  camera=np.asarray(state['camera_world_optical']);k=np.asarray(state['intrinsics']);actors=entry['scene']['actors']
  lo_count,hi_count=reg['distractor_counts'][entry['layout']];assert lo_count<=len(actors)-1<=hi_count
  assert len({db[a['model_id']]['semantic_class'] for a in actors})==len(actors)
  for actor in actors:
   info=db[actor['model_id']];bounds=info['bbox'];lo,hi=np.asarray(bounds['min']),np.asarray(bounds['max'])
   pts=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
   p=np.asarray(actor['pose']);rot=Rotation.from_quat(p[[4,5,6,3]]).as_matrix();world=pts@rot.T+p[:3]
   xyz=(world-camera[:3,3])@camera[:3,:3];assert xyz[:,2].min()>.1 and xyz[:,2].max()<2.
   uv=xyz@k.T;uv=uv[:,:2]/uv[:,2:];assert (uv.min(0)>=[12,12]).all() and (uv.max(0)<=[628,468]).all()
  key=hashlib.sha256(json.dumps([(a['model_id'],np.round(a['pose'],6).tolist()) for a in actors],sort_keys=True).encode()).hexdigest()
  assert key not in keys;keys.add(key)
  model_counts[entry['model_id']]+=1;class_counts[entry['semantic_class']]+=1;layouts[entry['layout']]+=1
  records.append({'id':entry['id'],'category':entry['semantic_class'],'model':entry['model_id'],'visible_pixels':q['visible_pixels'],'occlusion_fraction':q['occlusion_fraction']})
 assert len(records)==200 and class_counts==dict.fromkeys(reg['semantic_classes'],5)
 assert layouts==reg['layouts'] and len(model_counts)==80 and set(model_counts.values())=={2,3}
 for layout in reg['layouts']:
  assert Counter(e['semantic_class'] for e in manifest['schedule'] if e['layout']==layout)==dict.fromkeys(reg['semantic_classes'],1)
 checks=catalog['preflight_reports'];asset_failures=[r for r in checks if not r['passed']]
 result={'passed':True,'verifier_source_sha256':sha(Path(__file__)),'scenes':200,'unique_scenes':len(keys),'categories':40,'models':80,'layouts':dict(layouts),
   'target_class_counts':dict(class_counts),'target_model_counts':dict(model_counts),'source_models':dict(Counter(r['source'] for r in db.values())),
   'raw_visual_duplicates':0,'asset_preflight_checked':len(checks),'asset_preflight_rejected':asset_failures,
   'construction_revisions':revisions,'physical_grasps':0,'no_grasp_or_SAM3_selection':True,'records':records}
 save(root/'verification.json',result)
 gallery=object_gallery(catalog);doc=PROJECT/'docs/diverse_object_scenes/多类别物体与场景.md';resources=doc.parent/'resource';resources.mkdir(parents=True,exist_ok=True)
 shutil.copy2(gallery,resources/'object_gallery.jpg');shutil.copy2(root/'verification.json',resources/'verification.json')
 lines=['# 多类别物体与场景','','从本地RoboCasa和AI2THOR接入40类、80个不同型号。每类有2个型号，在五种布局中各出现1次，共200个场景。单个型号出现2或3次。',
  '', '| 检查项 | 结果 |','| --- | --- |','| 物体类别 | 40类 |','| 物体型号 | 80个，源外观网格无重复 |',
  f'| 物理检查 | 核查{len(checks)}个候选，通过80个，淘汰{len(asset_failures)}个 |','| 场景 | 五类各40个，共200个 |','| 物理抓取 | 0次 |','',
  '机器人、桌面和头部D405沿用当前配置，升降固定45.2厘米。物体需完整处于头部视野内并具有有效深度；部分遮挡场景的真实几何遮挡为20%至40%。场景构造不使用SAM3、抓取模型、IK或抓取成败筛选。',
  '', 'RoboCasa保留源网格比例、密度、滑动摩擦及碰撞几何；源文件中的方盒碰撞基本体等形转换为凸网格。AI2THOR没有质量和摩擦参数，统一采用密度100 kg/m³、滑动摩擦0.5、恢复系数0的仿真假设。AI2THOR优先复用已有CoACD分解，缺少的在本目录离线分解。',
  '', f'通过的80个型号均完成加载、稳定落桌和两种平面朝向检查。被淘汰候选的逐项原因保存在核查附件中。',
  '', '## 物体种类','','![40类80个型号](resource/object_gallery.jpg)','','| 类别 | 来源 | 型号数 | 场景数 |','| --- | --- | ---: | ---: |']
 for category in reg['semantic_classes']:
  row=next(r for r in db.values() if r['semantic_class']==category);lines.append(f"| {LABELS.get(category,category)} | {row['source']} | 2 | 5 |")
 lines+=['','## 场景预览','']
 for layout,label in zip(reg['layouts'],['开阔桌面','靠近桌边','稀疏混放','中度杂乱','部分遮挡']):
  shutil.copy2(root/'gallery'/(layout+'.jpg'),resources/(layout+'.jpg'));lines += [f'### {label}','',f'![{label}40例](resource/{layout}.jpg)','']
 lines+=['## 产物位置','',f'- 物体加载接口：`{HERE}/asset_environment.py`',f'- 选定物体与源文件散列：`{root}/object_catalog.json`',f'- 200场景：`{root}`',
  '- 每例保存头部RGB-D、独立真值掩码、第三人称图、物体位姿、碰撞网格引用和构造记录。','- [独立核查结果](resource/verification.json)']
 doc.write_text('\n'.join(lines)+'\n');print('VERIFIED',json.dumps({k:v for k,v in result.items() if k not in ['records','target_class_counts','target_model_counts']}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);verify(p.parse_args().root.resolve())
