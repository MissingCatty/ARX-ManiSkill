"""Project saved candidate grasps onto the original head and overview images."""
from arx_project.assets import load_object_catalog
from itertools import product
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
RUN=PROJECT/'experiments/repaired_grasp_benchmark/runs/geometry200_repaired_20261123_01'
DOC=PROJECT/'docs/diverse_grasp_benchmark/debug_20261123'
OUT=DOC/'resources/unreachable_candidates'
SITE=PROJECT.parent/'claro/进度网站'
FONT=PROJECT.parent/'tmp/arx-video-assets/NotoSansCJKsc-Regular.otf'
SLOTS=[('D0007_S1_open_banana','香蕉','banana'),('D0018_S1_open_boxed_food','盒装食品','boxed-food'),
       ('D0035_S1_open_donut','甜甜圈','donut')]
RED=(240,62,52,255)


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def font(size):return ImageFont.truetype(str(FONT),size)


def camera_overview(state):
    base=np.array(state['links']['base_link'])[:3,3]
    eye=base+np.array([1.2,.8,1.2]);target=base+np.array([.6,-.2,.6])
    forward=target-eye;forward/=np.linalg.norm(forward)
    left=np.cross([0.,0.,1.],forward);left/=np.linalg.norm(left)
    up=np.cross(forward,left)
    t=np.eye(4);t[:3,:3]=np.column_stack([-left,-up,forward]);t[:3,3]=eye
    f=540/np.tan(.8/2)
    return t,np.array([[f,0,960],[0,f,540],[0,0,1.]])


def projection(points,t,k):
    xyz=(np.asarray(points)-t[:3,3])@t[:3,:3]
    q=xyz@k.T
    return q[:,:2]/q[:,2:],xyz[:,2]>.01


def line(draw,points,color,width=3):
    p=[tuple(float(v) for v in x) for x in points]
    draw.line(p,fill=(255,255,255,230),width=width+2)
    draw.line(p,fill=color,width=width)


def arrow(draw,a,b,color=RED):
    line(draw,[a,b],color,3)
    v=b-a;n=np.linalg.norm(v)
    if n<1:return
    v=v/n;side=np.array([-v[1],v[0]])
    draw.polygon([tuple(b),tuple(b-9*v+5*side),tuple(b-9*v-5*side)],fill=color)


def glyph(draw,candidate,t,k,params,scale=1.):
    tcp=np.array(candidate['pose']);eef=tcp.copy();eef[:3,3]-=.043*tcp[:3,2]
    mid=np.array([-.006,float(params['offset_open'][0]),float(params['offset_open'][2])])
    tip=float(params['fingertip_depth']);half=float(params['extents_open'][0])/2
    local=np.array([[mid[0],mid[1]-half,tip],[mid[0],mid[1]-half,tip-.03],
                    [mid[0],mid[1]+half,tip-.03],[mid[0],mid[1]+half,tip]])
    world=local@eef[:3,:3].T+eef[:3,3]
    uv,ok=projection(world,t,k)
    if ok.all():line(draw,uv,RED,max(2,round(3*scale)))
    endpoints=np.stack([tcp[:3,3]-.055*tcp[:3,2],tcp[:3,3]])
    points,valid=projection(endpoints,t,k)
    if valid.all():
        arrow(draw,points[0],points[1])
        u,v=points[1];draw.ellipse((u-4,v-4,u+4,v+4),fill=RED,outline='white',width=1)
    return eef


def annotate(image,t,k,pool,params,mask=None,distribution=True,target_center=None):
    image=image.convert('RGBA');layer=Image.new('RGBA',image.size);draw=ImageDraw.Draw(layer)
    if mask is not None:
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            xy=contour.reshape(-1,2)
            if len(xy)>2:draw.line([tuple(p) for p in np.vstack([xy,xy[0]])],fill=(56,230,93,230),width=2)
    if distribution:
        points=np.array([c['pose'] for c in pool])[:,:3,3]
        uv,ok=projection(points,t,k)
        for (u,v),valid in zip(uv,ok):
            if valid and 0<=u<image.width and 0<=v<image.height:
                draw.ellipse((u-1,v-1,u+1,v+1),fill=(0,215,245,65))
    if target_center is not None:
        q,_=projection([target_center],t,k);u,v=q[0]
        draw.ellipse((u-15,v-15,u+15,v+15),outline=(60,240,100,255),width=3)
    eef=glyph(draw,pool[0],t,k,params)
    return Image.alpha_composite(image,layer).convert('RGB'),eef


def crop_region(target_uv,extra_uv,size):
    pts=np.concatenate([target_uv,extra_uv])
    lo,hi=pts.min(0),pts.max(0);center=(lo+hi)/2
    width=max(140.,float((hi-lo)[0]+55),float((hi-lo)[1]+55)*4/3)
    width=min(width,float(size[0]));height=width*3/4
    x=float(np.clip(center[0]-width/2,0,size[0]-width));y=float(np.clip(center[1]-height/2,0,size[1]-height))
    return tuple(int(round(v)) for v in [x,y,x+width,y+height])


def sheet(raw,t,k,pool,params,crop,name):
    canvas=Image.new('RGB',(1920,700),'#f2f5f7');draw=ImageDraw.Draw(canvas)
    draw.text((16,10),name+' · 原生评分前10个候选',font=font(26),fill='#152e3e')
    for i,c in enumerate(pool[:10]):
        marked,_=annotate(raw,t,k,[c],params,distribution=False)
        tile=marked.crop(crop).resize((384,288),Image.Resampling.LANCZOS)
        x=(i%5)*384;y=52+(i//5)*320
        canvas.paste(tile,(x,y+30))
        draw.text((x+9,y+3),f'Top{i+1}   分数 {c["score"]:.3f}',font=font(20),fill='#253b48')
    return canvas


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    desc=read(RUN/'gripper_descriptor.json');params=desc['physical_eef_parameters']
    catalog=load_object_catalog(RUN/'object_catalog.json')['models']
    sys.path.insert(0,str(PROJECT/'experiments/diverse_object_scenes'))
    from workspace_geometry import reach_bound
    records=[]
    for slot,name,slug in SLOTS:
        d=RUN/'trials'/slot;state=read(d/'state.json');entry=read(d/'entry.json')
        pool=read(d/'candidates/candidate_pool.json')['candidates']
        assert not read(d/'plan_result.json')['success']
        head=Image.open(d/'camera/color.png').convert('RGB');overview=Image.open(d/'initial_overview.png').convert('RGB')
        assert overview.size==(1920,1080) and head.size==(640,480)
        head_t=np.array(state['camera_world_optical']);head_k=np.array(state['intrinsics'])
        third_t,third_k=camera_overview(state)
        mask=np.array(Image.open(d/'camera/mask.png'))>0
        info=catalog[entry['model_id']];box=np.array(list(product(*zip(info['bbox']['min'],info['bbox']['max']))))
        obj=np.array(state['object_world']);box=box@obj[:3,:3].T+obj[:3,3]
        center=np.array(state['object_center'])
        head_box,_=projection(box,head_t,head_k);third_box,_=projection(box,third_t,third_k)
        points=np.array([c['pose'] for c in pool[:10]])[:,:3,3]
        hp,_=projection(points,head_t,head_k);tp,_=projection(points,third_t,third_k)
        head_crop=crop_region(head_box,hp,head.size)
        third_crop=crop_region(third_box,tp,overview.size)
        hmarked,eef=annotate(head,head_t,head_k,pool,params,mask=mask)
        tmarked,_=annotate(overview,third_t,third_k,pool,params,target_center=center)
        pivot,radius=reach_bound(state,PROJECT.parent/'arx_core/scripts/curobo_arx/static_pick/reference/model/arx.urdf')
        shoulder,_=projection([pivot],third_t,third_k);palm,_=projection([eef[:3,3]],third_t,third_k)
        draw=ImageDraw.Draw(tmarked)
        a,b=shoulder[0],palm[0]
        for f in np.arange(0,1,.035):
            v=a+(b-a)*f;u=a+(b-a)*min(1,f+.018);draw.line([tuple(v),tuple(u)],fill='white',width=2)
        draw.ellipse((a[0]-6,a[1]-6,a[0]+6,a[1]+6),fill='#e735b2',outline='white',width=2)
        distance=float(np.linalg.norm(eef[:3,3]-pivot))
        combined=Image.new('RGB',(2240,640),'#142737');c=ImageDraw.Draw(combined)
        c.text((18,10),f'{slot.split("_")[0]}  {name}  |  {len(pool)}个候选；红色为原生评分Top1（{pool[0]["score"]:.3f}）',font=font(26),fill='white')
        c.text((18,51),'头部主视角：全部候选位置',font=font(21),fill='#cce4f0')
        c.text((658,51),'头部局部放大：Top1夹爪位姿',font=font(21),fill='#cce4f0')
        c.text((1298,51),f'第三人称：Top1末端距右肩 {distance*100:.1f}厘米',font=font(21),fill='#cce4f0')
        combined.paste(hmarked,(0,90))
        hsingle,_=annotate(head,head_t,head_k,pool[:1],params,mask=mask,distribution=False)
        combined.paste(hsingle.crop(head_crop).resize((640,480),Image.Resampling.LANCZOS),(640,90))
        combined.paste(tmarked.resize((960,540),Image.Resampling.LANCZOS),(1280,80))
        c.text((18,597),'青点：全部候选TCP投影   红点：TCP43   红线：张开夹爪示意   红箭头：接近方向   紫点：右肩',font=font(21),fill='white')
        combined.save(OUT/(slug+'-overview.jpg'),quality=92,optimize=True)
        for label,raw,t,k,crop in [('head',head,head_t,head_k,head_crop),('thirdperson',overview,third_t,third_k,third_crop)]:
            sheet(raw,t,k,pool,params,crop,name+' · '+('头部视角' if label=='head' else '第三人称')).save(OUT/(slug+'-top10-'+label+'.jpg'),quality=91,optimize=True)
        raw_head=hsingle.resize((1280,960),Image.Resampling.LANCZOS);raw_head.save(OUT/(slug+'-head-top1.jpg'),quality=93,optimize=True)
        tmarked.save(OUT/(slug+'-thirdperson-top1.jpg'),quality=92,optimize=True)
        thumb=combined.copy();thumb.thumbnail((1120,320));thumb.save(OUT/(slug+'-thumb.jpg'),quality=84,optimize=True)
        records.append({'id':slot,'name':name,'slug':slug,'candidates':len(pool),'top1_score':pool[0]['score'],
            'top1_eef_distance_to_shoulder_m':distance,'conservative_eef_radius_m':radius,
            'head_camera_world_optical':head_t.tolist(),'head_K':head_k.tolist(),
            'overview_camera_world_optical':third_t.tolist(),'overview_K':third_k.tolist(),
            'head_crop':head_crop,'thirdperson_crop':third_crop,'top10':pool[:10],
            'source_sha256':{str(p):sha(p) for p in [d/'state.json',d/'camera/color.png',d/'initial_overview.png',d/'candidates/candidate_pool.json']}})
        print(slot,'images saved','eef shoulder cm',round(distance*100,1),flush=True)
    (OUT/'projection_manifest.json').write_text(json.dumps({'run':str(RUN),'model_inference':False,
        'new_robot_motion':False,'actual_tcp_m':.043,'cases':records},ensure_ascii=False,indent=2)+'\n')
    dest=SITE/'内容/资源/unreachable-candidates';dest.mkdir(parents=True,exist_ok=True)
    assets={}
    for f in OUT.glob('*.jpg'):
        shutil.copy2(f,dest/f.name);assets['unreachable-'+f.name]='unreachable-candidates/'+f.name
    css=dest/'review.css'
    css.write_text('body{font:16px/1.7 system-ui,sans-serif;background:#f1f5f7;color:#172e3c;margin:0}main{max-width:1180px;margin:24px auto;padding:0 18px}article{background:white;border-radius:12px;padding:18px;margin:24px 0}img{width:100%;height:auto;border-radius:8px}a{color:#076e8b}.links{display:flex;gap:18px;flex-wrap:wrap}h1{font-size:28px}h2{font-size:22px}p{margin:10px 0}')
    assets['unreachable-candidates-review.css']='unreachable-candidates/review.css'
    parts=['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三例未通过IK的抓取位姿</title><link rel="stylesheet" href="/media/unreachable-candidates-review.css"></head><body><main>',
        '<h1>香蕉、盒装食品、甜甜圈：抓取候选在哪里</h1>',
        '<p>使用本轮原始图像和实际送入规划器的候选。青点为全部候选的TCP投影；红色突出原生评分第一名，显示TCP43、张开夹爪示意和接近方向。三例均未选出可执行抓法。</p>',
        '<p>每张对照图从左到右为头部全图、头部局部放大、第三人称。点击图片放大；前10个抓法分别提供两个视角。</p>']
    for r in records:
        slug=r['slug'];url=lambda suffix:'/media/unreachable-'+slug+suffix
        parts += [f'<article><h2>{r["id"].split("_")[0]} · {html.escape(r["name"])} · {r["candidates"]}个候选</h2>',
            f'<a href="{url("-overview.jpg")}" target="_blank"><img loading="lazy" src="{url("-thumb.jpg")}" width="1120" height="320" alt="{r["name"]}抓取位姿对照"></a>',
            '<p class="links">'+''.join(f'<a href="{url(s)}" target="_blank">{label}</a>' for s,label in [('-overview.jpg','放大对照图'),('-top10-head.jpg','前10候选：头部视角'),('-top10-thirdperson.jpg','前10候选：第三人称'),('-head-top1.jpg','Top1头部原图'),('-thirdperson-top1.jpg','Top1第三人称原图')])+'</p></article>']
    parts+=['</main></body></html>'];page=dest/'review.html';page.write_text('\n'.join(parts))
    assets['unreachable-grasp-review-20261123.html']='unreachable-candidates/review.html'
    config=SITE/'内容/网站.json';web=read(config);web.setdefault('localAssets',{}).update(assets)
    config.write_text(json.dumps(web,ensure_ascii=False,indent=2)+'\n')
    md=['# 三例未通过IK的抓取位姿','','原生评分前10位来自本轮实际候选池，红色高亮Top1；实际TCP43和模型参考系换算与执行链路一致。','','[打开网页对照图](http://127.0.0.1:54390/media/unreachable-grasp-review-20261123.html)','']
    for r in records:
        slug=r['slug'];md += [f'## {r["name"]}：{r["id"]}','',f'共{r["candidates"]}个候选，Top1分数{r["top1_score"]:.3f}，Top1末端距右肩{r["top1_eef_distance_to_shoulder_m"]*100:.1f}厘米。','',f'![{r["name"]}位姿对照](resources/unreachable_candidates/{slug}-overview.jpg)','',f'[前10候选：头部](resources/unreachable_candidates/{slug}-top10-head.jpg) · [前10候选：第三人称](resources/unreachable_candidates/{slug}-top10-thirdperson.jpg)','']
    (DOC/'三例未通过IK抓取位姿.md').write_text('\n'.join(md))
    print('PAGE /media/unreachable-grasp-review-20261123.html',flush=True)


if __name__=='__main__':main()
