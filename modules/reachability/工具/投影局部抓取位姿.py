"""CPU projection of original/refined grasp ghosts on archived two-view images."""
import argparse
import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

from 诊断七轴抓取终点 import ROOT,MS,read,save,sha

PATH=ROOT.parent/'claro/工具/投影七例挡物规划终点.py'
spec=importlib.util.spec_from_file_location('archived_projection',PATH)
vis=importlib.util.module_from_spec(spec);spec.loader.exec_module(vis)

def draw(raw,pose,contact,width,geometry,camera,k,color):
    image=np.array(raw,copy=True);vis.ghost_mesh(image,pose,geometry,camera,k,color,opacity=.12,width=max(2,int(image.shape[1]/500)))
    point=np.asarray(contact)[:3,3];axis=np.asarray(contact)[:3,1]
    contacts=np.stack([point-width/2*axis,point+width/2*axis]);uv,depths=vis.project(contacts,camera,k)
    for i,(pixel,depth) in enumerate(zip(uv,depths)):
        if depth>.02:
            xy=tuple(np.round(pixel).astype(int));cv2.circle(image,xy,max(4,int(image.shape[1]/180)),(255,230,70),-1,cv2.LINE_AA)
            cv2.putText(image,'P'+str(i+1),(xy[0]+7,xy[1]-7),cv2.FONT_HERSHEY_SIMPLEX,max(.45,image.shape[1]/1800),(255,250,220),2,cv2.LINE_AA)
    for i,col in enumerate(vis.AXES):
        p=np.stack([pose[:3,3],pose[:3,3]+.035*pose[:3,i]])
        pixels,z=vis.project(p,camera,k)
        if (z>.02).all():
            a,b=[tuple(np.round(v).astype(int)) for v in pixels]
            cv2.arrowedLine(image,a,b,col,max(2,int(image.shape[1]/700)),cv2.LINE_AA,tipLength=.12)
    return image,{'nominal_contact_points_world':contacts.tolist(),
                  'contact_pixels':uv.tolist(),'contact_optical_depth_m':depths.tolist()}

def main(root,destination):
    reg=read(root/'登记.json');destination.mkdir(parents=True,exist_ok=True);evidence=[]
    for case in reg['cases']:
        if case['role']!='diagnostic_failure':continue
        file=root/'候选'/f'{case["case"]}.json'
        if not file.exists():continue
        variants=read(file)['variants'];chosen=None;label='地图优先建议，尚未通过完整规划'
        result_path=root/'比较'/case['case']/'map_local_search/结果.json'
        if result_path.exists():
            result=read(result_path)
            if result['success']:chosen=result['selected']['variant'];label='已通过离线规划，未实际抓取'
        if chosen is None:
            candidates=[v for v in variants if v['geometry_admitted'] and v['angle_deg']!=0]
            if not candidates:candidates=[v for v in variants if v['angle_deg']!=0]
            chosen=min(candidates,key=lambda v:v['map_score'])
        original=next(v for v in variants if v['source_index']==chosen['source_index'] and v['angle_deg']==0)
        source=Path(case['source']);state=read(source/'state.json');geometry,error=vis.meshes_in_eef(state)
        wrist=np.asarray(Image.open(source/'camera/color.png').convert('RGB'))
        third,frame,count=vis.last_frame(source/'execution.mp4');definition=read(source/'recording_camera.json')
        cameras=[('腕部原始图像',wrist,np.asarray(state['camera_world_optical']),np.asarray(state['intrinsics'])),
                 ('第三人称录像末帧',third,*vis.third_camera(state,definition))]
        canvas=Image.new('RGB',(1280,1080),'white')
        vis.text(canvas,(24,12),case['case']+'  抓取终点局部修正',size=30)
        vis.text(canvas,(24,55),'橙／蓝轮廓是虚拟终点夹爪；黄色 P1/P2 是候选名义接触点',size=20)
        vis.text(canvas,(24,86),label+'；没有新增机器人动作',size=20)
        record={'case':case['case'],'source_case':case['case_id'],'original':original,'proposal':chosen,
            'status':label,'third_video_frame_index':frame,'third_video_frames':count,'source_fk_max_error':error,'projections':[]}
        for row,(title,raw,camera,k) in enumerate(cameras):
            y=130+row*460
            vis.text(canvas,(24,y),title,size=22)
            for column,(variant,color,caption) in enumerate([(original,(245,127,28),'原位姿'),(chosen,(40,154,246),f'修正建议 {chosen["angle_deg"]:+.0f}°')]):
                image,proof=draw(raw,np.asarray(variant['eef_goal_world']),variant['contact_pose_world'],variant['width'],geometry,camera,k,color)
                picture=Image.fromarray(image);picture.thumbnail((620,390))
                x=10+column*640;canvas.paste(picture,(x+(620-picture.width)//2,y+35))
                vis.text(canvas,(x+14,y+430),caption+f'  · 指腹区观测点 {variant["observed_actual_pad_corridor_points"]}',size=18)
                record['projections'].append({'view':title,'variant':variant['id'],'camera_world_optical':camera.tolist(),'intrinsics':k.tolist(),**proof})
        canvas.save(destination/(case['case']+'_双视角位姿对照.jpg'),quality=90)
        record['source_sha256']={str(p):sha(p) for p in [source/'state.json',source/'camera/color.png',source/'execution.mp4',source/'recording_camera.json',file,PATH]}
        evidence.append(record)
        print('PROJECTED',case['case'],chosen['angle_deg'],label,flush=True)
    save(destination/'投影依据.json',{'scope':'Archived images and virtual visual-mesh projection, not execution or collision evidence','physical_attempts':0,'cases':evidence})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.run.resolve(),a.output.resolve())
