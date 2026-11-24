"""只读原始批次，汇总45.2厘米两阶段实验并生成可追溯的十场景图集。"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess

SITE=Path(__file__).resolve().parents[1]
MS=SITE.parents[1]/'ManiSkill'
RUN=MS/'outputs/arx_graspcl_compat160_20261120_01'
OLD=MS/'outputs/arx_selected_height_pair_20261120_01'
LOW=MS/'outputs/arx_physical_table_parallel_20261118_01'
DEST=SITE/'内容/资源/45厘米高度实验'
ARMS=['fixed_zerograsp','fixed_graspcl']
LAYOUTS={'S1_open':'开阔桌面','S2_edge':'靠近桌边','S3_sparse':'稀疏混放','S4_clutter':'中度杂乱'}
REASONS={
    'success':'成功', 'no_feasible_native_post_or_recalled_candidate':'未找到可执行初始方案',
    'insufficient_hold_height':'抬升保持高度不足','no_bilateral_target_contact':'未形成双指目标接触',
    'approach_not_converged':'接近未收敛','close_not_converged':'闭爪未收敛',
    'lift_not_converged':'抬升未收敛','forbidden_robot_contact':'触发禁止接触',
    'independent_scene_admission_failed':'原登记初始化未准入',
    'registered_occlusion_outside_range':'遮挡分组未准入','registered_wrist_view_not_admitted':'腕部视野未准入'}

def read(p):return json.loads(Path(p).read_text())
def save(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def results(root,arm):return {p.parent.name:read(p) for p in (root/arm/'trials').glob('*/result.json')}
def metrics(rows):
    items=list(rows.values());n=len(items)
    return {'completed':n,'planning_passes':sum(x['planning_success'] for x in items),
        'physical_attempts':sum(x['physical_attempt'] for x in items),'successes':sum(x['success'] for x in items),
        'failure_reasons':dict(Counter(x['reason'] for x in items if not x['success'])),
        'by_layout':{layout:metrics_simple([x for x in items if x['layout']==layout]) for layout in LAYOUTS}}
def metrics_simple(items):return {'n':len(items),'planning':sum(x['planning_success'] for x in items),'success':sum(x['success'] for x in items)}
def pair(a,b):
    ids=sorted(set(a)&set(b))
    return {'paired':len(ids), 'both_success':[i for i in ids if a[i]['success'] and b[i]['success']],
        'graspcl_only':[i for i in ids if b[i]['success'] and not a[i]['success']],
        'zerograsp_only':[i for i in ids if a[i]['success'] and not b[i]['success']],
        'both_failure':[i for i in ids if not a[i]['success'] and not b[i]['success']]}

def audit(final=True):
    progress=read(RUN/'progress.json')
    if final:
        end=read(RUN/'运行结束.json')
        assert progress['phase']=='complete' and progress['test_completed']==320
        assert not end['failed'] and end['all_owned_services_exited']
        assert all(x['exited'] and x['returncode']==0 for x in end['services'].values())
    for fault in ['STOP.json','控制器故障.json']:assert not (RUN/fault).exists(),fault
    counts=Counter();source_hashes={};registration_hashes={}
    bank=read(RUN/'bank_tcp43/bank_complete.json');frozen=read(RUN/'bank_frozen_before_tests.json')
    assert sha(RUN/'bank_tcp43/bank_complete.json')==frozen['bank_complete_sha256']
    assert bank['positive_demonstrations']==24 and bank['negative_feedback']==4
    assert read(RUN/'隔离核验.json')['passed'] and read(RUN/'bank_tcp43/provenance.json')['passed']
    ready=read(RUN/'resident/all_ready.json')
    assert ready['warmup_passed'] and set(ready['components'])=={'zerograsp','graspcl','curobo'}
    for group,budget in [('collection',40),*[(g,160) for g in ARMS]]:
        folder=RUN/group;reg=read(folder/'registration.json');registration_hashes[group]=sha(folder/'registration.json')
        for p,digest in {**reg['source_hashes'],**reg['asset_sha256']}.items():
            if p in source_hashes:assert source_hashes[p]==digest
            source_hashes[p]=digest
        ids={e['id'] for e in reg['schedule']}
        claim_ids={p.stem for p in (folder/'claims').glob('*.json')}
        assert len(ids)==budget and claim_ids.issubset(ids)
        if final:
            assert claim_ids==ids
            assert {p.name for p in (folder/'trials').iterdir() if p.is_dir()}==ids
            assert {p.stem for p in (folder/'claims').glob('*.reserved')}==ids
        completed_ids={id for id in claim_ids if read(folder/'claims'/(id+'.json'))['state']=='complete'}
        if final:assert completed_ids==ids
        for id in sorted(completed_ids):
            d=folder/'trials'/id;c=read(folder/'claims'/(id+'.json'));r=read(d/'result.json');e=read(d/'independent_evaluation.json')
            assert c['state']=='complete' and c['result_sha256']==sha(d/'result.json')
            assert read(d/'source_guard.json')['sha256']==registration_hashes[group]
            assert e['passed'] and e.get('independently_successful',False)==r['success']
            assert r['camera']=='right_wrist'
            if group!='collection':assert r['selected_lift_m']==.452
            assert r['orientation_mode']=='fixed_orientation' and read(d/'示教兼容核验.json')['passed']
            assert read(d/'independent_reset_match.json')['passed'] and read(d/'capture.json')['fresh_rgbd']
            entry=read(folder/'entries'/(id+'.json'));initial=read(d/'initial_conditions.json')
            assert initial['spawn']==entry['scene'] and abs(initial['initial_q'][0]-.452)<1e-6
            for name in ['execution.jsonl','execution_origin.json','execution.mp4','right_wrist.mp4',
                         'camera/color.png','camera/depth.png','camera/mask.png','camera/params.json',
                         'zg_outputs/grasp_top10.png','zg_outputs/grasp_top10_rgb.png','zg_outputs/grasp_top10.json']:
                p=d/name;assert p.is_file() and p.stat().st_size>0,(id,name)
            # Same frozen success predicate, re-read without importing the simulator or writing its results.
            z0=read(d/'execution_origin.json')['object_initial_z_m'];hold=0;minimum=float('inf');bad=0;fingers=set()
            previous_tick=-1
            for line in (d/'execution.jsonl').open():
                frame=json.loads(line);assert frame['tick']>previous_tick;previous_tick=frame['tick']
                if frame['stage']=='hold':hold+=1;minimum=min(minimum,frame['object_pose'][2]-z0)
                for contact in frame['contacts']:
                    bad+=not contact['allowed']
                    if frame['stage']=='close' and contact['allowed']:
                        fingers.update(f for f in ['right_link27','right_link28'] if f in contact['a']+contact['b'])
            stages=r.get('stages',{})
            recomputed=bool(minimum>=.01 and hold/120>=3 and bad==0 and len(fingers)==2
                and {'pregrasp','approach','close','lift','hold'}.issubset(stages)
                and all(s['settled'] for s in stages.values()))
            assert recomputed==r['success']==e['independently_successful'],id
            assert hold/120==e['hold_seconds'] and bad==e['forbidden']
            assert (minimum if hold else None)==e['minimum_lift_m']
            counts['trace_recomputed']+=1;counts['dual_videos_present']+=1
            if group=='fixed_graspcl':
                assert datetime.fromisoformat(frozen['time'])<datetime.fromisoformat(c['claimed_at'])
                assert datetime.fromisoformat(ready['ready_at'])<datetime.fromisoformat(c['claimed_at'])
                for name in ['frozen_memory_inference.json','frozen_memory_rescore.json']:
                    memory=read(d/name)
                    assert memory['before_sha256']==memory['after_sha256'] and memory['test_feedback_added']==0
                    counts['unchanged_memory_receipts']+=1
                compat=read(d/'示教兼容核验.json');assert compat['contact_storage_tcp_m']==.043
                assert compat['execution_tcp_m']==.043 and not compat['duplicate_compensation']
            counts['claims_complete']+=1
    for p,digest in source_hashes.items():assert sha(p)==digest,p
    if final:assert counts['claims_complete']==360
    assert set(bank['source_ids']).issubset({p.stem for p in (RUN/'collection/claims').glob('*.json')})
    for file in (RUN/'bank_tcp43/demonstrations').glob('*.json'):
        demo=read(file);trial=Path(demo['trial'])
        assert str(trial).startswith(str(RUN/'collection/trials')+'/')
        assert demo['result_sha256']==sha(trial/'result.json') and demo['evaluation_sha256']==sha(trial/'independent_evaluation.json')
    unit=subprocess.run(['systemctl','--user','show','maniskill-arx-graspcl-compat160-20261120-01.service',
        '-p','ActiveState','-p','MainPID','-p','ExecMainStatus'],capture_output=True,text=True,check=True).stdout
    if final:assert 'ActiveState=inactive' in unit and 'MainPID=0' in unit and 'ExecMainStatus=0' in unit,unit
    return {'passed':True,'final':final,'source':str(RUN),'time':datetime.now().astimezone().isoformat(),
        **dict(counts),'source_and_asset_hashes_checked':len(source_hashes),'registration_sha256':registration_hashes,
        'bank_sha256':frozen['bank_complete_sha256'],'training_test_overlap':0,'test_feedback_updates':0,
        'owned_services_exited_normally':True if final else None,'systemd':unit.strip().splitlines(),
        'new_robot_actions':0,'video_scope':f'All {counts["dual_videos_present"]} available dual-video pairs checked for presence; the 20 selected arm videos are decoded to extract screenshots. Not a full-frame audit of all videos.'}

def make_gallery(a,b,paired):
    import cv2
    from PIL import Image,ImageDraw,ImageFont,ImageOps
    fontfile=SITE.parent/'缓存/对比视频字体/NotoSansCJKsc-Regular.otf'
    fonts={n:ImageFont.truetype(str(fontfile),n) for n in [18,21,26,32]}
    selected=[];layout_use=Counter();model_use=Counter()
    # Deliberately include contrasting outcomes; this is an illustrative sample, not the statistical denominator.
    for key,quota in [('graspcl_only',3),('zerograsp_only',2),('both_success',3),('both_failure',2)]:
        pool=[i for i in paired[key] if i not in selected]
        assert len(pool)>=quota
        for _ in range(quota):
            def rank(i):return (model_use[a[i]['model_id']]*4+layout_use[a[i]['layout']]*2,i)
            id=min(pool,key=rank);pool.remove(id);selected.append(id)
            layout_use[a[id]['layout']]+=1;model_use[a[id]['model_id']]+=1
    assets={};items=[];sheet=Image.new('RGB',(5*320,2*230),'#eff4f7');all_evidence=[]
    def jpg(image,prefix,quality):
        out=io.BytesIO();image.save(out,'JPEG',quality=quality,optimize=True,progressive=True);payload=out.getvalue()
        name=prefix+'-'+hashlib.sha256(payload).hexdigest()[:12]+'.jpg';(DEST/name).write_bytes(payload)
        assets[name]=str((DEST/name).relative_to(SITE/'内容/资源'))
        return {'url':'/media/'+name,'bytes':len(payload)}
    def place(canvas,image,box):
        image=ImageOps.contain(image,box[2:],Image.Resampling.LANCZOS)
        canvas.paste(image,(box[0]+(box[2]-image.width)//2,box[1]+(box[3]-image.height)//2))
    def label(result):
        height=result.get('minimum_hold_lift_m')
        return REASONS.get(result['reason'],result['reason'])+(f' · 最低保持抬升 {height*1000:.1f} mm' if height is not None else '')
    for index,id in enumerate(selected):
        short=id.split('_')[0];r=a[id];left=RUN/ARMS[0]/'trials'/id;right=RUN/ARMS[1]/'trials'/id
        canvas=Image.new('RGB',(1600,1170),'#f3f6fa');draw=ImageDraw.Draw(canvas)
        draw.text((28,15),f'{short}  {LAYOUTS[r["layout"]]}  |  {r["model_id"]}',font=fonts[32],fill='#102a35')
        draw.text((28,65),'初始场景（ZeroGrasp 组当次重置）',font=fonts[26],fill='#234656')
        draw.text((1084,65),'同次腕部 RGB 输入',font=fonts[26],fill='#234656')
        initial=Image.open(left/'initial_overview.png').convert('RGB');wrist=Image.open(left/'camera/color.png').convert('RGB')
        place(canvas,initial,(20,105,1040,580));place(canvas,wrist,(1080,105,500,580))
        evidence=[{'path':str(left/'initial_overview.png'),'sha256':sha(left/'initial_overview.png')},
                  {'path':str(left/'camera/color.png'),'sha256':sha(left/'camera/color.png')}]
        for col,(folder,result,title) in enumerate([(left,a[id],'ZeroGrasp＋cuRobo'),(right,b[id],'ZeroGrasp＋GraspCL＋cuRobo')]):
            x=20+col*800;draw.text((x,701),title,font=fonts[26],fill='#102a35')
            draw.text((x,743),label(result),font=fonts[21],fill='#00685b' if result['success'] else '#a43729')
            path=folder/'execution.mp4';cap=cv2.VideoCapture(str(path));assert cap.isOpened()
            count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
            frame_index=max(0,count-2);cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index);ok,frame=cap.read();cap.release();assert ok,path
            place(canvas,Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),(x,785,780,350))
            evidence.append({'path':str(path),'sha256':sha(path),'frame_index':frame_index,'frame_count':count,'fps':fps,
                'result_sha256':sha(folder/'result.json'),'evaluation_sha256':sha(folder/'independent_evaluation.json')})
        draw.text((22,1140),'下排为各次原始录像的结束前画面；成功与否按整段轨迹核验，不按截图判定。',font=fonts[18],fill='#4b626d')
        image=jpg(canvas,'scene-h452'+short.lower(),83)
        cell=Image.new('RGB',(320,230),'#f3f6fa');d=ImageDraw.Draw(cell)
        place(cell,initial,(0,0,320,180))
        d.text((9,182),f'{short} · {LAYOUTS[r["layout"]]}',font=fonts[18],fill='#102a35')
        d.text((9,205),f'基线：{"成功" if a[id]["success"] else "失败"}  /  GraspCL：{"成功" if b[id]["success"] else "失败"}',font=fonts[18],fill='#102a35')
        sheet.paste(cell,((index%5)*320,(index//5)*230))
        caption=f'{LAYOUTS[r["layout"]]} · {r["model_id"]}；基线：{label(a[id])}；GraspCL：{label(b[id])}'
        items.append({'id':id,'caption':caption,**image,'zerograsp_success':a[id]['success'],'graspcl_success':b[id]['success']})
        all_evidence.append({'id':id,'sources':evidence})
    grid=jpg(sheet,'sheet-h452',77)
    manifest={'columns':5,'rows':2,'cases':items,'assets':assets,
        'sheet':{**grid,'width':1600,'height':460,'alt':'十个45.2厘米实验场景及两条链路的成败缩略拼图'},
        'selection_policy':'3 GraspCL-only, 2 ZeroGrasp-only, 3 both-success, 2 both-failure; deterministic layout/object diversity; illustrative only.',
        'provenance':all_evidence}
    save(SITE/'内容/45厘米高度实验图集.json',manifest)
    save(DEST/'截图来源.json',{'run':str(RUN),'selection_policy':manifest['selection_policy'],'cases':all_evidence})
    return manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');p.add_argument('--gallery',action='store_true');p.add_argument('--audit-completed',action='store_true');args=p.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    a,b=[results(RUN,g) for g in ARMS];paired=pair(a,b)
    high={g:results(OLD,g) for g in ARMS}
    low={g:results(LOW/('zerograsp' if g==ARMS[0] else 'graspcl'),g) for g in ARMS}
    ids={e['id'] for e in read(RUN/ARMS[0]/'registration.json')['schedule']}
    data={'time':datetime.now().astimezone().isoformat(),'final':args.final,'source':str(RUN),
        'selected_lift_m':.452,'registered_test_scenes':160,'registered_test_slots':320,
        'new':{g:metrics(rows) for g,rows in zip(ARMS,[a,b])},'paired':paired,
        'height_only_200':{g:metrics(high[g]) for g in ARMS},
        'history_matched_160':{'height_452_old_bank':{g:metrics({i:r for i,r in high[g].items() if i in ids}) for g in ARMS},
            'height_220_old_bank':{g:metrics({i:r for i,r in low[g].items() if i in ids}) for g in ARMS}},
        'source_collection':metrics(results(RUN,'collection')),'bank':read(RUN/'bank_tcp43/provenance.json')}
    data['per_scene']=[{'id':id,'layout':(a.get(id) or b[id])['layout'],'model_id':(a.get(id) or b[id])['model_id'],
        **{g:({k:r[k] for k in ['success','planning_success','physical_attempt','reason','minimum_hold_lift_m','hold_seconds'] if k in r} if r else None)
           for g,r in zip(ARMS,[a.get(id),b.get(id)])}} for id in sorted(set(a)|set(b))]
    data['historical_same_height_pair']={}
    for group,current in zip(ARMS,[a,b]):
        prior={i:r for i,r in high[group].items() if i in ids};p=pair(prior,current)
        comparison={'paired':p['paired'],'both_success':p['both_success'],'both_failure':p['both_failure'],
            'current_only':p['graspcl_only'],'historical_only':p['zerograsp_only']}
        comparison['regression_reasons']=dict(Counter(current[i]['reason'] for i in comparison['historical_only']))
        comparison['same_planning_outcomes']=sum(prior[i]['planning_success']==current[i]['planning_success'] for i in set(prior)&set(current))
        data['historical_same_height_pair'][group]=comparison
    data['slot_timing']={}
    for group in ARMS:
        durations=[]
        for file in (RUN/group/'claims').glob('*.json'):
            claim=read(file)
            if claim['state']=='complete':durations.append((datetime.fromisoformat(claim['finished_at'])-datetime.fromisoformat(claim['claimed_at'])).total_seconds())
        data['slot_timing'][group]={'samples':len(durations),'mean_seconds':statistics.mean(durations),'median_seconds':statistics.median(durations)}
    if args.final:
        assert set(a)==set(b)==ids and len(ids)==160
        data['audit']=audit();data['runtime']=read(RUN/'运行结束.json')
        save(DEST/'最终核验.json',data['audit'])
    elif args.audit_completed:
        data['partial_audit']=audit(final=False);save(DEST/'已完成槽核验.json',data['partial_audit'])
    save(DEST/('实验汇总.json' if args.final else '处理中汇总.json'),data)
    if args.gallery:make_gallery(a,b,paired)
    print(json.dumps({'final':args.final,'completed':[len(a),len(b)],'successes':[sum(x['success'] for x in a.values()),sum(x['success'] for x in b.values())],
        'paired':paired['paired'],'graspcl_only':len(paired['graspcl_only']),'zerograsp_only':len(paired['zerograsp_only'])},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
