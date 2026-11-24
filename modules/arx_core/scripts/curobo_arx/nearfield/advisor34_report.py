"""Independent final physical, record, event and object-relative step report."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np
from scripts.curobo_arx.data_factory.common import read,write,sha,digest,verify_files
from training.nearfield_arx.projection import Projector
from training.nearfield_arx import native_mask
from training.nearfield_arx.contract import pose_error
from .advisor34_audit import physical,audit_policy,image_binding,audit_partial
from .advisor34_policy import decision
from .advisor34_geometry import observe,PAD
from .advisor31_campaign import exited
from .advisor32_spec import validate_trials
from .capture import audit_episode
from .report import video
from .verify_physical import entry_matches_raw


def report(root):
    root=Path(root).resolve();reg=read(root/'registration.json');validate_trials(reg['seed_trials'])
    if (root/'advisor34_verification.json').exists():raise FileExistsError('Preserve final report')
    idle=exited(root);verify_files({**reg['execution_source_files'],**reg['dependencies']})
    projector=native_mask.projector_for(reg);details=[];steps=[];coverage=Counter();evidence={}
    roots=[root,*map(Path,reg.get('ancestor_campaigns',[]))]
    for trial in reg['seed_trials']:
        found=[p for base in roots for p in base.glob('groups/*/trials/'+trial['trial_id']+'/trial_result.json')]
        partial=[p for base in roots for p in base.glob('groups/*/trials/'+trial['trial_id']+'/ISOLATED_TIMEOUT.json')]
        item={k:trial.get(k) for k in ('trial_id','phase','object_id','condition','vision_arm','comparison_arm','repeat','target_delta_world_m')}
        item.update(task_success=False,record_integrity=None,model_calls=0,controller_calls=0,stale_pairs=0,video=None,
            physical_attempt=False,classification='unexecuted',outcome=None,event_applicability={})
        ep=None
        supplemental=reg.get('supplemental_interruptions',{}).get(trial['trial_id'])
        orphan_episodes=[p for base in roots for p in base.glob('groups/*/trials/'+trial['trial_id']+'/episodes/*') if p.is_dir()]
        if supplemental and found:raise RuntimeError('An interrupted slot was repeated')
        if found:
            if len(found)!=1:raise RuntimeError('Repeated slot result')
            result=read(found[0]);item.update(task_success=result['success'],physical_attempt=bool(result['physical_dispatches']),outcome=result['outcome'],
                classification='success' if result['success'] else 'executed_task_failure' if result['physical_dispatches'] else 'pre_motion_rejection')
            if result['attempts']:ep=Path(result['attempts'][0]['episode'])
            item['source_campaign']=str(next(b for b in roots if found[0].is_relative_to(b)))
        elif partial:
            receipt=read(partial[0]);ep=Path(receipt['episode']);item.update(classification='interrupted_unknown',outcome=receipt['error'],physical_attempt=True)
            item['record_integrity']=audit_partial(partial[0].parent)['pass']
            item['native_terminal_present']=False
        elif supplemental:
            verify_files(supplemental['evidence']);item.update(supplemental['detail'])
            ep=Path(item['source_episode']);evidence.update(supplemental['evidence'])
            if item['classification']!='interrupted_unknown' or item['task_success'] or item['native_terminal_present']:
                raise RuntimeError('Supplemental interruption cannot fabricate native completion')
            coverage.update({k:int(v) for k,v in item['event_applicability'].items()})
        elif orphan_episodes:
            if len(orphan_episodes)!=1:raise RuntimeError('Ambiguous interrupted episode')
            ep=orphan_episodes[0]
            item.update(classification='interrupted_unknown',outcome='missing_native_terminal_receipt',
                physical_attempt=bool(list(ep.glob('*_execution.json')) or list(ep.glob('*_command.json'))),
                native_terminal_present=False,source_episode=str(ep),record_integrity=None,
                record_integrity_scope='Native terminal gap detected; independent interrupted-prefix audit required')
        if ep is None:
            details.append(item);continue
        raw=[json.loads(l) for l in (ep/'samples.jsonl').read_text().splitlines()] if (ep/'samples.jsonl').exists() else []
        if found:
            measured=physical(ep,persist=False);recorded=read(ep/'physical_receipt.json')
            if digest(measured)!=digest(recorded) or measured['pass']!=item['task_success']:raise RuntimeError('Independent physical evaluation differs')
            item['physical']=measured;policy=audit_policy(ep)
            item.update(record_integrity=True,controller_calls=policy['policy_calls'],model_calls=policy['model_calls'],
                capture=audit_episode(ep) if raw else {'applicable':False,'frames':0})
            for r in policy['requests']:
                p=ep/'policy'/(r['request_id']+'_request.json');req=read(p);rep=read(p.with_name(r['request_id']+'_reply.json'))
                image_binding(reg,req,rep);row=read(req['observation'])
                entry_matches_raw(raw[req['raw_index']],row)
                if native_mask.configured(reg):row=native_mask.request_row(req)
                depth=np.load(req['depth'],allow_pickle=False)
                gate,masks,points=observe(projector,req['asset'],row,depth)
                if native_mask.configured(reg):
                    from PIL import Image
                    expected_views=native_mask.model_views(Image.open(req['rgb']),depth,masks,
                        reg['learned_policy']['visual_contract']['depth_scale'])
                    for view,name in zip(expected_views,('rgb','depth')):
                        path=p.with_name(req['request_id']+'_'+name+'.png')
                        if not np.array_equal(np.asarray(view),np.asarray(Image.open(path))):
                            raise RuntimeError('VLM image does not use the independently reconstructed native mask')
                if digest(gate)!=digest(rep['finger_box']):raise RuntimeError('Independent finger-box reconstruction differs')
                try:expected=decision(req,rep['prediction'].strip(),gate,points,row,reg['learned_policy']['visual_contract']['depth_scale'])
                except ValueError as exc:expected={'kind':'reject','reason':'invalid_prediction: '+str(exc),'goal_world':None,'closure_origin':None}
                if digest(expected)!=digest(rep['action']):raise RuntimeError('Independent action interpretation differs')
                item['stale_pairs']+=int(rep['model_inference'] and rep['visual_control']['replaced'])
                actual=np.array(row['snapshot']['link_world']['right_eef_link']);target=np.array(row['snapshot']['objects']['test_object']['world'])
                initial=np.array(read(ep/'capture_snapshot.json')['objects']['test_object']['world'])
                reference=target@np.linalg.inv(initial)@np.array(trial['source_Gstar_world'])
                err=pose_error(actual,reference);delta=actual[:3,:3].T@(reference[:3,3]-actual[:3,3])
                pad=actual[:3,3]+actual[:3,:3]@PAD
                steps.append({'trial_id':trial['trial_id'],'method':trial['method'],'arm':trial['vision_arm'],
                    'decision':req['motions_completed']+1,'mode':req['mode'],'model_inference':rep['model_inference'],
                    'raw_prediction':rep['prediction'],'action':rep['action']['kind'],'closure_origin':rep['action'].get('closure_origin'),
                    'reference_distance_mm':err['translation_m']*1000,'reference_lateral_mm':float(np.linalg.norm(delta[:2])*1000),
                    'reference_axial_mm':float(delta[2]*1000),'reference_rotation_degrees':float(np.rad2deg(err['rotation_rad'])),
                    'pad_to_object_origin_mm':float(np.linalg.norm(pad-target[:3,3])*1000),
                    'target_displacement_since_reset_mm':float(np.linalg.norm(target[:3,3]-initial[:3,3])*1000),
                    'in_hand_points':gate['in_hand_points'],'finger_box_pass':gate['pass'],
                    'surface_z_EE_mm':None if gate['near_surface']['median_z_EE_m'] is None else gate['near_surface']['median_z_EE_m']*1000,
                    'direct_visible_pass':gate['direct_visible_pass'],'closure_gate_origin':gate['gate_origin'],
                    'gravity_column_source_points':gate['occlusion_envelope']['source_points'],
                    'controller_axial_only':bool(rep['action'].get('preclosure_axial_correction')),
                    'axial_compensation_applied':bool(rep['action'].get('axial_alignment',rep['action'].get('interpretation') or {}).get('applied'))})
                evidence[str(p)]=sha(p)
            occurred=set(measured['executed_stages'])
            flags={'H_reached':'state_generation' in occurred,'first_D_motion':'d_refine_001' in occurred,
                'second_D_motion':'d_refine_002' in occurred,'close':'close' in occurred,'lift':'lift' in occurred,'hold':'hold' in occurred,
                'neural_inference':bool(item['model_calls']),'full_stale_two_pairs':item['stale_pairs']==2,
                'position_window':'settled_by_position_window' in measured['stop_origins'],
                'preclosure_axial_motion':'axial_align_001' in occurred,
                'occluded_column_gate':any(s['trial_id']==trial['trial_id'] and s['closure_gate_origin']=='gravity_column' for s in steps)}
            item['strict_original_success']=measured['strict_original_success']
            item['event_applicability']=flags;coverage.update({k:int(v) for k,v in flags.items()})
            for p in [*ep.parent.parent.glob('blocks/*/frozen_*.json'),ep.parent.parent/'frozen_after.json',ep.parent.parent/'reset_verification.json']:
                proof=read(p)
                if not proof['pass']:raise RuntimeError('Frozen configuration/reset failed')
                evidence[str(p)]=sha(p)
        if raw and item['video'] is None:
            old_video=Path(item.get('source_campaign',str(root)))/'videos'/(trial['trial_id']+'.json')
            if item.get('source_campaign')!=str(root) and old_video.is_file():
                receipt=read(old_video);verify_files({receipt['path']:receipt['sha256']})
                dest=root/'videos'/Path(receipt['path']).name;dest.parent.mkdir(exist_ok=True)
                dest.symlink_to(Path(receipt['path']));item['video']={**receipt,'path':str(dest),'retained_from':receipt['path']}
                write(dest.with_suffix('.json'),item['video'])
            else:item['video']=video(ep,root/'videos'/(trial['trial_id']+'.mp4'))
        details.append(item)
    complete=all(d['classification']!='unexecuted' for d in details)
    result={'complete':complete,'scheduled':30,'completed':sum(d['classification'] not in ('unexecuted','interrupted_unknown') for d in details),
        'successes':sum(d['task_success'] for d in details),'unknown':sum(d['classification']=='interrupted_unknown' for d in details),
        'unexecuted':sum(d['classification']=='unexecuted' for d in details),'physical_attempts':sum(d['physical_attempt'] for d in details),
        'record_integrity_pass':all(d['record_integrity'] is True for d in details if d['classification'] not in ('unexecuted','pre_motion_rejection')),
        'native_terminal_gaps':sum(d['classification']=='interrupted_unknown' for d in details),
        'event_coverage':dict(coverage),'trials':details,'steps':steps,'evidence':evidence,'idle':idle,
        'object_relative_reference':'Offline translated reference, advected by measured object pose; never supplied to D',
        'attribution_limits':'Fresh geometric first-plane depth and safety gates remain in full-stale arm; historical second-plane pose can differ from current pose.'}
    write(root/'advisor34_verification.json',result)
    write(root/'protocol_comparison.json',{'registered_dual_stop_successes':result['successes'],
        'strict_original_successes':sum(bool(d.get('strict_original_success')) for d in details),
        'unknown':result['unknown'],'unexecuted':result['unexecuted'],
        'original_contact_height_hold_planning_checks_unchanged':True,
        'trials':[{'trial_id':d['trial_id'],'dual_stop':d['task_success'],
                   'strict_original':d.get('strict_original_success'),'classification':d['classification']} for d in details]})
    if steps:
        with (root/'object_relative_steps.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(steps[0]));writer.writeheader();writer.writerows(steps)
    lines=['# 第34轮30槽物理结果','',f'终结 {30-result["unexecuted"]}/30；成功 {result["successes"]}；中断未知 {result["unknown"]}；未执行 {result["unexecuted"]}。','',
        '| 槽位 | 物体 | 对照 | 结果 | 原因 | 腕部录像 |','|---|---|---|---|---|---|']
    for d in details:
        link=f'[视频]({d["video"]["path"]})' if d['video'] else '无已记录图像'
        lines.append(f'| {d["trial_id"]} | {d["object_id"]} | {d["comparison_arm"] or d["vision_arm"]} | {d["classification"]} | {d["outcome"] or "—"} | {link} |')
    lines+=['','逐步距离见 object_relative_steps.csv。距离按实际移位及后续物体运动计算，仅用于离线分析。',
        '全程旧图替换两次神经推理的RGB-D；第一平面的几何轴向推进与执行安全检查仍使用新鲜观测，因此它衡量的是神经图像贡献，不能单独隔离整个视觉链路的贡献。',
        '历史旧图第二平面的相机姿态可能与新执行不同，这一干预同时含旧物体位置与旧观察视角影响。','']
    (root/'RESULTS.md').write_text('\n'.join(lines));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('campaign');print(report(p.parse_args().campaign)['successes'])
