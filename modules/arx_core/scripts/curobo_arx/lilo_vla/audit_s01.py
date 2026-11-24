"""CPU-only S01 audit. Inputs are immutable; no simulation, HTTP or GPU calls."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import scipy
from PIL import Image
from scipy.spatial.transform import Rotation
from observations import policy_observation
from summarize_p2 import policy_trial
from egad_evaluate import evaluate
from s01_checks import state_check, action_check, fault_tests, model_check

ROOT=Path(__file__).resolve().parents[3]
EXPECTED={
'action_head.pt':'00612c08708871d9035debb7d99fd122a855ed34c91f6a7da4f8af5a7634591d',
'dataset_statistics.json':'b16a71f31c75e471833885eccd8ddf8408692275d6208cc0f34eb12b7b180bf6',
'adapter_manifest.json':'fe156b712e87dc540b092b3edbe70713d12b40640c0bafe9f09d07a1852cf51c',
'../goal_dataset/dataset.json':'67446a8ed8e3ccc35b644e869c8f6a13baa8c8643142e8df38b3f54f96abe3af'}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--legacy-regression',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if shutil.disk_usage(out.parent).free<1073741824:raise RuntimeError('1 GiB reserve')
    out.mkdir(exist_ok=False,parents=True)
    evidence={};gaps=[];coverage=[]
    def sha(p):
        p=Path(p).resolve();h=digest(p)
        if str(p) in evidence and evidence[str(p)]!=h:raise RuntimeError('Input changed: '+str(p))
        evidence[str(p)]=h;return h
    def read(p,optional=False):
        p=Path(p)
        if optional and not p.exists():return None
        sha(p);return json.loads(p.read_text())
    def write(n,d):
        (out/n).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    def gap(scope,reason,blocking=False):gaps.append(dict(scope=scope,reason=reason,blocks_s02_comparison=blocking))
    spec=read(args.sources);adapter=ROOT/spec['baseline'];baseline=read(adapter/'adapter_manifest.json')
    policy=ROOT/'runs/research_contract_20261019/storage_policy.json'
    if policy.exists():sha(policy);shutil.copyfile(policy,out/'storage_policy.json')
    write('workspace_snapshot.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'status':subprocess.check_output(['git','status','--short','--branch'],cwd=ROOT,text=True),
        'python':sys.executable,'numpy':np.__version__,'scipy':scipy.__version__,'resources':{'gpu':0,'requests':0,'trajectories':0}})
    for p in [*Path(__file__).parent.glob('*s01*.py'),Path(__file__).with_name('observations.py'),Path(__file__).with_name('egad_evaluate.py'),*list((ROOT/'research_plan/docs').glob('*.md'))]:
        sha(p)
        snapshot=out/'source_snapshots'/p.resolve().relative_to(ROOT);snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,snapshot)
    identity={n:{'actual':sha(adapter/n),'expected':h} for n,h in EXPECTED.items()}
    for v in identity.values():v['pass']=v['actual']==v['expected']
    regression=read(args.legacy_regression/'validation.json')
    write('identity_checks.json',identity)
    write('baseline_manifest.json',{'adapter':str(adapter),'manifest':baseline,'identity':identity,'legacy_regression_source':str(args.legacy_regression),'legacy_regression':regression,
        'note':'Historical validator future_dependency text is stale; D1 scheduling is already implemented. Saved health is not a live service receipt.'})
    dataset=read(adapter.parent/'goal_dataset/dataset.json')
    exposure=[{k:e.get(k) for k in ['source','object_name','split','selected_grasp_world']} for e in dataset['episodes']]
    roots=[ROOT/p for p in spec['campaigns']]
    cursor=ROOT/'runs/lilo_P2_collection_20261028_05'
    while cursor not in roots:
        roots.append(cursor);r=read(cursor/'registration.json');prev=r.get('previous_collection')
        if not prev:break
        cursor=Path(prev)
    roots.append(ROOT/'runs/lilo_P2_teacher_probe_20261028_01')
    rows=[];links=[];reconciliations=[];configs=[];seen={}
    for root in roots:
        reg=read(root/'registration.json');teacher='collection_' in root.name or 'teacher_' in root.name
        results=read(root/('all_attempts.json' if (root/'all_attempts.json').exists() else 'results.json'))
        expected=reg.get('cases',reg.get('episodes'))
        key='case_id' if 'cases' in reg else 'label'
        if expected is None:
            if 'teacher_qualified_goal_sources' in reg:expected=[{'scene':s} for s in reg['teacher_qualified_goal_sources']];key='scene'
            elif 'screening_order' in reg:expected=[{'label':f'{s}_{v}_r1'} for s,v in reg['screening_order']]
            else:expected=None
        owned_results=[r for r in results if Path(r.get('source',str(root/r.get('label','unknown')))).parent.resolve()==root.resolve()] if 'episodes' in reg else results
        missing=[]
        if expected is not None:
            for e in expected:
                if not any(x.get(key)==e[key] for x in owned_results):missing.append({**e,'status':'not_executed','pass':None})
        else:
            # P3 adaptive registration fixes a cap and selection/repeat rule, not eight unconditional slots.
            if len(results)!=reg['max_cases']:gap(str(root),'Adaptive registration/result count requires review',True)
            selection=read(root/'point_selection.json',True)
            reconciliations.append({'campaign':str(root),'conditional_selection':selection,'max_cases':reg['max_cases'],'rule':reg['repeat_rule']})
        reconciliations.append({'campaign':str(root),'registered':len(expected) if expected is not None else reg['max_cases'],
            'results':len(owned_results),'results_including_inherited_references':len(results),'missing_or_not_executed':len(missing),'registration_sha256':sha(root/'registration.json')})
        for r in results+missing:
            t=Path(r.get('trial') or r.get('source') or root/r.get('label',r.get('case_id',r.get('scene','unknown')))).resolve()
            identity_key=(str(t),r.get('attempt_id'))
            if identity_key in seen:links.append({'reference_campaign':str(root),'canonical':seen[identity_key]});continue
            row={'id':len(rows),'campaign':str(root),'trial':str(t),'role':'teacher' if teacher else 'policy',
                'group':'core_goal' if ('goal_physical'==root.name or 'P3_' in root.name) else 'teacher' if teacher else root.name,
                'result':r,'status':r.get('status','recorded'),'registration':str((t.parent if (t.parent/'registration.json').exists() else root)/'registration.json')}
            seen[identity_key]=row['id'];rows.append(row)
        for p,h in reg.get('physical_source_sha256',{}).items():
            actual=sha(p) if Path(p).exists() else None
            configs.append({'campaign':str(root),'path':p,'registered_sha256':h,'current_sha256':actual,'match':actual==h,
                'qualification':'File identity only; no assertion about unrecorded runtime properties'})
    write('provenance_links.json',{'deduplicated_references':links,'registration_reconciliation':reconciliations})
    write('configuration_evidence.json',configs)
    write('split_and_exposure_registry.json',{'baseline_episodes':exposure,'train_episodes':sum(e['split']=='train' for e in dataset['episodes']),
        'validation_episodes':sum(e['split']=='validation' for e in dataset['episodes']),
        'chunks':{s:sum(len(e['chunk_starts']) for e in dataset['episodes'] if e['split']==s) for s in ['train','validation']},
        'development_objects':['A1','bbq_sauce_bottle','rubiks_cube','lemon_01','mustard_bottle','apple_01','005_tomato_soup_can'],
        'restriction':'D2 is later teacher data, not baseline training; all listed viewed objects are excluded from unseen confirmatory claims.'})
    handoffs=[];actions=[];conflicts=[];failures=[];tests={};numeric=[]
    fields=['setup.json','scenario.json','selected_grasp.json','handoff_target.json','handoff_actual_entry.json','handoff_actual_observation/observation.json',
        'lilo_observations/0000/observation.json','policy_0000_prediction.json','local_0000_prediction.json','local_0000_state.json',
        'local_0000_command.json','local_0000_validation.json','local_0000_execution.json','local_0000_settling.json','grasp_report.json',
        'interaction_first_observation.json','handoff_policy_admission.json']
    for row in rows:
        t=Path(row['trial']);r=row['result'];teacher=row['role']=='teacher';record={'id':row['id'],'trial':str(t),'fields':{}}
        if row['status']=='not_executed':failures.append({'id':row['id'],'stage':'not_executed'});continue
        for name in fields:
            p=t/name;v=read(p,True)
            status='recorded' if v is not None else 'not_applicable' if teacher and ('policy' in name or 'local_' in name or 'lilo_observations' in name) else 'missing'
            record['fields'][name]={'status':status,'source_path':str(p),'sha256':evidence.get(str(p.resolve())),'json_pointer':'/','value':v}
            coverage.append([row['id'],name,status])
        values={k:v['value'] for k,v in record['fields'].items()};obs=values['lilo_observations/0000/observation.json'];pred=values['policy_0000_prediction.json'];grasp=values['selected_grasp.json'];report=values['grasp_report.json'] or {}
        record['prediction_recorded']=pred is not None
        record['instruction']={'status':'recorded' if report.get('instruction') else 'missing','value':report.get('instruction'),'source':str(t/'grasp_report.json')}
        entry=values['handoff_actual_observation/observation.json']
        record['velocity_status']='recorded' if (entry or obs or {}).get('snapshot',{}).get('joint_velocities') is not None else 'missing'
        if entry and obs:
            a=np.asarray(entry['snapshot']['link_world']['right_eef_link']);b=np.asarray(obs['snapshot']['link_world']['right_eef_link'])
            dp=float(np.linalg.norm(a[:3,3]-b[:3,3]));dr=float(Rotation.from_matrix(b[:3,:3]@a[:3,:3].T).magnitude())
            record['handoff_to_first_input']={'status':'derived','position_m':dp,'rotation_rad':dr,'within_current_contract':dp<=.002 and dr<=.03,
                'historical_check_recorded':values['interaction_first_observation.json'] is not None}
        planned=values['handoff_target.json']
        if planned and grasp:
            record['planned_goal_matches_selection']=bool(np.array_equal(planned.get('grasp_world'),grasp['grasp_world']))
        if pred and obs and grasp:
            convention=pred.get('state_convention',report.get('state_convention'))
            modeldir=pred.get('adapter');statsfile=Path(modeldir)/'dataset_statistics.json' if modeldir else None
            if convention and statsfile and statsfile.exists():
                stats=read(statsfile)[pred['dataset_key']]['proprio']
                record['input_check']=state_check(obs,pred,grasp,convention,stats)
                image,boxes,_=policy_observation(obs,convention,grasp['grasp_world']);sha(obs['rgb_path'])
                buffer=io.BytesIO();Image.fromarray(image).save(buffer,format='PNG')
                record['image_chain']={'status':'derived','raw_sha256':sha(obs['rgb_path']),'reconstructed_request_png_sha256':hashlib.sha256(buffer.getvalue()).hexdigest(),
                    'mask_boxes_match':boxes==pred.get('distractor_boxes'),'preprocessing':pred.get('preprocessing'),'final_model_pixels':'missing',
                    'limitation':'Request/model image bytes were not saved historically; reconstructed PNG is not proof of exact consumed pixels.'}
                if row['group']=='core_goal':
                    artifact=pred.get('policy_artifact_sha256');record['historical_model_identity']={'status':'recorded' if artifact else 'derived','artifact_hashes':artifact,
                        'dataset_key_match':pred['dataset_key']=='arx_shared_goal_v1','selected_step_match':pred.get('adaptation_selected_step')==225,
                        'revision_match':pred.get('checkpoint_revision')=='f54c56709004f6680d2ce8bc9e06d8cc96567ad8','metadata_pass':model_check(pred,EXPECTED)}
            else:gap(str(t),'Historical convention or normalization source unavailable')
        if not teacher:
            # Preserve failures from the preliminary environment, recompute in the runtime environment.
            check=policy_trial(t,r);numeric.append({'id':row['id'],'trial':str(t),'pass':check['action_provenance_pass'],'errors':check['provenance_errors'],
                'actual_policy_call_count':len(list(t.glob('policy_*_prediction.json')))})
            # Hash all JSON sources read indirectly by the existing provenance checker/evaluator.
            for p in t.glob('*.json'):sha(p)
            names=['local_0000_prediction.json','local_0000_state.json','local_0000_command.json','local_0000_validation.json','local_0000_execution.json','local_0000_settling.json']
            if pred and all(values[n] is not None for n in names):
                check=action_check(pred,*[values[n] for n in names]);actions.append({'id':row['id'],'trial':str(t),**check})
                if not tests and row['group']=='core_goal' and obs and grasp and convention and statsfile:
                    tests=fault_tests(obs,pred,grasp,convention,stats,*[values[n] for n in names])
            else:actions.append({'id':row['id'],'trial':str(t),'pass':None,'status':'not_applicable' if not pred else 'missing','reason':'No saved prediction' if not pred else 'Incomplete first command chain'})
        try:
            ev=evaluate(t)
            if ev!=r.get('evaluation'):conflicts.append({'id':row['id'],'original':r.get('evaluation'),'recomputed':ev})
        except Exception as exc:gap(str(t),'Evaluator: '+str(exc))
        error=report.get('error') or r.get('evaluation',{}).get('reported_error') or r.get('error') or ''
        lower=str(error).lower();stage='unknown'
        if r.get('pass'):stage='success'
        elif 'no module' in lower or 'cuda' in lower or 'out of memory' in lower or 'file exists' in lower or 'directory reuse' in lower:stage='infrastructure'
        elif 'handoff' in lower or 'entry' in lower:stage='handoff'
        elif 'bilateral' in lower or 'closure' in lower:stage='close'
        elif 'hold' in lower:
            ev=r.get('evaluation',{});bad=[x for x in ev.get('executed_stages',[]) if x not in ev.get('converged_stages',[])]
            first=bad[0] if bad else 'hold';stage='reach' if first.startswith('pregrasp') else first if first in ['close','lift','hold'] else 'unknown'
        elif 'lift' in lower:stage='lift'
        elif 'pregrasp' in lower or 'reaching' in lower:stage='reach'
        elif 'approach' in lower:stage='local approach'
        elif 'perception' in lower or 'zerograsp' in lower:stage='setup/perception'
        elif 'candidate' in lower or 'grasp selection' in lower:stage='grasp selection'
        elif pred:stage='local approach' if not values['local_0000_command.json'] else 'unknown'
        failures.append({'id':row['id'],'stage':stage,'evidence':error,'classification':'derived from saved error; unknown retained when stage is ambiguous','prediction_recorded':bool(pred)})
        handoffs.append(record)
        print('S01 checked',row['id'],t.name,flush=True)
    test_pred=json.loads((ROOT/'runs/lilo_P3_guarded_20261028_02/ten_mm_r1/A1/policy_0000_prediction.json').read_text())
    tests['real_model_identity']=model_check(test_pred,EXPECTED)
    test_pred['policy_artifact_sha256']['action_head.pt']='wrong'
    tests['wrong_model_identity_detected']=not model_check(test_pred,EXPECTED)
    tests['prepolicy_failures_not_counted_as_predictions']=all(not h['prediction_recorded'] for h in handoffs if 'P3_guarded_20261028_01/' in h['trial'])
    for scope,reason in [('all historical runs','Complete observation→request→execution timestamps not recorded; end-to-end observation age remains unknown.'),
        ('image provenance','Historical post-preprocessing model pixels/request payload hashes absent. Add capture/request IDs and hashes in S02; preserve preprocessing.'),
        ('configuration','Runtime physical attributes beyond saved files were not comprehensively captured. Keep initial-state/config identities fixed and record runtime fingerprint in S02.'),
        ('instruction','Instruction is recorded in rollout report but historical complete request payload is absent; link request ID and instruction hash in S02.'),
        ('action target','Historical world target is reconstructed and corroborated by settling/candidate certificates; record intended target directly in S02.')]:gap(scope,reason)
    changed=[p for p,h in evidence.items() if digest(p)!=h]
    if changed:gap('input protection',str(changed),True)
    for name,data in [('attempt_ledger.jsonl',rows),('handoff_records.jsonl',handoffs)]:
        (out/name).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in data))
    with (out/'field_coverage.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['attempt_id','field','status']);w.writerows(coverage)
    write('source_manifest.json',{'declared_sources':spec,'expanded_campaigns':[str(p) for p in roots],'files':evidence})
    write('action_provenance_checks.json',{'first_action':actions,'all_saved_actions_runtime_environment':numeric})
    write('input_consistency.json',{'changed_sources':changed,'inputs':[{'id':h['id'],'check':h.get('input_check'),'handoff':h.get('handoff_to_first_input')} for h in handoffs]})
    write('failure_stage_summary.json',{'counts':dict(Counter(r['stage'] for r in failures)),'rows':failures})
    write('outcome_conflicts.json',conflicts);write('gap_register.json',gaps);write('fault_injection_tests.json',tests)
    (out/'logging_change_design.md').write_text('''# S02 补录设计（尚未实施）
复用现有观测和动作调用，不增加仿真步、不改变 K=1 或控制时序。
在 snapshot 中保留实测 joint_velocities 及关节名；禁止以命令速度或差分冒充实测。
为观测、请求、预测、动作建立贯通 ID，保存原图/遮罩后请求图/模型处理图哈希及预处理代码身份。
记录 selected G、计划 E/q、实测入口、完整模型哈希、指令、状态和归一化。
在现有 send 调用保存由动作重建的 world_goal、对应 IK 候选、命令与执行回执 ID。
分别记录 camera frame/capture 的仿真时间，客户端请求/命令发送的 monotonic 时间，服务端内部耗时；不同进程时钟不能直接相减。使用共享观测 ID 与同一客户端时钟建立端到端时间。
S02 前以原成功/失败 ARX case 验证关联完整性及采集不改变动作与物理参数，未经运行验证保持待验证。
''')
    identities=all(v['pass'] for v in identity.values()) and regression.get('pass_check') is True
    state_ok=all(h.get('input_check',{}).get('state_pass',True) and h.get('input_check',{}).get('normalization_pass',True) for h in handoffs)
    action_ok=all(a['pass'] is not False for a in actions) and all(n['pass'] for n in numeric)
    metadata_ok=all(h.get('historical_model_identity',{}).get('metadata_pass',True) for h in handoffs)
    checks_ok=all(tests.values()) and identities and state_ok and action_ok and metadata_ok and not changed
    receipt={'offline_audit_complete':True,'ready_for_s02_design':checks_ok,'ready_for_s02_execution':False,
        'identity_and_legacy_regression_pass':identities,'historical_model_metadata_pass':metadata_ok,'numerical_input_pass':state_ok,'action_checks_pass':action_ok,
        'fault_tests_pass':all(tests.values()),'registered_total':len(rows),'results':len(handoffs),'not_executed':sum(r['status']=='not_executed' for r in rows),
        'policy_predictions':sum(h['prediction_recorded'] for h in handoffs),'first_action_checked':sum(a['pass'] is not None for a in actions),
        'first_action_passed':sum(a['pass'] is True for a in actions),'outcome_conflicts':len(conflicts),'input_changes':changed,
        'qualification':'Offline review finished including explicitly missing historical evidence; no physical or policy capability qualification.'}
    write('acceptance_report.json',receipt)
    text=f'''# S01 验收

总判断：{'可准备 S02 方案；执行前需补录验证' if checks_ok else '需先处理下列核验问题'}。

| 项目 | 结果 |
|---|---|
| 模型和旧数据 | {'通过' if identities else '失败'}：P2 goal step 225；原数据精确回归 |
| 接管与首条动作 | {receipt['first_action_passed']}/{receipt['first_action_checked']} 条有完整首条动作记录通过；{receipt['policy_predictions']} 次有预测；历史模型最终图像及完整时延仍待补证据 |
| 登记和失败覆盖 | 共 {len(rows)} 项，{len(handoffs)} 项有结果，{receipt['not_executed']} 项未执行；控制器分组保留 |

核验完成不等于历史证据全部齐全。S02 实际运行未放行。

主要待补项：模型实际输入图像关联、观测到执行的完整时间链、运行时物理属性记录。详见 [缺口](gap_register.json)。

[机器验收](acceptance_report.json) · [动作核验](action_provenance_checks.json) · [错误注入](fault_injection_tests.json) · [总账](attempt_ledger.jsonl)
'''
    (out/'S01_ACCEPTANCE.md').write_text(text)
    (out/'S01_report.md').write_text(text+'\n## 数值复查\n\n原数据构建环境的 7 条状态差异应与本次原执行环境结果并列查看，未放宽 1e-10 容差。没有预测的记录不因旧函数空循环返回 true 而获得策略资格。\n\n## 比较边界\n\nP2 goal/P3 为核心策略证据；object/old 单列，D2 与 P2 来源示范是 teacher。原数据集 5 train/3 validation，210/126 chunk；D2 不在该旧模型训练集内。已查看对象仅能用于开发回归。\n\n完整字段来源和缺失见 handoff_records.jsonl、field_coverage.csv；后续补录接口见 logging_change_design.md。\n')
    print(json.dumps(receipt,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
