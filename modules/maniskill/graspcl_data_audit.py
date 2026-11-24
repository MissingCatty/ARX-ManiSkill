"""Read-only provenance and candidate-origin audit; no model or simulation runs."""
from collections import Counter
from datetime import datetime
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

CODE = Path(__file__).resolve().parent
INPUTS = CODE / 'benchmark_inputs/arx_native_post_20261117_01'
ORIGIN = CODE / 'outputs/arx_zerograsp_graspcl_demo_20261117_02'
PAIR = CODE / 'outputs/arx_native_post_pair_20261117_02'
TRUNC = CODE / 'outputs/arx_table_tcp43_20261117_01'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def audit(out):
    out.mkdir(parents=True,exist_ok=False)
    bank = INPUTS / 'frozen_bank'
    complete = read(bank / 'bank_complete.json')
    oldreg = read(ORIGIN / 'registration.json')
    schedule = read(INPUTS / 'manifest.json')['schedule']
    ids = set(complete['source_ids']);test_ids = {e['id'] for e in schedule}
    labelled = {e['id']:e for e in oldreg['training_trials'] if e['label'] is not None}
    assert ids==set(labelled)
    models = {e['model_id'] for e in labelled.values()}
    with np.load(bank / 'model/score_cl/online_data.npz',allow_pickle=False) as f:
        features = f['X'].copy();labels = f['y'].copy();weights = f['w'].copy()
    assert len(features)==len(complete['source_ids'])==25
    assert list(labels.astype(int))==[labelled[i]['label'] for i in complete['source_ids']]
    recall = []
    for path in sorted((bank / 'model').glob('recall*/**/grasps.csv')):
        for line in path.read_text().splitlines():
            if line.strip():
                grasp = json.loads(line)
                recall.append({'path':str(path),'source_id':grasp['metadata']['source_trial_id'],
                    'label':grasp['success']})
    assert len(recall)==24 and all(r['label']==1 and labelled[r['source_id']]['label']==1 for r in recall)
    feedback = [{k:labelled[i].get(k) for k in ['id','model_id','layout','label','use','original_reason','trial']}
                for i in complete['source_ids']]
    with (out / '反馈来源.csv').open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(feedback[0]));writer.writeheader();writer.writerows(feedback)
    results = {};per_case = [];exclusion_errors = [];fingerprint_errors = [];self_recall = []
    for root,arm in [(PAIR,'graspcl_200'),(TRUNC,'fixed_graspcl'),(TRUNC,'free_graspcl')]:
        armreg = read(root / arm / 'registration.json')
        counts = Counter();origins = Counter();selected_sources = Counter();retained_negative_queries = 0
        candidate_totals = Counter();overlap_ids = []
        for entry in armreg['schedule']:
            slot = entry['id'];trial = root / arm / 'trials' / slot
            if slot in ids:overlap_ids.append(slot)
            log = trial / 'self_case_exclusion.json'
            if not log.exists():counts['no_inference']+=1;continue
            receipt = read(log);counts['inference_receipts']+=1
            expected = ids-{slot}
            if set(receipt['retained_feedback_ids'])!=expected or slot in receipt['retained_feedback_ids']:
                exclusion_errors.append(str(log))
            mask = np.array([i!=slot for i in complete['source_ids']],dtype=bool)
            digest = hashlib.sha256(features[mask].tobytes()).hexdigest()
            if receipt['retained_feature_sha256']!=digest:fingerprint_errors.append(str(log))
            counts['self_excluded']+=int(bool(receipt['excluded_feedback_ids']))
            if any(labelled[i]['label']==0 for i in expected):retained_negative_queries+=1
            pool = read(trial / 'graspcl/candidate_pool.json')
            for grasp in pool['recalled_before_filter']:
                source = grasp['metadata']['source_trial_id']
                if source==slot or source not in ids or labelled[source]['label']!=1:self_recall.append(str(trial))
            candidate_totals['candidates_after_last_transform']+=len(pool['candidates'])
            candidate_totals['native_postprocessed_input']+=pool['counts']['native_postprocessed_zero']
            candidate_totals['recalled_before_filter']+=len(pool['recalled_before_filter'])
            rescore = trial / 'refined_rescore.json'
            if rescore.exists():
                counts['rescore_receipts']+=1;rr=read(rescore)['self_case_exclusion']
                if set(rr['retained_feedback_ids'])!=expected:exclusion_errors.append(str(rescore))
                if rr['retained_feature_sha256']!=digest:fingerprint_errors.append(str(rescore))
            selected = None
            if (trial / 'plan_result.json').exists():selected=read(trial / 'plan_result.json').get('selected')
            if selected:
                counts['planning_passes']+=1;m=selected['candidate_metadata'];origins[m['origin']]+=1
                if m.get('source_trial_id'):selected_sources[m['source_trial_id']]+=1
                if m.get('source_trial_id')==slot:self_recall.append(str(trial)+':selected')
            per_case.append({'batch':root.name,'arm':arm,'id':slot,'source_feedback_query':slot in ids,
                'expected_memory_points':len(expected),'negative_memory_points':sum(labelled[i]['label']==0 for i in expected),
                'self_excluded':receipt['excluded_feedback_ids'],'feature_hash_matches_source_mask':receipt['retained_feature_sha256']==digest,
                'selected_origin':selected['candidate_metadata']['origin'] if selected else None,
                'selected_source_id':selected['candidate_metadata'].get('source_trial_id') if selected else None})
        runtime = root / 'resident/graspcl_model/model/score_cl/online_data.npz'
        assert sha(runtime)==sha(bank / 'model/score_cl/online_data.npz')
        health = read(root / arm / 'resident_health_after_round.json')['graspcl']
        assert health['bank_updates']==0 and health['encoder_load_count']==health['bank_load_count']==1
        results[root.name+'/'+arm]={'registered':len(armreg['schedule']),**dict(counts),
            'selected_origins':dict(origins),'selected_demo_sources':dict(selected_sources),
            'queries_retaining_other_negative_feedback':retained_negative_queries,
            'source_ids_in_arm':overlap_ids,'candidate_totals':dict(candidate_totals),
            'runtime_bank_file_unchanged':True,'reported_online_bank_updates':health['bank_updates']}
    assert not exclusion_errors and not fingerprint_errors and not self_recall
    pair_counts = Counter();gains=[];losses=[];native_same=0;native_changed=0
    for entry in schedule:
        slot=entry['id'];z=PAIR/'zerograsp_200/trials'/slot;c=PAIR/'graspcl_200/trials'/slot
        zr=read(z/'result.json');cr=read(c/'result.json')
        a=bool(zr['planning_success']);b=bool(cr['planning_success'])
        pair_counts[f'zero_{a}_graspcl_{b}']+=1
        if not a and b:
            m=read(c/'plan_result.json')['selected']['candidate_metadata']
            gains.append({'id':slot,'origin':m['origin'],'source_id':m.get('source_trial_id'),
                'source_label':labelled[m['source_trial_id']]['label'] if m.get('source_trial_id') else None,
                'novel_against_all_raw_zero':m.get('novelty_against_all_zero',{}).get('novel')})
        if a and not b:losses.append(slot)
        if (z/'postprocessed_grasps.npy').exists() and (c/'postprocessed_grasps.npy').exists():
            same=np.array_equal(np.load(z/'postprocessed_grasps.npy'),np.load(c/'postprocessed_grasps.npy'))
            native_same+=int(same);native_changed+=int(not same)
    subsets={}
    for name,entries in [('all_200',schedule),('exclude_25_memory_sources',[e for e in schedule if e['id'] not in ids]),
                         ('exclude_25_memory_object_models',[e for e in schedule if e['model_id'] not in models])]:
        stats={}
        for arm in ['zerograsp_200','graspcl_200']:
            rows=[read(PAIR/arm/'trials'/e['id']/'result.json') for e in entries]
            stats[arm]={'registered':len(rows),'planning_passes':sum(r['planning_success'] for r in rows),
                'successes':sum(r['success'] for r in rows)}
        subsets[name]=stats
    latest={}
    for mode in ['fixed','free']:
        z={p.name:read(p/'result.json') for p in (TRUNC/(mode+'_zerograsp')/'trials').iterdir()}
        c={p.name:read(p/'result.json') for p in (TRUNC/(mode+'_graspcl')/'trials').iterdir()}
        common=set(z)&set(c)
        latest[mode]={'common_cases':len(common),'zerograsp_planning_passes':sum(z[i]['planning_success'] for i in common),
            'graspcl_planning_passes':sum(c[i]['planning_success'] for i in common)}
    summary={'created_at':datetime.now().astimezone().isoformat(),'analysis_only':True,'new_physical_attempts':0,
        'bank':{'source':str(bank),'original_source':str(ORIGIN),'positives':int(sum(labels==1)),
            'negatives':int(sum(labels==0)),'weights':sorted(set(weights.tolist())),
            'failed_feedback_reasons':dict(Counter(e['original_reason'] for e in labelled.values() if e['label']==0)),
            'recall_templates':len(recall),'recall_unique_source_ids':len({r['source_id'] for r in recall}),
            'negative_recall_templates':sum(r['label']==0 for r in recall),
            'feedback_id_overlap_with_200':len(ids&test_ids),'feedback_object_models':len(models),
            'evaluation_cases_with_memory_object_model':sum(e['model_id'] in models for e in schedule),
            'online_file_sha256':sha(bank/'model/score_cl/online_data.npz')},
        'query_audit':results,'invalid_exclusions':exclusion_errors,'invalid_feature_masks':fingerprint_errors,
        'self_or_negative_recall_violations':self_recall,'planning_pair_counts':dict(pair_counts),
        'zero_failed_graspcl_passed':gains,'zero_passed_graspcl_failed':losses,
        'native_arrays_bitwise_same':native_same,'native_arrays_bitwise_different':native_changed,
        'retrospective_subsets':subsets,'latest_common_subset':latest,
        'evaluation_scope':'Same-benchmark historical adaptation with per-query leave-one-case-out exclusion; not a wholly disjoint fixed train/test split or unseen-object baseline. Scene/object-excluded summaries are post-hoc diagnostics, not a newly preregistered independent benchmark.',
        'causal_scope':'Candidate recall and scoring both differ. Higher observed planning availability cannot be attributed solely to reranking or negative feedback without ablation.',
        'code_evidence':{str(CODE/n):sha(CODE/n) for n in ['graspcl_pipeline.py','graspcl_batch.py','native_resident_services.py','postprocess_pipeline.py','table_truncation_services.py']}}
    save(out/'audit.json',summary);save(out/'逐例排除核查.json',per_case);save(out/'反馈来源.json',feedback)
    shutil.copy2(__file__,out/'source.py')
    return summary


def publish(out,data):
    resources=CODE/'docs/resource'/out.name;resources.mkdir(parents=True,exist_ok=False)
    for name in ['audit.json','逐例排除核查.json','反馈来源.json','反馈来源.csv']:
        shutil.copy2(out/name,resources/name)
    prefix='resource/'+out.name+'/'
    lines=['# GraspCL 数据来源与可达性核查','',
        '核查结论：GraspCL 确实使用同一套 benchmark 旧运行中的 12 个成功示教和 13 个失败反馈。25 个来源场景均包含在后来的 200 例评测名单中。逐例排除有效，但整体不是固定训练集与评测集完全分离的测试。','',
        '- 12 个成功记录用于评分记忆及物体/局部示教召回；13 个失败记录仅以标签 0 用于评分，不作为成功示教召回。失败样本均来自已执行到闭爪后的失败，不是 IK 不可达标签。',
        '- 原离线设计从前 40 个场景收集示教、用后面的场景诊断；扩为 200 例时，将 25 个带标签来源场景也纳入总评测，改成测试每例时排除该例自身。',
        '- 库覆盖 25 种目标物体；200 例中有 125 例使用这些物体，其中 25 例是记忆来源自身，另 100 例是同物体的其他场景。',
        '- 已核查 305 份推理排除记录及 110 次重评分排除记录；直接同 ID 反馈、召回及评分特征掩码均未发现违规。运行时库文件与冻结库相同，未发现本轮失败结果在线回写。',
        '- 逐例排除同 ID 不会排除同物体的其他布局。例如 B0031 布丁盒的历史失败在测试 B0031 时被排除，在测试 B0032 时仍可作为负反馈使用。','',
        '原 200 例中，ZeroGrasp 初始规划通过 57 例，GraspCL 通过 96 例。两者都通过 56 例，仅 GraspCL 通过 40 例，仅 ZeroGrasp 通过 1 例。40 个新增通过场景选中的全部是其他场景的成功示教召回位姿（物体召回 27、局部召回 13）。GraspCL 的 96 个通过计划中共有 60 个选用了示教召回，36 个选用 ZeroGrasp 候选。','',
        '这说明两条链路的候选集合不同。尚未做去除负反馈、仅重评分、仅召回等消融，不能把可达性增益单独归因于失败反馈或评分模型。两分支分别重新推理，195 对原生候选数组中仅 7 对逐位一致，188 对存在差异；它们也不是共享固定候选池的评分消融。','',
        '| 原 200 例的回顾式重统计 | 场景数 | ZeroGrasp 规划通过 / 抓取成功 | GraspCL 规划通过 / 抓取成功 |',
        '|---|---:|---:|---:|',
        '| 全部原分配 | 200 | 57 / 34 | 96 / 63 |',
        '| 排除 25 个记忆来源场景 | 175 | 49 / 28 | 82 / 52 |',
        '| 排除记忆涉及的全部物体型号 | 75 | 23 / 12 | 34 / 19 |','',
        '以上两个排除子集是对已有结果的事后核查，不是新登记、重新运行的独立 benchmark。现有优势并不只出现在那 25 个来源场景上，但目前整体 200 例应标为历史反馈适配后的同域对照。','',
        '43 mm 截断试验还存在比较口径差异：ZeroGrasp 测其桌面拦截 72 例，GraspCL 测其桌面拦截 55 例，不能把两组总比率直接当作相同样本上的优劣。共同 44 例中，固定朝向规划通过数为 6 对 25，放开朝向为 18 对 36。','',
        '原文件中的“独立判定通过”指执行轨迹与物理成功条件的独立复核，不表示训练/评测数据独立。`test_outcomes_reused: false` 也不足以说明没有使用同 benchmark 的历史标签；其有效范围应限定为当前运行结果不回灌。','',
        '后续严格评测需要在测试前另行固定训练/示教来源、评测场景和物体划分，禁止评测反馈入库；完整召回链路与共享候选池的重评分对照分别报告。本次核查未重跑、删除或改写任何物理结果、模型及冻结登记。','',
        f'[完整数据与来源](<{prefix}audit.json>) · [25 条反馈清单](<{prefix}反馈来源.csv>) · [逐例排除记录](<{prefix}逐例排除核查.json>)','']
    path=CODE/'docs/GraspCL数据来源与可达性核查.md';path.write_text('\n'.join(lines))
    correction={'created_at':data['created_at'],'scope_correction_only':True,'raw_results_changed':False,
        'models_changed':False,'same_benchmark_historical_feedback':{'positive':12,'negative':13},
        'same_id_exclusion_verified':True,'overall_disjoint_train_test_split':False,
        'raw_physical_trace_verification_still_valid':True,'report':str(path),'audit':str(out/'audit.json')}
    for root in [PAIR,TRUNC]:save(root/'evaluation_scope_correction.json',correction)
    record=CODE/'抓取基线批次记录.md'
    notice='> 评测口径更正：GraspCL 使用同一 benchmark 历史结果构成的 12 个成功示教和 13 个失败反馈；逐例排除自身，但整体训练/评测名单并未分离。现有结果属于历史反馈适配后的对照，“独立判定”仅指物理轨迹复核。见 [数据来源核查](docs/GraspCL数据来源与可达性核查.md)。\n\n'
    text=record.read_text()
    if '评测口径更正：GraspCL' not in text:record.write_text(notice+text)
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();data=audit(out);report=publish(out,data)
    print(json.dumps({'bank':data['bank'],'query_audit':data['query_audit'],'planning_pairs':data['planning_pair_counts'],
        'exclusion_errors':data['invalid_exclusions'],'feature_hash_errors':data['invalid_feature_masks'],'report':str(report)},ensure_ascii=False))
