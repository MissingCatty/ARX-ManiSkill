"""Create a clean pretrained bank and add only the registered 50 executions."""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import shutil
import zipfile
from unittest.mock import patch

import graspcl_pipeline as common


def extract(bank):
    bank.mkdir(parents=True,exist_ok=False)
    components = ['config.json','encoder/best.ckpt','score_cl/config.json','score_cl/offline_data.npz']
    with zipfile.ZipFile(common.ARCHIVE) as archive:
        for name in components:
            dest = bank/'model'/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(archive.read('bootstrapped/'+name))


def receipt(bank,module,records):
    common.save(bank/'bank_complete.json',{'positive_demonstrations':sum(r['label']==1 for r in records),
        'negative_feedback':sum(r['label']==0 for r in records),
        'object_templates':len(module.grasp_recall_module_object._db),
        'patch_templates':len(module.grasp_recall_module_patch._db),
        'online_points':len(module.grasp_scoring_module_cl.X_on),'source_ids':[r['id'] for r in records],
        'model_sha256':{str(p):common.sha(p) for p in (bank/'model').rglob('*') if p.is_file()},
        'pretrained_archive':str(common.ARCHIVE),'pretrained_archive_sha256':common.sha(common.ARCHIVE),
        'contact_storage_tcp_m':.040,'planner_recall_contact_shift_m':.003})


def build(root):
    suite = common.read(root/'registration.json')
    source = root/'collection/fixed_zerograsp'
    reg = common.read(source/'registration.json')
    mapping = common.read(suite['semantic_map'])
    assert common.read(source/'progress.json')['phase']=='complete'
    baseline = root/'banks/no_experience'
    extract(baseline)
    common.seed()
    module = common.load_model(baseline)
    assert len(module.grasp_scoring_module_cl.X_on)==0
    assert not module.grasp_recall_module_object._db and not module.grasp_recall_module_patch._db
    receipt(baseline,module,[])
    # Load a separate clean copy; never mutate the B bank while building C.
    trained = root/'banks/experience50'
    extract(trained)
    module = common.load_model(trained)
    records, excluded = [], []
    for entry in reg['schedule']:
        trial = source/'trials'/entry['id']
        result = common.read(trial/'result.json')
        evaluation = common.read(trial/'independent_evaluation.json')
        assert evaluation['passed']
        claim = common.read(source/'claims'/(entry['id']+'.json'))
        assert claim['state']=='complete' and claim['result_sha256']==common.sha(trial/'result.json')
        label = None
        if result['success'] and evaluation.get('independently_successful'):
            label = 1
        elif (result['physical_attempt'] and result['reason'] in ['no_bilateral_target_contact','insufficient_hold_height']
                and result.get('stages',{}).get('close',{}).get('settled')
                and not common.read(trial/'forbidden_contacts.json')):
            label = 0
        if label is None:
            excluded.append({'id':entry['id'],'reason':result['reason'],'use':'no_completed_clean_grasp_label'})
            continue
        state = common.read(trial/'state.json')
        models = {obj['model_id'] for obj in state['objects'].values()}
        assert models <= set(reg['test_object_ids'])
        assert not {mapping[m]['category'] for m in models} & set(suite['test_categories'])
        try:
            # The newer selector stores canonical-pose metadata instead of the
            # legacy network_row field. Supply that legacy provenance view only
            # to the adapter; preserve the actual plan/result files byte-for-byte.
            original_read = common.read
            plan_path = trial/'plan_result.json'
            legacy_plan = deepcopy(original_read(plan_path))
            selected = legacy_plan['selected']
            pool = original_read(trial/'graspcl/candidate_pool.json')
            candidate = pool['candidates'][selected['index']]
            selected['network_row'] = candidate['metadata']['source_postprocessed_row']
            def read_with_row(path):
                return legacy_plan if Path(path)==plan_path else original_read(path)
            with patch.object(common,'read',read_with_row):
                grasp = common.commanded_grasp(trial,{**entry,'label':label,'original_reason':result['reason']})
        except RuntimeError as error:
            if 'command' not in str(error).lower() and 'track' not in str(error).lower(): raise
            excluded.append({'id':entry['id'],'reason':str(error),'use':'command_measurement_mismatch'})
            continue
        scene,target,_ = common.clouds(trial)
        encoded = module.encoder.predict(scene,[grasp])
        module.grasp_scoring_module_cl.add_datapoint(encoded)
        if label==1:
            args=(target,scene,deepcopy(encoded),'unknown','arx_verified_success')
            module.grasp_recall_module_object.add_demonstration(*args)
            module.grasp_recall_module_patch.add_demonstration(target,scene,deepcopy(encoded),'unknown','arx_verified_success',debug=False)
        record={'id':entry['id'],'label':label,'trial':str(trial),'grasp':grasp,
            'scene_models':sorted(models),'input_sha256':common.inputs(trial),
            'result_sha256':common.sha(trial/'result.json'),'evaluation_sha256':common.sha(trial/'independent_evaluation.json'),
            'scene_points':len(scene.points),'target_points':len(target.points)}
        common.save(trained/'demonstrations'/(entry['id']+'.json'),record)
        records.append(record)
        print('NEW EXPERIENCE',entry['id'],label,flush=True)
    module.feedback_counts={'arx_new50':{'labels':len(records),'demonstrations':sum(r['label']==1 for r in records)}}
    module.save(trained/'model')
    receipt(trained,module,records)
    common.save(trained/'excluded_collection_cases.json',excluded)
    for name in ['encoder/best.ckpt','score_cl/offline_data.npz']:
        assert common.sha(baseline/'model'/name)==common.sha(trained/'model'/name)
    valid_ids={e['id'] for e in reg['schedule']}
    assert set(common.read(trained/'bank_complete.json')['source_ids'])<=valid_ids
    common.save(root/'bank_provenance_audit.json',{'passed':True,'registered_collection_scenes':50,
        'clean_feedback_points':len(records),'positive':sum(r['label']==1 for r in records),
        'negative':sum(r['label']==0 for r in records),'excluded_without_clean_label':len(excluded),
        'same_pretrained_encoder_and_offline_memory':True,'historical_arx_feedback':0,
        'test_images_clouds_feedback_or_templates_added':0,'category_model_mesh_intersections':0,
        'banks':{b.name:common.sha(b/'bank_complete.json') for b in [baseline,trained]}})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    build(parser.parse_args().output.resolve())
