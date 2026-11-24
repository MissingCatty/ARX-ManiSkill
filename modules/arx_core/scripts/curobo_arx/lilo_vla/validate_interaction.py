"""Offline ARX contract receipt: recorded observations and legacy data regression.

Never sends simulation commands, loads a model, or creates new demonstrations.
Run with the original dataset-builder Python for exact historical numerics.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image

from interaction_contract import ROOT, load, resolve, write, file_sha, adapter_compatibility, digest
from observations import policy_observation, encode_state, mask_distractors
from build_dataset import build


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--adapter',type=Path,default=ROOT/'runs/lilo_P2_comparison_20261028_02/goal_adapter')
    parser.add_argument('--sources',type=Path,default=ROOT/'runs/lilo_P2_collection_20261028_05/dataset_sources.json')
    parser.add_argument('--baseline',type=Path,default=ROOT/'runs/lilo_P2_comparison_20261028_02/goal_dataset')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(out).free<=1073741824:raise RuntimeError('Local 1GiB reserve reached')
    frozen=[ROOT/p for p in ('arx-sim/config/sim.yaml','arx-sim/config/runtime.yaml',
        'arx-sim/config/controllers.yaml','configs/calibration.yaml','asset/robots/arx_x5_lift/1.0.0/usd/robot.usd')]
    frozen += [args.adapter/n for n in ('action_head.pt','dataset_statistics.json','adapter_manifest.json')]
    before={str(p):file_sha(p) for p in frozen}
    report={'scope':'Offline protocol/data checks only; no new physical qualification or successful demonstrations',
            'physical_execution_performed':False,'cuda_or_model_loading':False,'profiles':{},'legacy_relabeling':False}
    for name in ('handoff_10mm','handoff_40_60_80mm'):
        config=load(ROOT/'configs/lilo_vla'/f'{name}.json')
        pair={role:resolve(config,'mustard_bottle',role) for role in ('teacher','policy')}
        assert pair['teacher']['contract_sha256']==pair['policy']['contract_sha256']
        for role,value in pair.items():write(out/f'{name}_{role}_resolved.json',value)
        report['profiles'][name]={'common_hash_matches':True,'contract_sha256':digest(config)}
    compatibility=adapter_compatibility(args.adapter,config);write(out/'model_compatibility.json',compatibility)
    sources=json.loads(args.sources.read_text());observations=[]
    for source in sources['train']+sources['validation']:
        source=Path(source);goal=json.loads((source/'selected_grasp.json').read_text())['grasp_world']
        rows=[json.loads(x) for x in (source/'teacher_raw/samples.jsonl').read_text().splitlines()]
        selected=rows[::max(1,len(rows)//8)]
        for row in selected:
            image,boxes,state=policy_observation(row,'grasp_goal_world_error_v1',goal)
            with Image.open(row['rgb_path']) as raw:old_image,old_boxes=mask_distractors(raw,row)
            np.testing.assert_array_equal(image,old_image);assert boxes==old_boxes
            np.testing.assert_array_equal(state,encode_state(row['snapshot'],'grasp_goal_world_error_v1',goal))
        observations.append({'source':str(source),'frames_checked':len(selected),'image_and_state_equal':True})
    report['recorded_observations']=observations
    with tempfile.TemporaryDirectory(prefix='legacy_rebuild_',dir=out) as temporary:
        rebuilt=Path(temporary)/'dataset'
        with contextlib.redirect_stdout(io.StringIO()):
            build([Path(p) for p in sources['train']],rebuilt,[Path(p) for p in sources['validation']],
                state_convention='grasp_goal_world_error_v1',dataset_key='arx_shared_goal_v1',object_split=True,
                require_entry=True,include_command_starts=True,exclude_gated_dwell_starts=True)
        current=json.loads((rebuilt/'dataset.json').read_text());baseline=json.loads((args.baseline/'dataset.json').read_text())
        assert len(current['episodes'])==len(baseline['episodes'])==8
        for a,b in zip(current['episodes'],baseline['episodes']):
            for key in ('instruction','object_name','split','chunk_starts','selected_grasp_world'):
                assert a[key]==b[key],key
            a_np=np.load(rebuilt/a['folder']/'trajectory.npz');b_np=np.load(args.baseline/b['folder']/'trajectory.npz')
            for key in ('state','action'):np.testing.assert_array_equal(a_np[key],b_np[key])
        # Dataset keys differ between historical adapters; compare the statistics contents.
        stats=json.loads((rebuilt/'dataset_statistics.json').read_text())
        old_stats=json.loads((args.baseline/'dataset_statistics.json').read_text())
        assert next(iter(stats.values()))==next(iter(old_stats.values()))
        report['legacy_dataset_regression']={'episodes':len(current['episodes']),
            'train_chunks':sum(len(e['chunk_starts']) for e in current['episodes'] if e['split']=='train'),
            'validation_chunks':sum(len(e['chunk_starts']) for e in current['episodes'] if e['split']=='validation'),
            'states_actions_chunks_instructions_statistics_exactly_equal':True}
    after={str(p):file_sha(p) for p in frozen};assert before==after
    report.update(pass_check=True,frozen_sources_and_policy_unchanged=True,frozen_sha256=after,
        live_runtime_properties='Not inspected; this is an offline validation',model_compatibility=compatibility,
        future_dependency='D1 still must remove dual-fixture scheduling dependencies before new collection')
    write(out/'validation.json',report)
    print(json.dumps({'pass':True,'output':str(out),'legacy':report['legacy_dataset_regression']}))


if __name__=='__main__':main()
