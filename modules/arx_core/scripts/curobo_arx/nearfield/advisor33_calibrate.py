"""One read-only check of all20 advisor32 archived ARX observations."""
import argparse
from pathlib import Path
import numpy as np
from scripts.curobo_arx.data_factory.common import ROOT, read, write, sha
from training.nearfield_arx.projection import Projector
from .advisor33_geometry import observe, first_plane


def run(out):
    out = Path(out); old = ROOT/'runs/nearfield_advisor32_20261111_01/diagnosis_complete/review.json'
    review = read(old); reg = read(ROOT/'runs/nearfield_closedloop30_20261110_01/registration.json')
    projector = Projector(reg['learned_policy']['robot_usd']); rows = []
    for item in review['camera_views']:
        row = read(item['observation']); asset = next(a for a in reg['catalog']['objects'] if a['object_id']==row['object_id'])
        gate, masks, points = observe(projector, asset['usd_path'], row, np.load(item['depth'], allow_pickle=False))
        first = first_plane(row['snapshot']['link_world']['right_eef_link'], 'REFINE 0 0 0 0 0 0', points, gate['box'])
        ep = Path(item['observation']).parent
        if ep.name == 'policy': ep = ep.parent
        trial = read(ep.parent.parent/'trial_result.json')
        positive = trial['success'] and item['stage']=='policy_3'
        rows.append({'trial_id': item['trial_id'], 'stage': item['stage'], 'object_id': row['object_id'],
                     'historical_physical_success': trial['success'], 'required_closure_positive': positive,
                     'gate': gate, 'zero_lateral_first_plane': first,
                     'files': {p: sha(p) for p in (item['observation'], item['depth'])}})
    positives = [r for r in rows if r['required_closure_positive']]
    entries = [r for r in rows if r['stage']=='H' and r['historical_physical_success']]
    result = {'pass': len(rows)==20 and len(positives)==4 and all(r['gate']['pass'] for r in positives)
                       and all(r['zero_lateral_first_plane']['pass'] for r in entries),
              'rows': rows, 'historical_successful_preclose_states': len(positives),
              'false_rejections': sum(not r['gate']['pass'] for r in positives),
              'physical_trials': 0, 'model_inferences': 0, 'thresholds_fit_to_success_labels': False,
              'geometry_source': 'Frozen ARX distal pads and fingertip; current measured finger states',
              'source_files': {str(p): sha(p) for p in [old, Path(__file__), Path(__file__).with_name('advisor33_geometry.py')]}}
    write(out, result)
    if not result['pass']: raise RuntimeError('Historical compatibility check failed; no physical launch')
    return {'pass': True, 'observations': len(rows), 'preclose_positives': len(positives), 'false_rejections': 0}


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out', required=True); print(run(parser.parse_args().out))
