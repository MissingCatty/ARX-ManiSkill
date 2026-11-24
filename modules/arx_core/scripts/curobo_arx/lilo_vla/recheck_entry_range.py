"""Recompute P3 provenance in the SAME Python/SciPy used by action execution.

Keeps the original report and the original strict 1e-10 state comparison.
Physical outcomes are recomputed and must remain exactly unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy

from egad_evaluate import evaluate
from observations import encode_state
from summarize_p2 import policy_trial


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--campaign',type=Path);mode.add_argument('--trial',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.trial:
        trial=args.trial.resolve();result=json.loads((trial/'result.json').read_text())
        row=policy_trial(trial,result)
        if not row['action_provenance_pass']:raise ValueError('Strict policy provenance fails in execution environment')
        row['state_recheck_environment']={'python':sys.executable,'scipy':scipy.__version__,'numpy':np.__version__,'atol':1e-10}
        args.output.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');return
    source=args.campaign.resolve()/'summary.json'
    original=json.loads(source.read_text());rows=[];maximum=0.;states=0
    for old in original['rows']:
        trial=Path(old['trial']);score=evaluate(trial)
        if score!=old['evaluation']:raise ValueError('Physical evaluation changed; stop provenance-only recheck')
        row=policy_trial(trial,old);rows.append(row)
        selection=json.loads((trial/'selected_grasp.json').read_text())
        report=json.loads((trial/'grasp_report.json').read_text())
        for action in report.get('action_sources',[]):
            snapshot=json.loads((trial/f"local_{action['step']:04d}_state.json").read_text())
            prediction=json.loads((trial/action['prediction_file']).read_text())
            expected=encode_state(snapshot,row['state_convention'],selection['grasp_world'])
            maximum=max(maximum,float(np.max(np.abs(expected-prediction['raw_state']))));states+=1
    result={**original,'rows':rows,'all_action_provenance_pass':all(r['action_provenance_pass'] for r in rows),
        'original_report_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'recheck_environment':{'python':sys.executable,'scipy':scipy.__version__,'numpy':np.__version__,
            'strict_state_atol':1e-10,'compared_states':states,'maximum_absolute_state_difference':maximum,
            'strict_state_check_pass':maximum<=1e-10},
        'physical_outcomes_unchanged':True,
        'reason':'Cross-environment SciPy rotation arithmetic differed by about 2e-10 rad; authoritative recheck uses action-execution environment and unchanged thresholds'}
    if not result['all_action_provenance_pass'] or not result['recheck_environment']['strict_state_check_pass']:
        raise ValueError('Actual action/state provenance still fails in execution environment')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'states':states,'max_state_difference':maximum,'strict_provenance_pass':True,'physical_outcomes_unchanged':True}))


if __name__=='__main__':main()
