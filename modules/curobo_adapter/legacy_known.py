"""Read-only legacy-environment comparison on the same certified FK targets."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'maniskill/arx_pipeline'))
from planner import Planner


def main() -> None:
    import curobo,importlib.metadata
    root=HERE/'research/20261117_01';out=root/'known_legacy_original';out.mkdir(exist_ok=False)
    scope=json.loads((root/'scope.json').read_text());known=json.loads((root/'known_goals.json').read_text())
    results=[]
    for identifier,targets in known.items():
        trial=Path(scope['source_campaign'])/'trials'/identifier;state=json.loads((trial/'state.json').read_text())
        folder=out/identifier;folder.mkdir();planner=Planner(folder/'model',state);start=planner.measured(state);records=[]
        with (folder/'requests.jsonl').open('x') as requests,(folder/'calls.jsonl').open('x') as calls:
            for i,item in enumerate(targets):
                witness=torch.tensor(item['q_witness'],dtype=torch.float32,device='cuda')
                assert planner.checker.validate(witness[None,None]).all(),'Certified target witness changed validity'
                requests.write(json.dumps({'index':i,'world_goal':item['world_goal']})+'\n');requests.flush()
                q=planner.ik(np.asarray(item['world_goal']),start)
                record={'index':i,'accepted':q is not None,'q':None if q is None else q.tolist()}
                calls.write(json.dumps(record)+'\n');calls.flush();records.append(record)
        result={'case':identifier,'calls':len(records),'accepted':sum(r['accepted'] for r in records)}
        (folder/'result.json').write_text(json.dumps(result,indent=2));results.append(result);print(result,flush=True)
        if hasattr(planner.planner,'destroy'):planner.planner.destroy()
        del planner
        import gc
        gc.collect();torch.cuda.empty_cache()
    result={'version':importlib.metadata.version('nvidia-curobo'),'source':curobo.__file__,
            'calls':sum(r['calls'] for r in results),'accepted':sum(r['accepted'] for r in results),
            'results':results,'physical_trials':0}
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
