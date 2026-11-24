"""Paired ARX physical trials with frozen ZeroGrasp goals and two bound weights."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
out=root/'runs/weight_comparison_20261015'
out.mkdir(exist_ok=False)
sources=[root/'runs/tabletop_trajopt_pilot_20261015/01_rubiks_cube']+sorted((root/'runs/tabletop_trajopt_batch_20261015').glob('[0-9][0-9]_*'))
manifest=[]
for i,source in enumerate(sources):
    name=json.loads((source/'result.json').read_text())['name']
    for weight in ([1,100] if i%2==0 else [100,1]):
        dest=out/f'{i+1:02d}_{name}_w{weight}'
        env=os.environ.copy();env.update(GRASP_FIXED_SELECTION=str(source/'selected_grasp.json'),GRASP_BOUND_WEIGHT_MULTIPLIER=str(weight))
        print('PAIR',name,'WEIGHT',weight,flush=True)
        code=subprocess.call([sys.executable,str(root/'scripts/curobo_arx/tabletop_batch.py'),'--output',str(dest),'--names',name,'--flat-remote','--planner','prm_trajopt','--timeout','900'],env=env,cwd=root)
        result_path=dest/'results.json'
        if not result_path.exists():raise RuntimeError(f'Batch failed: {dest}, code {code}')
        result=json.loads(result_path.read_text())[0]
        folder=dest/f'01_{name}'
        optimizations={p.stem:json.loads(p.read_text()) for p in folder.glob('*_optimization.json')}
        executions=[p.stem.removesuffix('_execution') for p in folder.glob('*_execution.json')]
        row={'name':name,'weight_multiplier':weight,'success':result['success'],'error':result.get('error'),'executed_stages':executions,'started_execution':bool(executions),'lift_executed':'lift' in executions,'optimizations':optimizations,'folder':str(folder)}
        manifest.append(row);(out/'comparison.json').write_text(json.dumps(manifest,indent=2))
        print('PAIRED_RESULT',json.dumps({k:v for k,v in row.items() if k not in ('optimizations','folder')}),flush=True)
        if code!=0:raise RuntimeError(f'Batch unavailable: {code}')
print('COMPARISON_COMPLETE',flush=True)
