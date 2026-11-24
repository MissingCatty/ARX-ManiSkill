"""Prepare a fresh 49-case campaign from the original EGAD ARX transfer."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parents[3]


def prepare(source,out,initial):
    source=source.resolve();out=out.resolve()
    read=lambda name:json.loads((source/name).read_text())
    grouped=defaultdict(list)
    for scenario in read('scenarios.json'):grouped[scenario['egad_id']].append(scenario)
    expected={f'{letter}{number}' for letter in 'ABCDEFG' for number in range(7)}
    if set(grouped)!=expected or any(len(pair)!=2 for pair in grouped.values()):
        raise ValueError('Expected exactly the 49 original two-policy pairs')
    cases=[]
    for index,(name,pair) in enumerate(sorted(grouped.items())):
        for key in ('egad_id','asset','target','placement'):
            if pair[0][key]!=pair[1][key]:raise ValueError('Historical reset inputs differ: '+name)
        case={k:pair[0][k] for k in ('egad_id','asset','target','placement','case_id')}
        case.update(name=f'{index:02d}_{name}_lilo',source_scenario_names=[s['name'] for s in pair]);cases.append(case)
    assets=read('egad_catalog.json')
    for asset in assets:
        path=ROOT/'asset/egad49_arx_20261020/usd'/Path(asset['container_path']).name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=asset['usd_sha256']:
            raise ValueError('Original USD changed: '+asset['egad_id'])
    catalog=(json.dumps(read('original_tabletop_catalog.json')+assets,indent=2)+'\n').encode()
    catalog_sha=hashlib.sha256(catalog).hexdigest()
    guard=read('configuration_guard.json')['file_sha256']
    expected_catalog=next(h for p,h in guard.items() if p.endswith('/runs/arx_grasp/tabletop_catalog.json'))
    if catalog_sha!=expected_catalog:raise ValueError('Reconstructed catalog differs from historical catalog')
    expected_setup=next(h for p,h in guard.items() if p.endswith('/runs/tabletop_trials/initial_setup.json'))
    if initial.exists() and hashlib.sha256(initial.read_bytes()).hexdigest()!=expected_setup:
        raise ValueError('Provided initial setup differs from historical file')
    out.mkdir(parents=True,exist_ok=False);ref=out/'runtime/historical_reference';ref.mkdir(parents=True)
    write=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n')
    write(out/'scenarios.json',cases);(ref/'catalog.json').write_bytes(catalog)
    write(ref/'setup.json',{'fingerprint':read('expanded_environment_fingerprint.json')})
    for name in ('dataset_manifest.json','asset_preparation_contract.json','original_viewport.json',
                 'configuration_guard.json','egad_catalog.json'):
        shutil.copyfile(source/name,out/name)
    if initial.exists():shutil.copyfile(initial,ref/'initial_setup.json')
    write(out/'preparation.json',{'source_dataset':str(source),'new_attempts':49,'completed_attempts':0,
        'initial_setup_expected_sha256':expected_setup,'historical_initial_setup_sha256_verified':initial.exists(),
        'catalog_reconstructed_exact_historical_sha256':catalog_sha,'all_49_usd_sha256_match':True,
        'no_scored_trials_started':True,'created_unix':time.time()})
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'runs/egad49_arx_20261020')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--initial-setup',type=Path,default=ROOT/'runs/tabletop_trials/initial_setup.json')
    args=parser.parse_args();print(prepare(args.source,args.output,args.initial_setup))
