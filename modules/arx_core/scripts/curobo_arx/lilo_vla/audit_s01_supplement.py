"""Offline profile/source/registration verification supplement for S01."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from handoff_profile import admit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);args=p.parse_args();out=args.audit
    output=out/'supplemental_checks.json'
    if output.exists():raise FileExistsError(output)
    evidence={}
    def read(p):
        p=Path(p);raw=p.read_bytes();evidence[str(p)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    rows=[json.loads(s) for s in (out/'attempt_ledger.jsonl').read_text().splitlines()]
    profiles=[];sources=[];schedule=[]
    for row in rows:
        t=Path(row['trial']);receipt=t/'handoff_policy_admission.json'
        if receipt.exists():
            recorded=read(receipt);profile=read(recorded['profile'])
            if not recorded.get('pass'):
                profiles.append({'trial':str(t),'recorded_rejection':recorded,'status':'recorded'});continue
            pred=read(t/'policy_0000_prediction.json')
            inputs=[read(t/f) for f in ['scenario.json','selected_grasp.json','lilo_observations/0000/observation.json','handoff_actual_entry.json','setup.json']]
            instruction=read(t/'grasp_report.json')['instruction']
            result=admit(profile,pred,*inputs,instruction)
            mismatch={k:{'original':recorded.get(k),'recomputed':v} for k,v in result.items() if recorded.get(k)!=v}
            changed=copy.deepcopy(profile);changed['policy_artifact_sha256']['action_head.pt']='wrong'
            try:admit(changed,pred,*inputs,instruction);reject=False
            except ValueError:reject=True
            profiles.append({'trial':str(t),'pass':not mismatch,'mismatch':mismatch,'wrong_model_rejected':reject,'scope':list(profile['cases'])})
    manifest=read(out/'source_manifest.json')
    for campaign in manifest['expanded_campaigns']:
        root=Path(campaign);reg=read(root/'registration.json')
        for name,expected in reg.get('algorithm_sha256',{}).items():
            snap=root/'registered_source_snapshot'/Path(name).name
            if not snap.exists():
                projects=Path(__file__).resolve().parents[4]
                try:snap=root/'registered_source_snapshot'/Path(name).relative_to(projects)
                except ValueError:pass
            actual=hashlib.sha256(snap.read_bytes()).hexdigest() if snap.exists() else None
            sources.append({'campaign':campaign,'original_source':name,'snapshot':str(snap),'expected':expected,'actual':actual,'status':'missing' if actual is None else 'recorded','pass':actual==expected if actual else None})
            if actual:evidence[str(snap)]=actual
        if 'P3_' in root.name:
            results=read(root/'results.json');keys=[(r.get('entry_group'),r.get('repeat'),r['scene']) for r in results]
            if 'entry_range' in root.name:
                select=read(root/'point_selection.json');chosen=select['selected_for_repeats']
                expected=[(point,1,scene) for point in reg['initial_order'] for scene in ['A1','bottle']]+[(chosen,rep,scene) for rep in [2,3] for scene in ['A1','bottle']]
            else:expected=[('ten_mm',1,s) for s in ['A1','bottle']]
            schedule.append({'campaign':campaign,'expected':expected,'actual':keys,'pass':sorted(keys)==sorted(expected)})
    result={'profiles':profiles,'source_snapshots':sources,'adaptive_registrations':schedule,'source_hashes':evidence,
        'pass':all(x.get('pass',True) and x.get('wrong_model_rejected',True) for x in profiles) and all(x['pass'] is not False for x in sources) and all(x['pass'] for x in schedule)}
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'pass':result['pass'],'profiles':len(profiles),'source_snapshots':len(sources),'missing_snapshots':sum(x['pass'] is None for x in sources),'adaptive_campaigns':len(schedule)}))

if __name__=='__main__':main()
