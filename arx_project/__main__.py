"""One entry point for layout inspection, checks and explicit worker startup."""
import argparse
import json
from pathlib import Path
from .layout import ROOT,read_json,sha256,write_json,ensure_run_path
from .registry import module_records


def doctor(verify_dataset=False):
    from .assets import resource_path
    checks=[]
    def check(name,ok):checks.append({'name':name,'passed':bool(ok)})
    for name,record in module_records().items():
        for field in ['source','environment','adapter']:
            if field in record:check(name+':'+field,(ROOT/record[field]).exists())
        for path in record.get('weights',[]):check(name+':'+path,(ROOT/path).exists())
        for resource in record.get('asset_resources',[]):
            check(name+':asset:'+resource,resource_path(resource).is_dir())
    import sys
    sys.path.insert(0,str(ROOT/'modules/pipeline/experiments/current_configuration'))
    from configuration import load_current
    config=load_current(verify_dataset=verify_dataset)
    check('current_871_configuration',config['collision_profile_id']=='arx_refined_871_v1')
    check('scene_count',config['scene_count']==200)
    check('website_snapshot',(ROOT/'website/resources/site/index.html').is_file())
    return {'passed':all(row['passed'] for row in checks),'checks':checks,'dataset_files_checked':2406 if verify_dataset else 0,'physical_grasps':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('inspect')
    doc=commands.add_parser('doctor');doc.add_argument('--verify-dataset',action='store_true');doc.add_argument('--output',type=Path)
    worker=commands.add_parser('serve');worker.add_argument('--kind',required=True)
    for name in ['root','socket','warmup-trial','descriptor']:worker.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.command=='inspect':result={'root':str(ROOT),'modules':module_records()}
    elif args.command=='doctor':result=doctor(args.verify_dataset)
    else:
        from .service import serve
        serve(args);return
    if getattr(args,'output',None):write_json(ensure_run_path(args.output),result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result.get('passed') is False:raise SystemExit(2)


if __name__=='__main__':main()
