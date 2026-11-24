"""Run a pinned augmentation package with the registered real project root."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('job');p.add_argument('--request',required=True)
    p=sub.add_parser('run');p.add_argument('campaign');p.add_argument('--compute-uuid',required=True);p.add_argument('--sim-uuid',required=True)
    args=parser.parse_args()
    if args.action=='job':
        request=json.loads(Path(args.request).read_text());config=json.loads(Path(request['config']).read_text())
    else:
        config=json.loads((Path(args.campaign)/'registration.json').read_text())
    # Resolve assets, interpreters and derived caches against the actual project;
    # Python imports still come from this immutable source snapshot.
    from scripts.curobo_arx.data_factory import common
    common.ROOT=Path(config['project_root']).resolve()
    if args.action=='job':
        try:
            from .production import job
            job(Path(args.request))
        except Exception as error:
            import traceback
            common.write(request['result'],{'status':'blocked','error':repr(error),
                'traceback':traceback.format_exc(),'stage':'pinned_helper_import'})
    else:
        from .campaign import run
        run(args.campaign,args.compute_uuid,args.sim_uuid)


if __name__=='__main__':main()
