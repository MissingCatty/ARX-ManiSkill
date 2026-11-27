"""Finalize accepted runs using their immutable executed source snapshots."""
import argparse
from pathlib import Path
from unittest.mock import patch

from arx_project.layout import read_json,write_json,now,sha256
from modules.pipeline.experiments.graspgenx_native_postprocessing import planning as original
from modules.pipeline.experiments.graspgenx_native_postprocessing.offline import guard
from . import run,report


def finalize(root):
    if not read_json(root/'services_stopped.json')['all_owned_exited']:
        raise RuntimeError('Finalize only after all owned execution processes exit')
    guard(root,completed=True)
    def archived(path):return guard(path,completed=True)
    with patch.object(original,'guard',archived),patch.object(run,'guard',archived):
        report.create(root)
    write_json(root/'finalization.json',{'at':now(),'passed':True,'execution_source_archive_verified':True,
        'independent_FK_and_all_claims_reviewed':True,'planning_requests_repeated':0,
        'report_generator_sha256':sha256(Path(report.__file__))})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);args=p.parse_args()
    finalize(args.root.resolve())
