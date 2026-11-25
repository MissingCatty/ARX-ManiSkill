"""Wait for a registered independent GPU check before resuming unclaimed slots."""
import argparse
import os
from pathlib import Path
import time

from arx_project.artifacts import atomic_json
from arx_project.json_io import read_json
from arx_project.layout import ROOT,now,sha256
from modules.pipeline.experiments.sam3_zerograsp_comparison.io_parallel.campaign import guard


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root
    reg=guard(root)
    dependency=ROOT/reg['startup_dependency']['acceptance']
    fault=ROOT/reg['startup_dependency']['fault']
    while not dependency.exists():
        guard(root)
        if fault.exists():raise RuntimeError('Independent reconstruction interface check failed')
        atomic_json(root/'preflight/startup_wait.json',{'at':now(),'reason':'waiting_for_independent_reconstruction_interface_check',
            'acceptance':str(dependency.relative_to(ROOT))})
        time.sleep(10)
    if not read_json(dependency)['passed']:raise RuntimeError('Independent interface check not accepted')
    atomic_json(root/'preflight/startup_dependency.json',{'at':now(),'passed':True,
        'acceptance':str(dependency.relative_to(ROOT)),'acceptance_sha256':sha256(dependency)})
    python=str(ROOT/'environments/simulation/bin/python')
    os.execv(python,[python,'-u','-m','modules.pipeline.experiments.sam3_zerograsp_comparison.io_parallel.campaign','run','--root',str(root)])


if __name__=='__main__':main()
