"""Materialize explicit historical inputs and print (or launch) the selected fixture."""
import argparse
import os
from pathlib import Path
import subprocess

from batch_manifest import ROOT, resolve_manifest, materialize_fixture, startup
from batch_ledger import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--start',action='store_true',help='Launch only the named owned fixture; default just prepares inputs')
    args=parser.parse_args();value=resolve_manifest(args.manifest)
    runtime=Path(value['fixture']['runtime'])
    # Current launcher mounts this checkout. Refuse invisible external inputs.
    runtime.relative_to(ROOT)
    if (Path(value['fixture']['mailbox'])/'ready.json').exists():raise RuntimeError('Existing mailbox: inspect/reuse its identity; never overwrite it')
    description=startup(value);materialize_fixture(value)
    write(runtime/'fixture_registration.json',value);write(runtime/'startup.json',description)
    print(runtime/'startup.json',flush=True)
    if args.start:
        Path(value['resources']['storage_fallback_root']).mkdir(parents=True,exist_ok=True)
        subprocess.run(description['command'],cwd=ROOT,env={**os.environ,**description['environment']},check=True)


if __name__=='__main__':main()
