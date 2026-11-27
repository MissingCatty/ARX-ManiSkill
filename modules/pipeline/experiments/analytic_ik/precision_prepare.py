"""Prepare independent CPU references without importing Torch or CUDA."""
import argparse
from pathlib import Path

from arx_project.layout import read_json, write_json, ensure_run_path
from .prepare import prepare


def register(root,gpu):
    prepare(root,gpu)
    reg=read_json(root/'registration.json')
    reg.update(variant='FP64_algebra_with_verified_FP32_commands',
        input_dtype='float64_original_targets',command_dtype='float32_checked_against_original_target',
        precision_predecessor='tmp/analytic_ik/batch256_20261127_01',
        rotation_normalization='internal_only; acceptance uses original raw target')
    write_json(root/'registration.json',reg)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--gpu',required=True);args=parser.parse_args()
    register(ensure_run_path(args.root),args.gpu)
