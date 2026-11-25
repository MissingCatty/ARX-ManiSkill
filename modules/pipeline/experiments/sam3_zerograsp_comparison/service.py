"""Bind the existing ZeroGrasp module to its full native grasp adapter."""
import argparse
from pathlib import Path

from arx_project.service import serve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['zerograsp'], required=True)
    for name in ['root', 'socket', 'warmup-trial', 'descriptor']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    from modules.zerograsp.arx_grasp_adapter import create
    serve(args, worker=create(args))


if __name__ == '__main__':
    main()
