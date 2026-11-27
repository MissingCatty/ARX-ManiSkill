"""Run the same frozen metrics on exported native PhysX convex vertices."""
import argparse
from pathlib import Path
import traceback
from unittest.mock import patch

import numpy as np

from arx_project.layout import ROOT, now, read_json, write_json, sha256, ensure_run_path
from . import geometry_fit as base


def loader(source):
    index = read_json(source/'index.json')
    def load(asset, arm='right'):
        for name, digest in index['input_sha256'].items():
            if sha256(name) != digest:
                raise ValueError('Physical source geometry changed: '+name)
        result = {};paths = [source/'index.json', *map(Path, index['input_sha256'])]
        for record in index['records']:
            if record['arm'] != arm:
                continue
            path = ROOT/record['file']
            if sha256(path) != record['sha256']:
                raise ValueError('Native cooked geometry changed')
            with np.load(path, allow_pickle=False) as values:
                result[record['canonical_link']] = [values['part_'+str(i)] for i in range(record['parts'])]
            paths.append(path)
        return result, paths
    return load


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--native-geometry', type=Path, required=True)
    args = parser.parse_args()
    root, source = ensure_run_path(args.root), ensure_run_path(args.native_geometry)
    with patch.object(base, 'collision_meshes', loader(source)):
        if args.stage == 'prepare':
            base.prepare(root)
            reg = read_json(root/'registration.json')
            reg.update(native_geometry_source=str(source.relative_to(ROOT)),
                actual_native_PhysX_convex_vertices=True,
                raw_geometry_predecessor='tmp/sphere_count_diagnosis/geometry_fit_20261127_01',
                raw_geometry_results_preserved=True)
            for p in [Path(__file__), Path(__file__).with_name('native_cooking.py')]:
                reg['sources'][str(p)] = sha256(p)
            write_json(root/'registration.json', reg)
        else:
            reg = read_json(root/'registration.json')
            if reg['native_geometry_source'] != str(source.relative_to(ROOT)):
                raise ValueError('Native geometry source differs from registration')
            try:
                base.run(root)
            except BaseException:
                write_json(root/'STOP', {'at': now(), 'error': traceback.format_exc(), 'automatic_retry': False})
                raise
