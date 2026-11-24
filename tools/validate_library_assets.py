"""Verify shared SDK meshes and textures without importing a simulator."""
import argparse
import json
import os
from pathlib import Path

from arx_project.assets import AssetStore
from arx_project.layout import ROOT, sha256


def validate_library_assets(root=ROOT):
    store = AssetStore(root / 'simulation/assets/catalog.json')
    models = (root / 'simulation/assets/models').resolve()
    checks = []
    for name, record in store.data['resources'].items():
        if 'inventory' not in record:
            continue
        errors = []
        directory = store.resource(name)
        inventory = store.path_for(record['inventory'])
        if not inventory.is_file() or sha256(inventory) != record['inventory_sha256']:
            checks.append({'resource': name, 'passed': False, 'errors': ['inventory checksum mismatch']})
            continue
        entries = json.loads(inventory.read_text())['files']
        for relative, expected in entries.items():
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts:
                errors.append('invalid relative path: ' + relative)
                continue
            file = directory / path
            if not file.resolve().is_relative_to(models):
                errors.append('asset escapes shared storage: ' + relative)
            elif 'symlink' in expected:
                if not file.is_symlink() or os.readlink(file) != expected['symlink'] or not file.exists():
                    errors.append('asset link mismatch: ' + relative)
            elif not file.is_file() or file.is_symlink():
                errors.append('missing asset: ' + relative)
            elif file.stat().st_size != expected['bytes'] or sha256(file) != expected['sha256']:
                errors.append('asset checksum mismatch: ' + relative)
        actual = {str(p.relative_to(directory)) for p in directory.rglob('*')
                  if (p.is_file() or p.is_symlink()) and p != inventory}
        if actual != set(entries):
            errors.append('asset file inventory mismatch')
        checks.append({'resource': name, 'passed': not errors, 'files': len(entries), 'errors': errors})
    return {'passed': all(c['passed'] for c in checks), 'bundles': len(checks),
            'files': sum(c.get('files', 0) for c in checks), 'checks': checks, 'physical_grasps': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    result = validate_library_assets(parser.parse_args().root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['passed']:
        raise SystemExit(2)
