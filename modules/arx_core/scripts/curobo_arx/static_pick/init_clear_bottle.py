"""Expand the committed bottle_clear_00 reference into a fresh run directory."""
from arx_project.layout import canonical_path
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = Path(__file__).with_name('reference')


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def initialize(output):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite a previous run: {output}')
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Reference checksum mismatch: {name}')
    catalog = json.loads((BUNDLE / 'historical_reference/catalog.json').read_text())
    asset_hashes = json.loads((BUNDLE / 'historical_reference/catalog_asset_sha256.json').read_text())
    for asset in catalog:
        relative = asset.pop('asset_relative_path')
        path = canonical_path(ROOT / 'asset' / relative)
        if not path.is_file():
            raise FileNotFoundError(f'Missing original asset: {path}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != asset_hashes[relative]:
            raise ValueError(f'Original asset changed: {path}')
        asset['path'] = str(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free <= 1073741824:
        raise RuntimeError('Keep at least 1 GiB free; select another output directory')
    output.mkdir()
    reference = output / 'historical_reference'
    shutil.copytree(BUNDLE / 'historical_reference', reference)
    compressed = reference / 'setup.json.gz'
    (reference / 'setup.json').write_bytes(gzip.decompress(compressed.read_bytes()))
    compressed.unlink()
    save(reference / 'catalog.json', catalog)
    save(reference / 'catalog_asset_sha256.json', {
        str(canonical_path(ROOT / 'asset' / name)): digest for name, digest in asset_hashes.items()
    })
    asset = next(a for a in catalog if a['name'] == 'bbq_sauce_bottle')
    save(reference / 'active_asset.json', asset)
    save(output / 'active_asset.json', asset)
    shutil.copy2(reference / 'registration.json', output / 'registration.json')
    (output / 'environment_evidence').mkdir()
    save(output / 'environment_evidence/reference_manifest.json', manifest)
    print(f'Initialized {output}; all {len(catalog)} original catalog asset hashes match.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    initialize(parser.parse_args().output)
