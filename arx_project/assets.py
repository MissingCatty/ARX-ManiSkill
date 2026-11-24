"""Shared 3D assets and scene-to-asset references, independent of simulators."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from .layout import ROOT, canonical_path, read_json


ASSETS = ROOT / 'simulation/assets'
CATALOG = ASSETS / 'catalog.json'


def recipe_sha256(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


class AssetStore:
    def __init__(self, catalog=CATALOG):
        self.path = Path(catalog).resolve()
        self.root = self.path.parent
        self.data = json.loads(self.path.read_text())
        if self.data.get('kind') != 'shared_3d_asset_catalog':
            raise ValueError('Expected a shared 3D asset catalog')

    def path_for(self, value):
        path = Path(value)
        if path.is_absolute():
            raise ValueError('Asset catalog paths must be relative: ' + value)
        result = (self.root / path).resolve()
        if not result.is_relative_to((self.root / 'models').resolve()):
            raise ValueError('Asset path escapes the shared models directory: ' + value)
        return result

    def resource(self, name):
        row = self.data['resources'][name]
        return self.path_for(row.get('file', row.get('directory')))

    def model(self, name):
        record = deepcopy(self.data['models'][name])
        for item in record['visuals'] + record['collisions']:
            item['file'] = str(self.path_for(item['file']))
        record['source_definition'] = str(self.path_for(record['source_definition']))
        record['source_hashes'] = {str(self.path_for(file)): digest
                                   for file, digest in record['source_hashes'].items()}
        return record


@lru_cache(maxsize=2)
def _resource_store(filename, mtime_ns):
    return AssetStore(filename)


def resource_path(name):
    """Resolve shared library assets, refreshing after catalog changes."""
    return _resource_store(str(CATALOG), CATALOG.stat().st_mtime_ns).resource(name)


def load_object_catalog(path):
    """Read current ID-only scene bindings and older self-contained receipts."""
    document = read_json(path)
    if document.get('kind') == 'shared_3d_asset_catalog':
        store = AssetStore(canonical_path(path))
        return {**document, 'models': {name: store.model(name) for name in store.data['models']}}
    if document.get('kind') != 'shared_asset_catalog_reference':
        return document
    reference = Path(document['catalog_ref'])
    store = AssetStore(canonical_path(reference if reference.is_absolute() else ROOT / reference))
    names = document['asset_ids']
    if len(names) != len(set(names)):
        raise ValueError('Duplicate asset IDs in scene catalog reference')
    models = {}
    for name in names:
        record = store.data['models'][name]
        if recipe_sha256(record) != document['recipe_sha256'][name]:
            raise ValueError('Scene asset recipe changed: ' + name)
        models[name] = store.model(name)
    return {**document, 'models': models}
