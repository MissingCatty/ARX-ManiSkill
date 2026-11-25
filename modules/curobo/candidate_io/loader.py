"""Load registered staged modules only inside an isolated validation process."""
import importlib
import importlib.util
from pathlib import Path
import sys


def load_file(name, path):
    parent = name.rpartition('.')[0]
    if parent:
        importlib.import_module(parent)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if parent:
        setattr(sys.modules[parent], name.rsplit('.', 1)[1], module)
    return module


def load_staged(root, *, planning=True, client=True, service=False):
    root = Path(root)/'staged_source'
    loaded = {}
    if client:
        loaded['client'] = load_file('arx_project.client', root/'arx_project/client.py')
    if service:
        loaded['service'] = load_file('arx_project.service', root/'arx_project/service.py')
    if planning:
        for name in ['modules.curobo.dual_arm.adapter', 'modules.curobo.dual_arm.accelerated_adapter',
                     'modules.curobo.dual_arm.placement_service', 'modules.curobo.fingertip_contact.planning']:
            loaded[name] = load_file(name, root/(name.replace('.', '/')+'.py'))
    return loaded
