"""Load one registered adapter without importing other algorithm modules."""
import importlib.util
import json
from pathlib import Path
from .layout import ROOT,method_name


def module_records():return json.loads((ROOT/'configs/modules.json').read_text())['modules']


def load_component(name,field='adapter'):
    method_name(name);record=module_records()[name]
    file=(ROOT/record[field]).resolve()
    if not file.is_relative_to((ROOT/'modules').resolve()):raise ValueError('Adapter must be owned by modules/')
    spec=importlib.util.spec_from_file_location('arx_'+field+'_'+name,file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def load_adapter(name):return load_component(name)


def create_worker(name,args):return load_adapter(name).create(args)
