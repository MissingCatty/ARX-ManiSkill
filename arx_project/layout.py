"""Central path ownership. Modules never infer another module's location."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
from functools import lru_cache

ROOT=Path(__file__).resolve().parents[1]
SLUG=re.compile(r'^[a-z][a-z0-9_\-]*$')


def method_name(name):
    if not SLUG.fullmatch(name):raise ValueError('Invalid module/method name: '+name)
    return name


def module_dir(name):return ROOT/'modules'/method_name(name)
def weights_dir(name):return ROOT/'weights'/method_name(name)


def run_dir(method,run_id,*,create=False):
    method_name(method)
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_\-.]*',run_id):raise ValueError('Invalid run ID')
    result=ROOT/'tmp'/method/run_id
    ensure_run_path(result)
    if create:result.mkdir(parents=True,exist_ok=False)
    return result


def ensure_run_path(path):
    result=Path(path).resolve()
    parent=(ROOT/'tmp').resolve()
    if not result.is_relative_to(parent) or len(result.relative_to(parent).parts)<2:
        raise ValueError('Runtime products require tmp/<method>/<run_id>/')
    method_name(result.relative_to(parent).parts[0])
    return result


@lru_cache(maxsize=2)
def _load_aliases(filename,mtime_ns):
    data=json.loads(Path(filename).read_text())
    return sorted(data['aliases'].items(),key=lambda pair:len(pair[0]),reverse=True)


def relocation_aliases():
    file=ROOT/'configs/path_aliases.json'
    if not file.exists():return []
    return _load_aliases(str(file),file.stat().st_mtime_ns)


def relocated(value,aliases=None):
    """Resolve archived absolute metadata paths in memory; frozen bytes stay intact."""
    aliases=relocation_aliases() if aliases is None else aliases
    if isinstance(value,str):
        seen=set()
        while value not in seen:
            seen.add(value)
            for source,target in aliases:
                if value==source or value.startswith(source+'/'):
                    target=Path(target)
                    if not target.is_absolute():target=ROOT/target
                    updated=str(target)+value[len(source):]
                    if updated==value:return value
                    value=updated
                    break
            else:return value
        raise ValueError('Cyclic archived path mapping')
    if isinstance(value,list):return [relocated(v,aliases) for v in value]
    if isinstance(value,dict):return {relocated(k,aliases):relocated(v,aliases) for k,v in value.items()}
    return value


def canonical_path(path):return Path(relocated(str(Path(path).absolute())))


def read_json(path):return relocated(json.loads(canonical_path(path).read_text()))


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.pending')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=lambda v:v.tolist(),allow_nan=False)+'\n')
    temp.replace(path)


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(8*1024*1024):digest.update(chunk)
    return digest.hexdigest()


def now():return datetime.now().astimezone().isoformat()
