"""Verify the stopped, newly-created side-task container cache before removal."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path);p.add_argument('report',type=Path);a=p.parse_args()
    rows=[]
    for src in sorted(a.source.rglob('*')):
        if not src.is_file() or src.is_symlink():continue
        relative=src.relative_to(a.source);dst=a.destination/relative
        h=digest(src)
        if not dst.exists() or dst.stat().st_size!=src.stat().st_size or digest(dst)!=h:
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        assert dst.stat().st_size==src.stat().st_size and digest(dst)==h
        rows.append({'file':str(relative),'bytes':src.stat().st_size,'sha256':h})
    value={'verified':True,'scope':'Only cache of NEW side-task startup container; no original research evidence moved',
        'files':rows,'total_bytes':sum(r['bytes'] for r in rows),'destination':str(a.destination)}
    a.report.write_text(json.dumps(value,indent=2));print(json.dumps({'verified':True,'files':len(rows),'bytes':value['total_bytes']}))
