"""Rebind owned environments after moving the project to a new parent folder."""
import json
import os
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def main():
    aliases=json.loads((ROOT/'configs/path_aliases.json').read_text())['aliases']
    pairs=sorted(aliases.items(),key=lambda row:len(row[0]),reverse=True)
    def rewrite(text):
        for source,target in pairs:text=text.replace(source,str(ROOT/target))
        return text
    sites=list((ROOT/'environments').glob('*/lib/python*/site-packages'))
    sites+=list((ROOT/'environments/runtime_bundle').glob('*/lib/python*/site-packages'))
    rewritten=[]
    for site in sites:
        if not site.is_dir():continue
        (site/'arx_project_root.pth').write_text(str(ROOT)+'\n')
        # Editable installs and shebangs carry the old absolute project prefix.
        for file in [*site.glob('*.pth'),*site.glob('__editable__*.py')]:
            text=file.read_text()
            updated=rewrite(text)
            if updated!=text:file.write_text(updated);rewritten.append(str(file.relative_to(ROOT)))
        environment=site.parents[2]
        for file in [environment/'pyvenv.cfg',*list((environment/'bin').iterdir())]:
            if file.is_symlink():
                original=os.readlink(file);updated=rewrite(original)
                if updated!=original:file.unlink();file.symlink_to(updated)
            elif file.is_file():
                with file.open('rb') as stream:prefix=stream.read(256)
                if prefix.startswith(b'#!') or file.name=='pyvenv.cfg':
                    try:text=file.read_text()
                    except UnicodeDecodeError:continue
                    updated=rewrite(text)
                    if updated!=text:file.write_text(updated)
    print(json.dumps({'project':str(ROOT),'environments_bound':len(sites),'rewritten':rewritten},indent=2))


if __name__=='__main__':main()
