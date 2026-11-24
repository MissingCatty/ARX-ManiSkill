"""Check that module, simulation and weight links stay inside the project."""
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def check_ownership(root=ROOT):
    outside=[];broken=[];links=0
    for base in ['modules','simulation','weights']:
        for folder,dirs,files in os.walk(root/base,followlinks=False):
            dirs[:]=[name for name in dirs if name not in ['.git','__pycache__']]
            for name in dirs+files:
                path=Path(folder)/name
                if not path.is_symlink():continue
                links+=1;target=path.resolve()
                row={'path':str(path.relative_to(root)),'target':str(target)}
                if not target.is_relative_to(root):outside.append(row)
                if not path.exists():broken.append(row)
    forbidden=[str(p.relative_to(root)) for p in
               [root/'modules/tmp',root/'modules/asset',root/'modules/compat_tmp']
               if p.exists() or p.is_symlink()]
    return {'passed':not outside and not broken and not forbidden,'links_checked':links,
            'outside':outside,'broken':broken,'forbidden_module_entries':forbidden}


def main():
    result=check_ownership()
    output=ROOT/'tmp/migration/validation/link_ownership.json';output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['passed']:raise SystemExit(2)


if __name__=='__main__':main()
