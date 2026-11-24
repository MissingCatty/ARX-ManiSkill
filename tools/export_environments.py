"""Record the dependencies and Python versions of each isolated environment."""
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'configs/environments';output.mkdir(exist_ok=True)
    manifest=[]
    environments=sorted((ROOT/'environments').glob('*/bin/python'))
    environments+=list((ROOT/'environments/runtime_bundle').glob('env/bin/python'))
    for python in environments:
        name='runtime_bundle' if 'runtime_bundle' in python.parts else python.parent.parent.name
        version=subprocess.run([str(python),'--version'],capture_output=True,text=True,timeout=20)
        frozen=subprocess.run([str(python),'-m','pip','freeze'],capture_output=True,text=True,timeout=60)
        exporter='pip freeze'
        if frozen.returncode:
            # Several uv-created environments intentionally contain no pip.
            exporter='importlib.metadata'
            frozen=subprocess.run([str(python),'-c',"import importlib.metadata as m; print('\\n'.join(sorted(d.metadata['Name']+'=='+d.version for d in m.distributions() if d.metadata['Name'])))"],capture_output=True,text=True,timeout=60)
        if frozen.returncode==0:
            text=frozen.stdout
            aliases=json.loads((ROOT/'configs/path_aliases.json').read_text())['aliases']
            for source,target in sorted(aliases.items(),key=lambda row:len(row[0]),reverse=True):text=text.replace(source,str(ROOT/target))
            text=text.replace(str(ROOT),'${ARX_PROJECT_ROOT}')
            lines=[]
            for line in text.splitlines():
                if line.startswith('nvidia-curobo=='):
                    module='curobo_adapter' if name=='curobo_v090' else 'curobo'
                    line='-e ${ARX_PROJECT_ROOT}/modules/'+module+'/source'
                elif '#egg=mani_skill' in line:line='-e ${ARX_PROJECT_ROOT}/modules/maniskill/source'
                lines.append(line)
            text='\n'.join(lines)
            text='\n'.join(dict.fromkeys(text.splitlines()))+'\n'
            (output/(name+'.txt')).write_text(text)
            (output/(name+'.error.txt')).unlink(missing_ok=True)
        else:(output/(name+'.error.txt')).write_text(frozen.stderr)
        manifest.append({'name':name,'python':str(python.relative_to(ROOT)),'version':version.stdout.strip(),'requirements_exported':frozen.returncode==0,'exporter':exporter})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
