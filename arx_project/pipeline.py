"""Declarative stage orchestration; algorithms are injected through clients."""
import json
from pathlib import Path
import time
from .contracts import Stage,StageOutcome
from .layout import ROOT,ensure_run_path,write_json,sha256,method_name


def stages_from_config(name):
    method_name(name)
    file=ROOT/'configs/pipelines'/(name+'.json')
    if file.parent!=(ROOT/'configs/pipelines'):raise ValueError('Invalid pipeline name')
    data=json.loads(file.read_text())
    return [Stage(id=r['id'],module=r['module'],required_files=tuple(r.get('requires',[])),
        produced_files=tuple(r.get('produces',[])),options=r.get('options',{})) for r in data['stages']]


class Pipeline:
    def __init__(self,stages,clients):
        self.stages=list(stages);self.clients=clients
        if len({s.id for s in self.stages})!=len(self.stages):raise ValueError('Duplicate stage ID')
        if any(s.module not in clients for s in self.stages):raise ValueError('Missing module client')

    @staticmethod
    def artifact(directory,name):
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Artifact must be trial-relative: '+name)
        result=(directory/path).resolve()
        if not result.is_relative_to(directory):raise ValueError('Artifact escapes trial: '+name)
        return result

    def run(self,directory):
        directory=ensure_run_path(directory);outcomes=[];artifacts={}
        # Validate the entire configuration before invoking any worker.
        for stage in self.stages:
            for name in (*stage.required_files,*stage.produced_files):self.artifact(directory,name)
        for stage in self.stages:
            for name in stage.required_files:
                if not self.artifact(directory,name).is_file():raise FileNotFoundError(stage.id+': '+name)
            started=time.perf_counter()
            result=self.clients[stage.module].request(directory,**stage.options)
            if 'status' in result:
                if result['status']!='ok' or 'result' not in result:raise RuntimeError(str(result))
                result=result['result']
            success=bool(result.get('success')) and result.get('target_found',True) and result.get('point_count',1)>0 and result.get('count',1)>0
            if success:
                for name in stage.produced_files:
                    if not self.artifact(directory,name).is_file():raise RuntimeError(stage.id+' missing output: '+name)
                    artifacts[name]=sha256(self.artifact(directory,name))
            outcome=StageOutcome(stage.id,'ok' if success else 'failed',time.perf_counter()-started,result)
            outcomes.append(outcome)
            write_json(directory/'pipeline_progress.json',{'stages':[o.__dict__ for o in outcomes],
                'artifacts':artifacts})
            if not success:break
        return outcomes
