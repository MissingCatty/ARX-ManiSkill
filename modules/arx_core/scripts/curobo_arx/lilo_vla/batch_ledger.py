"""Durable attempt registration, immutable terminal receipts and continuation."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time
import uuid

from interaction_contract import digest, read


def write(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    with temp.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    temp.replace(path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def append(path,value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def process_identity(pid=None):
    pid=os.getpid() if pid is None else pid
    return {'pid':pid,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'start_ticks':Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]}


def is_alive(owner):
    try:return process_identity(owner['pid'])==owner
    except FileNotFoundError:return False


@contextmanager
def lock(path):
    with Path(path).open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def seal(trial,result):
    path=Path(trial)/'result.json'
    if path.exists():raise FileExistsError('Terminal receipts are immutable: '+str(path))
    write(path,{'result':result,'sha256':digest(result)})


def receipt(path):
    record=read(path)
    if digest(record['result'])!=record['sha256']:raise ValueError('Terminal receipt hash mismatch: '+str(path))
    return record['result']


def previous(directory,registration):
    """Use per-attempt receipts, not an aggregate that could lag a crash."""
    directory=Path(directory).resolve()
    if read(directory/'registration.json')['manifest_sha256']!=registration['manifest_sha256']:
        raise ValueError('Resume input/config/algorithm identity changed; register a new campaign')
    if is_alive(read(directory/'owner.json')):raise RuntimeError('Previous campaign process still owns its run')
    for record in (directory/'attempts').glob('*/*.process.json'):
        child=read(record)
        if is_alive(child['child']):raise RuntimeError('Previous owned child still running: '+str(record))
        if child['owner']['boot_id']==process_identity()['boot_id']:
            try:os.killpg(child['process_group'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError('Previous child group still exists; ownership must be resolved: '+str(record))
    results=read(directory/'inherited_attempts.json') if (directory/'inherited_attempts.json').exists() else []
    # Verify inherited terminals again; never rely on copied unsealed results.
    for result in results:
        if result['status']!='interrupted' and receipt(Path(result['source'])/'result.json')!=result:
            raise ValueError('Inherited receipt changed')
    for intent in sorted((directory/'attempts').glob('*/started.json')):
        trial=intent.parent;start=read(intent);terminal=trial/'result.json'
        if terminal.exists():result=receipt(terminal)
        else:
            result={**start,'source':str(trial),'status':'interrupted','demonstration_ready':False,'pass':False,
                    'error':'Started without a valid terminal receipt; never automatically rerun',
                    'elapsed_seconds':0,'motion_sent':motion_sent(trial)}
        if result['attempt_id']!=start['attempt_id'] or result['case_id']!=start['case_id']:raise ValueError('Attempt receipt identity differs')
        results.append(result)
    if len({r['attempt_id'] for r in results})!=len(results) or len({r['case_id'] for r in results})!=len(results):
        raise ValueError('Duplicate attempts/cases in continuation chain')
    return results


def journal(trial):
    path=Path(trial)/'requests.jsonl'
    if not path.exists():return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def motion_sent(trial):
    # An intent alone is ambiguous after a crash, not an executed motion count.
    try:events=journal(trial)
    except (ValueError,OSError):return None
    if any(e['event'] in ('sent','response') and e['action']=='trajectory' for e in events):return True
    if any(e['event']=='intent' and e['action']=='trajectory' for e in events):return None
    return False


class Ledger:
    def __init__(self,output,registration,inherited=()):
        self.output=Path(output).resolve();self.registration=registration
        self.output.mkdir(parents=True,exist_ok=False);(self.output/'attempts').mkdir()
        self.results=list(inherited);self.unexecuted=[]
        write(self.output/'registration.json',registration)
        write(self.output/'owner.json',process_identity())
        write(self.output/'inherited_attempts.json',self.results)
        self.progress('registered')

    def begin(self,case):
        if len(self.results)>=self.registration['budget']['max_attempts']:raise RuntimeError('Cumulative attempt budget exhausted')
        ident=uuid.uuid4().hex;trial=self.output/'attempts'/(case['case_id']+'__'+ident)
        trial.mkdir();(trial/'environment_evidence').mkdir()
        start={**case,'attempt_id':ident,'registered_unix':time.time()}
        write(trial/'started.json',start)
        self.event('started',**start)
        return trial,start

    def event(self,event,**values):append(self.output/'events.jsonl',{'event':event,'wall_time':time.time(),**values})

    def progress(self,status,**values):
        data={'status':status,'completed':len(self.results),'physical_successes':sum(r.get('pass',False) for r in self.results),
              'valid_demonstrations':sum(r.get('demonstration_ready',False) for r in self.results),**values}
        write(self.output/'state.json',data)
        print('ARX_BATCH',json.dumps(data,ensure_ascii=False),flush=True)

    def finish(self,trial,result):
        if result.get('budget_expires_monotonic') is not None:
            result['elapsed_seconds']=time.monotonic()-(result['budget_expires_monotonic']-900)
            if time.monotonic()>=result['budget_expires_monotonic']:
                result.update(status='timeout',demonstration_ready=False)
        seal(trial,result);self.results.append(result);self.aggregate()
        self.event('terminal',case_id=result['case_id'],attempt_id=result['attempt_id'],status=result['status'])

    def aggregate(self):
        write(self.output/'all_attempts.json',self.results)
        write(self.output/'dataset_sources.json',{
            **{role:[r['source'] for r in self.results if r.get('demonstration_ready') and r['split']==role]
               for role in ('train','validation')},
            'development':[r['source'] for r in self.results if r['split']=='development'],
            'excluded_failures':[{'source':r['source'],'status':r['status'],'reason':r.get('error') or r.get('quality')} for r in self.results if not r.get('demonstration_ready')],
            'roles':self.registration['roles'],'all_attempts':str(self.output/'all_attempts.json'),
            'contract_sha256':{k:digest(v) for k,v in self.registration['contracts'].items()}})
        seconds=sum(r.get('elapsed_seconds',0) for r in self.results)
        write(self.output/'summary.json',{'planned_cases':len(self.registration['cases']),'started_attempts':len(self.results),
            'motion_attempts':sum(r.get('motion_sent') is True for r in self.results),
            'motion_delivery_unknown':sum(r.get('motion_sent') is None for r in self.results),
            'physical_successes':sum(r.get('pass',False) for r in self.results),
            'valid_demonstrations':sum(r.get('demonstration_ready',False) for r in self.results),
            'valid_demonstrations_per_attempt_hour':sum(r.get('demonstration_ready',False) for r in self.results)*3600/seconds if seconds else None,
            'unexecuted':self.unexecuted,'budget_consumed':len(self.results),'maximum_attempts':self.registration['budget']['max_attempts']})
