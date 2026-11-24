"""Atomic local mailbox client for the ARX-only simulation diagnostic runner."""
import json,time,uuid,os
from pathlib import Path


class Client:
    def __init__(self,root,*,journal=None,deadline=None,session=None,explicit_root=False,heartbeat=None):
        self.root=Path(root if explicit_root else os.environ.get('ARX_CUROBO_VALIDATION_DIR',root))
        self.journal=journal or os.environ.get('ARX_REQUEST_JOURNAL')
        self.deadline=deadline if deadline is not None else float(os.environ.get('ARX_ATTEMPT_DEADLINE','inf'))
        self.session=session or os.environ.get('ARX_COLLECTION_SESSION')
        self.heartbeat=heartbeat

    def event(self,value):
        if self.journal:
            with Path(self.journal).open('a') as stream:
                stream.write(json.dumps({**value,'wall_time':time.time()},allow_nan=False)+'\n')
                stream.flush();os.fsync(stream.fileno())

    def request(self,action,timeout=180,**values):
        timeout=min(timeout,self.deadline-time.monotonic())
        if timeout<=0:raise TimeoutError('Shared attempt deadline exhausted before '+action)
        if self.session:
            status=json.loads((self.root/'stream_state.json').read_text())['collection_status']
            age=time.time()-status['wall_time']
            if status['session_id']!=self.session or not 0<=age<=5:
                raise RuntimeError('Collection instance changed or status is stale')
        guard=os.environ.get('ARX_RESEARCH_GUARD_FILE')
        audit_flag=os.environ.get('ARX_RESEARCH_AUDIT')
        if audit_flag not in (None,'0','1'):
            raise RuntimeError('ARX_RESEARCH_AUDIT must be 0 or 1')
        # An explicitly supplied legacy research guard retains its old behavior.
        # Normal runners set 0, keeping configuration writes forbidden while
        # avoiding full simulator-property scans around each motion command.
        audit=bool(guard) and audit_flag!='0'
        if guard:
            from research_guard import FORBIDDEN,check
            if action in FORBIDDEN:
                raise RuntimeError('Action forbidden by frozen research configuration: '+action)
        if not (self.root/'ready.json').exists():raise RuntimeError('ARX diagnostic runner is not ready')
        if (self.root/'request.json').exists():raise RuntimeError('Mailbox already has a pending request')
        request={'id':uuid.uuid4().hex,'action':action,**values}
        if audit and action=='trajectory':
            check(self,guard,request['id'],'before_trajectory')
        self.event({'event':'intent','request_id':request['id'],'action':action,'timeout_seconds':timeout})
        tmp=self.root/'request.tmp';tmp.write_text(json.dumps(request));tmp.replace(self.root/'request.json')
        self.event({'event':'sent','request_id':request['id'],'action':action})
        end=time.monotonic()+timeout
        next_heartbeat=time.monotonic()+30
        while time.monotonic()<end:
            if self.heartbeat and time.monotonic()>=next_heartbeat:
                self.heartbeat();next_heartbeat=time.monotonic()+30
            path=self.root/'response.json'
            if path.exists():
                result=json.loads(path.read_text())
                if result['id']==request['id']:
                    self.event({'event':'response','request_id':request['id'],'action':action,'ok':result['ok']})
                    if not result['ok']:raise RuntimeError(result['error'])
                    if audit and action=='trajectory':
                        # Retain execution evidence even if the post-motion guard fails.
                        raw=Path(guard).parent/'guard_checks'/(request['id']+'_execution.json')
                        raw.write_text(json.dumps(result['result']))
                        check(self,guard,request['id'],'after_trajectory')
                    return result['result']
            time.sleep(0.1)
        raise TimeoutError('Timed out waiting for ARX '+action)
