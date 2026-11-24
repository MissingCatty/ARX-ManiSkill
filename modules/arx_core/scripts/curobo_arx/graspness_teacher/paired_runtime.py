"""Bounded paired trials using the unchanged ARX factory execution/evaluation.

Each preregistered trial has a new ledger, one dispatch maximum, and fresh
perception. No episode, motion, camera frame, or previous success is replayed.
"""
import argparse
import copy
from pathlib import Path
import time
import traceback
from scripts.curobo_arx.data_factory.common import read,write,space
from scripts.curobo_arx.data_factory.ledger import Ledger
from scripts.curobo_arx.data_factory.runtime import Group


def choose_pool(pool, policy):
    if policy not in ('baseline','variant'):raise ValueError('Unknown registered comparison arm')
    pool=copy.deepcopy(pool)
    extension=pool.get('proposal_extension')
    if not extension:
        if pool.get('candidates'):raise RuntimeError('Missing original-prefix evidence')
        return pool
    else:
        ids=extension['original_prefix_ids'];actual=[g['goal_id'] for g in pool['candidates']]
        if actual[:len(ids)]!=ids:raise RuntimeError('Original candidate prefix changed')
    if policy=='baseline':
        pool['candidates']=pool['candidates'][:len(ids)]
    pool['paired_policy']=policy
    return pool


class NeuralArm:
    def __init__(self, client, policy):self.client,self.policy,self.directory=client,policy,None
    def submit(self,*args,**kwargs):
        self.directory=Path(kwargs['directory']);return self.client.submit(*args,**kwargs)
    def poll(self,job):
        response=self.client.poll(job)
        if response is None or not response['ok']:return response
        response=copy.deepcopy(response)
        response['result']=choose_pool(response['result'],self.policy)
        write(self.directory/'selected_pool.json',response['result'])
        return response


class PlannerArm:
    def __init__(self,client,policy):self.client,self.policy=client,policy
    def submit(self,*args,**kwargs):
        return self.client.submit(*args,**kwargs,comparison_arm=self.policy)
    def poll(self,job):return self.client.poll(job)


class PairedGroup(Group):
    def run(self):
        base_out=self.out;raw_client=self.neural;raw_planner=self.planner;results=[]
        self.ledger.close()
        try:
            for ident in self.launch['trial_ids']:
                if self.stop or (self.campaign/'STOP').exists() or time.time()>=self.launch['stop_unix']:break
                if not self.app.is_running():raise RuntimeError('ARX application stopped before registered trial')
                trial=next(t for t in self.registration['paired_trials'] if t['trial_id']==ident)
                space(self.campaign)
                self.out=base_out/'trials'/ident;self.out.mkdir(parents=True,exist_ok=False)
                self.ledger=Ledger(self.campaign/'trial_ledgers'/(ident+'.sqlite3'))
                compare_lift=self.registration.get('compare_lift_policy',False)
                self.neural=NeuralArm(raw_client,'variant' if compare_lift else trial['policy'])
                self.planner=PlannerArm(raw_planner,trial['policy']) if compare_lift else raw_planner
                asset=self.ledger.acquire_object(self.owner,object_id=trial['object_id'],renderer=self.renderer)
                if asset is None:raise RuntimeError('Registered trial has no exact frozen placement')
                began=time.time();result=self.run_block(asset,0)
                if any(env.task or env.bridge.active for env in self.envs):raise RuntimeError('Unconfirmed idle after paired trial')
                rows=self.ledger.rows()
                if len(rows)>1:raise RuntimeError('One registered trial dispatched multiple attempts')
                evidence=[read(Path(row['episode'])/'physical_receipt.json') for row in rows]
                quality=[read(Path(row['episode'])/'quality_receipt.json') for row in rows
                    if (Path(row['episode'])/'quality_receipt.json').exists()]
                receipt={**trial,'started_unix':began,'finished_unix':time.time(),
                    'completed':True,'success':bool(evidence and evidence[0]['pass']),
                    'admitted':bool(rows and rows[0]['state']=='committed'),
                    'physical_dispatches':sum(row['physical'] for row in rows),
                    'outcome':rows[0]['outcome'] if rows else result['status'],
                    'attempts':rows,'physical_evidence':evidence,'quality_evidence':quality,'block':result}
                write(self.out/'trial_result.json',receipt);results.append(receipt)
                write(base_out/'paired_results.json',{'status':'running','trials':results})
                self.ledger.release_object(self.owner);self.ledger.close()
                print('PAIRED_TRIAL',ident,receipt['outcome'],receipt['success'],flush=True)
            write(base_out/'paired_results.json',{'status':'stopped' if len(results)<len(self.launch['trial_ids']) else 'complete','trials':results})
        except BaseException:
            write(base_out/'FAULT.json',{'wall':time.time(),'error':traceback.format_exc(),
                'completed_trials':[r['trial_id'] for r in results]});raise
        finally:
            self.writer.close();self.ledger.close();self.diagnostics.close()
            idle=not any(env.bridge.active for env in self.envs)
            write(base_out/'shutdown_receipt.json',{'wall':time.time(),'all_trajectories_idle':idle,
                'raw_writer_drained_and_closed':True,'ledger_closed':True})
            self.app.close(wait_for_replicator=False,skip_cleanup=True,exit_code=1 if (base_out/'FAULT.json').exists() else 0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);parser.add_argument('--launch',required=True)
    args=parser.parse_args();PairedGroup(args.campaign,args.launch).run()
