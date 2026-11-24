"""Resident, bounded offline ARX planning experiments; never connects to simulation."""
from 公共 import EXPERIMENTS
import argparse
from collections import Counter
import inspect
import os
from pathlib import Path
import sys
import textwrap
import time
import traceback
import types

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from 公共 import ROOT,PROJECTS,MS,read,save,sha,now
from 能力索引 import CapabilityMap

CAMPAIGN=EXPERIMENTS / '自动研究_20261120_01'


class Budget(Exception):pass


class Worker:
    def __init__(self,gpu):
        os.environ['CUDA_VISIBLE_DEVICES']=gpu
        sys.path[:0]=[str(MS),str(MS/'arx_pipeline'),str(MS/'tmp')]
        import torch
        import planner as frozen
        import resident_planner
        from fast_resident_planner import FastPlannerPool
        from short_lift_planner import ShortLiftPlanner
        from physical_table_pair_20261118_01 import install_collision_policy
        from scene_tools import ArmFK
        self.torch=torch;self.frozen=frozen;self.fk=ArmFK();self.registration=read(CAMPAIGN/'登记.json')
        self.guard();started=time.monotonic();install_collision_policy();resident_planner.GraspCLPlanner=ShortLiftPlanner
        self.atlas=CapabilityMap(ROOT/'产物/首版_01')
        warmup=MS/'benchmark_inputs/arx_native_post_20261117_01/warmup/B0069_S3_sparse_014_lemon'
        self.pool=FastPlannerPool(CAMPAIGN/'常驻组件',warmup)
        source=textwrap.dedent(inspect.getsource(frozen.Planner.ik));old='current_state=self.js(start), return_seeds=4)'
        assert source.count(old)==1
        namespace=dict(frozen.__dict__)
        exec(compile(source.replace(old,'current_state=self.js(start), seed_config=self.map_seed_tensor, return_seeds=4)'),__file__+':seed_ik','exec'),namespace)
        self.seed_ik=namespace['ik'];self.cases=read(CAMPAIGN/'场景清单.json')['cases']
        save(CAMPAIGN/'组件就绪.json',{'ready_at':now(),'pid':os.getpid(),'gpu_uuid':gpu,'loads':self.pool.loads,
            'warmup':self.pool.warmup,'map_load_seconds':self.atlas.load_seconds,
            'startup_seconds':time.monotonic()-started,'required_components':self.registration['required_components'],
            'all_components_loaded_once':all(v==1 for v in self.pool.loads.values()),
            'physical_attempts':0,'simulator_connection':False})
        self.guard();print('RESEARCH_READY',gpu,flush=True)

    def guard(self):
        bad=[p for p,h in self.registration['source_sha256'].items() if sha(p)!=h]
        if bad:raise RuntimeError('Protected sources changed: '+str(bad))

    def bind(self,label,full=False):
        self.context=self.pool.bind(self.out/label,self.state,include_target=full);return self.context

    def limit(self):
        if time.monotonic()>self.deadline or self.counts['ik_calls']>=96:raise Budget()

    def quality(self,goal):
        pose=self.context.invbase@goal@self.atlas.tcp
        _,idx=self.atlas.tree.query(pose[:3,3],k=8192,workers=1)
        rows=np.asarray(self.atlas.bank[idx]);q=Rotation.from_matrix(pose[:3,:3]).as_quat()[[3,0,1,2]]
        angle=2*np.arccos(np.clip(np.abs(rows[:,9:13]@q),0,1))
        distance=np.linalg.norm(rows[:,6:9]-pose[:3,3],axis=1)
        score=distance/.02+angle/.35-.02*rows[:,14]
        return float(score.min())

    def inverse(self,goal,start,settings):
        self.limit();self.counts['ik_calls']+=1
        self.context.planner.reset_seed()
        if settings.get('seed'):
            begin=time.monotonic();request=self.atlas.query_pose(self.context.invbase@goal@self.atlas.tcp,
                current_q=start.cpu().numpy(),num_seeds=self.context.planner.ik_solver.config.num_seeds)
            self.counts['map_query_seconds']+=time.monotonic()-begin
            self.context.map_seed_tensor=self.torch.as_tensor(request['seeds'],device='cuda',dtype=self.torch.float32)[None]
            result=types.MethodType(self.seed_ik,self.context)(goal,start)
        else:result=self.context.ik(goal,start)
        if result is None and settings.get('refine') and self.counts['refinements']<16:
            target=self.context.invbase@goal
            if settings.get('random_seeds'):
                rng=np.random.default_rng(self.seed+self.counts['ik_calls']+201120)
                seeds=self.fk.lower+1e-4+rng.random((settings.get('refine_seeds',4),6))*(self.fk.upper-self.fk.lower-2e-4)
            else:
                request=self.atlas.query_pose(target@self.atlas.tcp,current_q=None,num_seeds=settings.get('refine_seeds',4))
                seeds=request['seeds']
            for seed in seeds:
                if self.counts['refinements']>=16:break
                self.limit();self.counts['refinements']+=1
                def residual(q):
                    if time.monotonic()>self.deadline:raise Budget()
                    t=self.fk.fk(q)
                    return np.r_[(t[:3,3]-target[:3,3])/.02,
                                 Rotation.from_matrix(target[:3,:3].T@t[:3,:3]).as_rotvec()/.35]
                answer=least_squares(residual,np.clip(seed,self.fk.lower+1e-4,self.fk.upper-1e-4),
                    bounds=(self.fk.lower+1e-4,self.fk.upper-1e-4),max_nfev=settings.get('refine_nfev',80),
                    ftol=1e-7,xtol=1e-7,gtol=1e-7)
                error=residual(answer.x)
                if np.linalg.norm(error[:3]*.02)>.002 or np.linalg.norm(error[3:]*.35)>.03:continue
                q=self.torch.as_tensor(answer.x,device='cuda',dtype=self.torch.float32)
                if not bool(self.context.checker.validate(q[None,None]).all()):continue
                self.counts['refined_solutions']+=1;result=q;break
        return result

    def dense_valid(self,route):
        q=np.asarray(route['q']);fractions=np.arange(5)/5
        dense=np.concatenate([(q[:-1,None]*(1-fractions[None,:,None])+q[1:,None]*fractions[None,:,None]).reshape(-1,6),q[-1:]])
        t=self.torch.as_tensor(dense,device='cuda',dtype=self.torch.float32)
        return all(bool(self.context.checker.validate(chunk[:,None]).all()) for chunk in t.split(64))

    def check_paths(self,index,grasp,pre,lift,qg,qp,ql):
        self.limit()
        if self.counts['path_pairs']>=12:raise Budget()
        pair=self.counts['path_pairs'];self.counts['path_pairs']+=1
        ctx=self.bind(f'候选{index}_路径{pair}_含目标进场',True);start=ctx.measured(self.state)
        if not bool(ctx.checker.validate(qp[None,None]).all()):return None,'full_scene_pregrasp_collision'
        ctx.planner.reset_seed();entry=ctx.path(start,qp)
        if entry is None:return None,'full_scene_entry_path'
        try:
            route=ctx.optimize('pregrasp',entry,6.)
            if not self.dense_valid(route):return None,'full_scene_optimized_entry_collision'
            ctx=self.bind(f'候选{index}_路径{pair}_原接近世界')
            ctx.planner.reset_seed();approach=ctx.path(qp,qg)
            if approach is None:return None,'approach_path'
            approach_route=ctx.optimize('approach',approach,5.)
            if not self.dense_valid(approach_route):return None,'optimized_approach_collision'
        except RuntimeError as error:
            if str(error)=='Trajectory optimizer rejected all attempts; no PRM-only fallback':return None,'optimizer_rejected'
            raise
        return {'candidate_index':index,'grasp_world':grasp.tolist(),'pregrasp_world':pre.tolist(),
                'nominal_lift_world':lift.tolist(),'q_grasp':qg.cpu().tolist(),'q_pre':qp.cpu().tolist(),
                'q_nominal_lift':ql.cpu().tolist(),'paths':{'pregrasp':route,'approach':approach_route},
                'original_full_scene_entry_checked':True,'original_dense_checks_passed':True},'passed'

    def solve_case(self,case,settings,seed,destination):
        self.seed=seed;self.out=destination;self.out.mkdir(parents=True,exist_ok=False)
        self.state=read(Path(case['directory'])/'state.json');native=read(Path(case['directory'])/'candidates/candidate_pool.json')['candidates']
        began=time.monotonic();self.deadline=began+12;self.counts=Counter();attempts=[];selected=None;reason='no_admitted_plan'
        context=self.bind('原场景');context.planner.reset_seed()
        # Seed stream is controlled identically between comparisons.
        if seed:
            import random
            random.seed(seed);np.random.seed(seed);self.torch.manual_seed(seed)
        start=context.measured(self.state)
        goals=[]
        for c in native:
            g=np.asarray(c['pose']).copy();g[:3,3]-=.043*g[:3,2];goals.append(g)
        order=list(range(len(goals)))
        try:
            if settings.get('candidate_rank'):
                order=sorted(order,key=lambda i:self.quality(goals[i]))
            for index in order:
                self.limit();g=goals[index];lift=g.copy();lift[2,3]+=.03
                self.bind(f'候选{index}_原场景')
                qg=self.inverse(g,start,settings)
                if qg is None:attempts.append({'index':index,'stage':'grasp_ik'});continue
                ql=self.inverse(lift,qg,settings)
                if ql is None:attempts.append({'index':index,'stage':'nominal_lift_ik'});continue
                distances=[.08,.06,.04,.02] if settings.get('adaptive') else [.08]
                if settings.get('distance_rank'):
                    def rank(d):
                        h=g.copy();h[:3,3]-=d*g[:3,2];return self.quality(h)
                    distances=sorted(distances,key=rank)
                for distance in distances:
                    self.bind(f'候选{index}_H{int(distance*100)}cm_原场景');self.limit()
                    h=g.copy();h[:3,3]-=distance*g[:3,2]
                    qp=self.inverse(h,qg,settings)
                    if qp is None:attempts.append({'index':index,'distance_m':distance,'stage':'pregrasp_ik'});continue
                    selected,stage=self.check_paths(index,g,h,lift,qg,qp,ql)
                    attempts.append({'index':index,'distance_m':distance,'stage':stage})
                    if selected is not None:selected['pregrasp_distance_m']=distance;break
                if selected is not None:reason='passed';break
        except Budget:reason='declared_compute_budget_exhausted'
        wall=time.monotonic()-began
        if wall>12 and selected is not None:selected=None;reason='declared_compute_budget_exhausted_after_atomic_solver_call'
        result={'case':case['case'],'stratum':case['stratum'],'group':case['group'],'success':selected is not None,
            'reason':reason,'selected':selected,'attempts':attempts,'counts':dict(self.counts),'wall_seconds':wall,
            'native_candidate_count':len(native),'input_state_sha256':case['source_state_sha256'],
            'input_candidates_sha256':case['source_candidates_sha256'],'G_modified':False,
            'hard_map_rejection':False,'physical_attempts':0,'grasp_success_measured':False}
        save(self.out/'规划结果.json',result);return result

    def run(self,request):
        self.guard();name=request['name'];assert '/' not in name and not name.startswith('.')
        destination=CAMPAIGN/'轮次'/name;destination.mkdir(parents=True,exist_ok=False)
        save(destination/'轮次登记.json',{**request,'created_at':now(),'executor_sha256':sha(__file__),
             'protected_registration_sha256':sha(CAMPAIGN/'登记.json'),'physical_attempts':0})
        selected=[c for c in self.cases if c['group']==request.get('group','development')]
        rows=[];started=time.monotonic()
        for case in selected:
            result=self.solve_case(case,request['settings'],request.get('seed',0),destination/case['case']);rows.append(result)
            save(destination/'当前进度.json',{'completed':len(rows),'total':len(selected),'passes':sum(r['success'] for r in rows),'physical_attempts':0})
            print('RESEARCH_CASE',name,case['case'],result['success'],result['reason'],round(result['wall_seconds'],2),result['counts'],flush=True)
            self.guard()
        summary={'name':name,'request':request,'rows':rows,'passes':sum(r['success'] for r in rows),'total':len(rows),
                 'wall_seconds':time.monotonic()-started,'protected_sources_unchanged':True,'physical_attempts':0}
        save(destination/'汇总.json',summary);print('RESEARCH_ROUND_DONE',name,summary['passes'],'/',len(rows),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--显卡',required=True);args=parser.parse_args()
    started=time.monotonic()
    try:
        worker=Worker(args.显卡)
        commands=CAMPAIGN/'请求';commands.mkdir(exist_ok=True)
        while time.monotonic()-started<1800 and not (CAMPAIGN/'停止.json').exists():
            for p in sorted(commands.glob('*.json')):
                receipt=p.with_suffix('.完成')
                if receipt.exists():continue
                request=read(p);state=read(CAMPAIGN/'状态.json');state.update(status='running',current=request['name']);save(CAMPAIGN/'状态.json',state)
                worker.run(request);receipt.write_text(now()+'\n')
                state.update(status='waiting_for_next_iteration',current=None);save(CAMPAIGN/'状态.json',state)
            time.sleep(.2)
        save(CAMPAIGN/'组件退出.json',{'at':now(),'pid':os.getpid(),'normal':True,'physical_attempts':0})
    except BaseException as error:
        save(CAMPAIGN/'故障停止.json',{'time':now(),'type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),'physical_attempts':0})
        raise


if __name__=='__main__':main()
