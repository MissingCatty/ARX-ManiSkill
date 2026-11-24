"""Matched offline IK queries: original initialization versus map-based seeds."""
import argparse
from collections import Counter
import inspect
from pathlib import Path
import sys
import textwrap
import time
import types

import numpy as np
from scipy.spatial.transform import Rotation

from 公共 import ROOT,PROJECTS,MS,JOINTS,MODEL,URDF,TCP,read,save,sha,now
from 能力索引 import CapabilityMap


def queries():
    chapter=PROJECTS/'claro/docs/03_夹走并放下挡物/规划失败核查/resources'
    registration=read(chapter/'20261119_01/核查登记.json');short=read(chapter/'20261119_02/预抓取约束核查汇总.json')
    short_by_case={c['case']:c['planning_witness'] for c in short['cases'] if c['planning_witness'] and not c['control']}
    rows=[]
    for case in registration['cases']:
        directory=Path(case['directory']);state=read(directory/'state.json');pool=read(directory/'candidates/candidate_pool.json')['candidates']
        selected=range(len(pool))
        if case['control']:selected=[read(directory/'plan_result.json')['selected']['index']]
        for index in selected:
            grasp=np.asarray(pool[index]['pose']).copy();grasp[:3,3]-=.043*grasp[:3,2]
            for stage,distance in [('G',0.),('H8',.08)]:
                goal=grasp.copy();goal[:3,3]-=distance*grasp[:3,2]
                rows.append({'case':case['case'],'candidate':int(index),'stage':stage,'matrix':goal.tolist(),
                             'state_source':str(directory/'state.json')})
        if case['case'] in short_by_case:
            proof=short_by_case[case['case']]
            rows.append({'case':case['case'],'candidate':proof['candidate_index'],'stage':'H_short',
                         'matrix':proof['pregrasp_world'],'state_source':str(directory/'state.json')})
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--地图',type=Path,default=ROOT/'产物/首版_01')
    parser.add_argument('--显卡',required=True);args=parser.parse_args()
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=args.显卡
    out=args.地图/'查询验证_01';out.mkdir(exist_ok=False)
    inputs=queries();save(out/'查询清单.json',{'queries':inputs,'physical_attempts':0})
    source_files={Path(__file__).resolve(),Path(__file__).with_name('能力索引.py')}
    source_files.update([MODEL,URDF,TCP,MS/'arx_pipeline/planner.py',MS/'resident_planner.py',
                         MS/'fast_resident_planner.py',MS/'tmp/physical_table_pair_20261118_01.py'])
    source_files.update(Path(row['state_source']) for row in inputs)
    frozen={str(p):sha(p) for p in source_files}
    save(out/'验证登记.json',{'created_at':now(),'source_sha256':frozen,'query_count':len(inputs),
         'map_bank_contains_archived_query_solutions':False,'same_solver_and_optimizer_seed_budget':True,
         'algorithm_difference':'explicit map FK seeds versus original current-state initialization',
         'physical_attempts':0,'current_clutter_checked':True,'grasp_successes_measured':False})
    atlas=CapabilityMap(args.地图)
    sys.path[:0]=[str(MS),str(MS/'arx_pipeline'),str(MS/'tmp')]
    import torch
    import planner as original
    import resident_planner
    from fast_resident_planner import FastPlannerPool
    from short_lift_planner import ShortLiftPlanner
    from physical_table_pair_20261118_01 import install_collision_policy

    install_collision_policy();resident_planner.GraspCLPlanner=ShortLiftPlanner
    warmup=MS/'benchmark_inputs/arx_native_post_20261117_01/warmup/B0069_S3_sparse_014_lemon'
    pool=FastPlannerPool(out/'规划器预热',warmup)
    source=textwrap.dedent(inspect.getsource(original.Planner.ik))
    old='current_state=self.js(start), return_seeds=4)'
    assert source.count(old)==1
    source=source.replace(old,'current_state=self.js(start), seed_config=self.map_seed_tensor, return_seeds=4)')
    namespace=dict(original.__dict__);exec(compile(source,str(Path(__file__))+':map_seed_ik','exec'),namespace)
    assisted_function=namespace['ik'];results=[];counts=Counter();context=None;current_case=None;began=time.monotonic();grasp_starts={}
    for index,row in enumerate(inputs):
        state=read(row['state_source'])
        if row['case']!=current_case:
            context=pool.bind(out/row['case']/'原完整场景模型',state);current_case=row['case']
        start=context.measured(state);goal=np.asarray(row['matrix']);key=(row['case'],row['candidate'])
        initial_source='measured_observation'
        if row['stage']!='G' and grasp_starts.get(key) is not None:
            start=grasp_starts[key].clone();initial_source='baseline_grasp_solution'
        context.planner.reset_seed();t=time.monotonic();baseline=context.ik(goal,start)
        if row['stage']=='G':grasp_starts[key]=None if baseline is None else baseline.clone()
        baseline_seconds=time.monotonic()-t
        action=context.invbase@goal@atlas.tcp
        count=context.planner.ik_solver.config.num_seeds
        query=atlas.query_pose(action,current_q=start.cpu().numpy(),num_seeds=count)
        context.map_seed_tensor=torch.as_tensor(query['seeds'],dtype=torch.float32,device='cuda')[None]
        context.planner.reset_seed();t=time.monotonic()
        assisted=types.MethodType(assisted_function,context)(goal,start)
        assisted_seconds=time.monotonic()-t
        b=baseline is not None;a=assisted is not None
        counts.update(queries=1,baseline_pass=int(b),map_pass=int(a),gains=int(a and not b),regressions=int(b and not a))
        result={**row,'baseline_pass':b,'map_seed_pass':a,'optimizer_seed_count':count,'initial_state_source':initial_source,
                'baseline_seconds':baseline_seconds,'map_query_seconds':query['search_seconds'],'map_solver_seconds':assisted_seconds,
                'map_coverage_status':query['status'],'best_seed_pose':query['best_pose_sample'],
                'baseline_q':None if baseline is None else baseline.cpu().tolist(),
                'map_q':None if assisted is None else assisted.cpu().tolist()}
        results.append(result)
        if row['case']=='I0038' and row['candidate']==0:
            save(out/('I0038_'+row['stage']+'_查询示例.json'),{'input':row,'map_response':query,'result':result})
        if index%16==0 or index+1==len(inputs):
            save(out/'当前进度.json',{'processed':index+1,'total':len(inputs),**dict(counts),'physical_attempts':0})
            print('IK_QUERY',index+1,'/',len(inputs),dict(counts),flush=True)
    assert all(sha(p)==h for p,h in frozen.items())
    timings={k:float(np.median([r[k] for r in results])) for k in ['baseline_seconds','map_query_seconds','map_solver_seconds']}
    save(out/'查询对照结果.json',{'counts':dict(counts),'cases':results,'median_timings':timings,
         'map_load_seconds':atlas.load_seconds,'wall_seconds':time.monotonic()-began,
         'physical_attempts':0,'source_hashes_unchanged':True,
         'scope':'Saved failure-selected G/H endpoint queries; not grasp success or an unbiased workspace success-rate estimate.'})


if __name__=='__main__':main()
