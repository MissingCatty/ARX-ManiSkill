"""Check the current and two scan-selected lifts on the same twelve CLARO inputs."""
import argparse
from pathlib import Path
import os
import sys
import time
import traceback
import numpy as np

from 公共 import ROOT,MS,JOINTS,read,save,sha,now
from 比较升降高度 import CLARO,shifted

SELF=Path(__file__).resolve()


def main(out):
    registration=read(out/'登记.json');scan=read(out/'高度扫描.json');assert scan['passed']
    os.environ['CUDA_VISIBLE_DEVICES']=registration['gpu_uuid']
    sys.path[:0]=[str(MS),str(MS/'arx_pipeline'),str(MS/'tmp')]
    import arx_runtime_imports
    arx_runtime_imports.install()
    import resident_planner
    import short_lift_planner
    from fast_resident_planner import FastPlannerPool
    from physical_table_pair_20261118_01 import install_collision_policy
    import arx_direct_grasp_20261120 as continuous
    import arx_direct_terminal_20261120 as terminal
    install_collision_policy();terminal.install()
    resident_planner.GraspCLPlanner=continuous.make_planner(short_lift_planner.ShortLiftPlanner)
    protected={**registration['source_sha256'],str(SELF):sha(SELF)}
    protocol={'created_at':now(),'direct_grasp':True,'source_hashes':protected,
        'lifts_m':scan['path_check_lifts_m'],'cases':[e['case'] for e in read(CLARO/'registration.json')['schedule']],
        'physical_attempts':0,'MPC_execution':False,'same_frozen_continuous_planning_policy':True,
        'poses':'Same fixed world candidates, actual measured start arm joints; only lift/body4 height changes.',
        'metric':'Complete continuous grasp path, terminal corridor and nominal 3cm lift IK; no execution or physical grasp claim.'}
    reg_path=out/'规划核验登记.json';assert not reg_path.exists();save(reg_path,protocol)
    warm=MS/'benchmark_inputs/arx_native_post_20261117_01/warmup/B0069_S3_sparse_014_lemon'
    started=time.perf_counter();pool=FastPlannerPool(out/'常驻规划组件',warm)
    save(out/'规划组件就绪.json',{'pid':os.getpid(),'loads':pool.loads,'startup_seconds':time.perf_counter()-started,
        'per_height_robot_reloads':0,'physical_attempts':0})
    results=[]
    for lift in protocol['lifts_m']:
        for entry in read(CLARO/'registration.json')['schedule']:
            case=entry['case'];source=CLARO/'trials'/case/'01_抓挡'
            original=read(source/'state.json');old_lift=original['qpos'][original['joint_names'].index('joint4')]
            state=shifted(original,lift,old_lift)
            directory=out/'规划核验'/str(round(lift*100))/case;directory.mkdir(parents=True,exist_ok=False)
            save(directory/'state.json',state);save(directory/'graspcl/candidate_pool.json',read(source/'graspcl/candidate_pool.json'))
            save(directory/'source_guard.json',{'registration':str(reg_path),'sha256':sha(reg_path)})
            began=time.perf_counter()
            try:
                context=pool.bind(directory/'plan',state,include_target=False);context._direct_pool=pool
                result=context.select_tcp43(state,np.empty((0,17)),np.asarray(state['camera_world_optical']))
                if result['success']:
                    assert set(result['paths'])=={'approach'} and 'q_pre' not in result['selected']
                    assert read(directory/'continuous_grasp_plan.json')['passed']
                    selected=result['selected'];qg=np.asarray(selected['q_grasp']);ql=np.asarray(selected['q_lift'])
                    xyz,rotation=continuous.world_poses(context,np.vstack([qg,ql]))
                    for i,key in enumerate(['grasp_world','lift_world']):
                        desired=np.asarray(selected[key]);assert np.linalg.norm(xyz[i]-desired[:3,3])<=.00205
                    actual_lift=np.asarray(selected['lift_world'])[:3,3]-np.asarray(selected['grasp_world'])[:3,3]
                    assert np.allclose(actual_lift,[0,0,.03],atol=1e-10)
            except AssertionError as error:
                if str(error)!='Measured state collides in unchanged ARX model':raise
                result={'success':False,'reason':'measured_start_state_in_collision','rejected':{}}
            result.update(case=case,lift_m=lift,wall_seconds=time.perf_counter()-began,physical_attempts=0)
            save(directory/'规划结果.json',result);results.append(result)
            save(out/'规划核验进度.json',{'completed':len(results),'total':len(protocol['cases'])*len(protocol['lifts_m']),
                'lift_m':lift,'case':case,'last_success':result['success'],'physical_attempts':0})
            print('PLAN',lift,case,result['success'],result.get('reason'),round(result['wall_seconds'],2),flush=True)
    for p,h in protected.items():assert sha(p)==h,p
    grouped=[{'lift_m':h,'planned':sum(r['success'] for r in results if r['lift_m']==h),
              'cases':12,'wall_seconds':sum(r['wall_seconds'] for r in results if r['lift_m']==h)} for h in protocol['lifts_m']]
    save(out/'规划核验汇总.json',{'passed':True,'groups':grouped,'rows':results,
         'model_load_counts':pool.loads,'physical_attempts':0,'source_files_unchanged':True,
         'wall_seconds':time.perf_counter()-started})
    print('PLANNING COMPLETE',grouped,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wait-for-scan-unit')
    args=parser.parse_args()
    try:
        if args.wait_for_scan_unit:
            import subprocess
            deadline=time.monotonic()+1800
            while True:
                status=subprocess.check_output(['systemctl','--user','show',args.wait_for_scan_unit,
                    '-p','ActiveState','-p','MainPID','-p','ExecMainStatus'],text=True)
                values=dict(line.split('=',1) for line in status.splitlines())
                if values['ActiveState']=='failed':raise RuntimeError('Height scan failed; planning did not start.')
                path=args.output/'高度扫描.json'
                if path.exists() and read(path).get('passed') and values['MainPID']=='0':break
                if time.monotonic()>deadline:raise TimeoutError('Waiting for completed height scan timed out')
                time.sleep(2)
        main(args.output.resolve())
    except BaseException:
        save(args.output/'规划核验故障.json',{'error':traceback.format_exc(),'physical_attempts':0});raise
