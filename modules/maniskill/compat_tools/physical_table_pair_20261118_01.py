"""Paired 3 cm benchmark with measured geometry and the actual table surface."""
import argparse
import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
import tempfile
import textwrap
import time
import types

from arx_project.layout import module_dir
CODE = module_dir('maniskill')
SELF = Path(__file__).resolve()
sys.path.insert(0, str(CODE))
PROTOCOL = 'arx_tcp43_lift30_open_grasp_actual_table_no_nominal_v1'
REPORT = CODE/'docs/小幅抬升测试与IK核查.md'
INSTALLED = False


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PolicyTransform(ast.NodeTransformer):
    def __init__(self, mode):
        self.mode = mode
        self.changes = Counter()

    def visit_If(self, node):
        test = ast.unparse(node.test)
        if self.mode in ['constructor','bind'] and test == 'not closed':
            self.changes['remove_nominal_model'] += 1
            return None
        if self.mode in ['constructor','bind'] and test == "closed and key == 'table'":
            self.changes['remove_extra_table_shift'] += 1
            return None
        if self.mode == 'selector' and test == 'not self.closed_checker.validate(qg[None, None]).all()':
            self.changes['remove_nominal_rejection'] += 1
            return None
        return self.generic_visit(node)

    def visit_For(self, node):
        if self.mode == 'templates' and ast.unparse(node.target) == '(checker, margin)':
            assert isinstance(node.iter, ast.List) and len(node.iter.elts)==2
            assert 'closed_checker' in ast.unparse(node.iter.elts[1])
            node.iter.elts = node.iter.elts[:1]
            self.changes['open_template_only'] += 1
        return self.generic_visit(node)

    def visit_Dict(self, node):
        if self.mode == 'pool':
            for i,key in enumerate(node.keys):
                if isinstance(key,ast.Constant) and key.value=='table_closed_margin_m':
                    assert isinstance(node.values[i],ast.Constant) and node.values[i].value==.008
                    node.values[i] = ast.Constant(0.)
                    node.keys.append(ast.Constant('nominal_closed_precheck'))
                    node.values.append(ast.Constant(False))
                    node.keys.append(ast.Constant('collision_policy'))
                    node.values.append(ast.Constant(PROTOCOL))
                    self.changes['metadata_actual_table'] += 1
                    break
        return self.generic_visit(node)


EXPECTED = {
    'constructor':{'remove_nominal_model':1,'remove_extra_table_shift':1},
    'bind':{'remove_nominal_model':1,'remove_extra_table_shift':1},
    'selector':{'remove_nominal_rejection':1},
    'templates':{'open_template_only':1},
    'pool':{'metadata_actual_table':1},
}


def transform_tree(source, mode):
    tree = ast.parse(textwrap.dedent(source))
    change = PolicyTransform(mode)
    tree = ast.fix_missing_locations(change.visit(tree))
    assert dict(change.changes)==EXPECTED[mode], (mode,change.changes)
    return tree


def changed_function(function, mode, overrides=None):
    tree = transform_tree(inspect.getsource(function),mode)
    namespace = function.__globals__ if overrides is None else {**function.__globals__,**overrides}
    scope = dict(namespace)
    exec(compile(tree,str(SELF)+':'+mode,'exec'),scope)
    result = scope[function.__name__]
    return types.FunctionType(result.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)


def install_collision_policy():
    global INSTALLED
    if INSTALLED:
        return
    import numpy as np
    import resident_planner as resident
    import table_truncation_model as truncation
    import short_lift_planner as selector
    truncation.PROTOCOL = PROTOCOL
    resident.frozen.Planner.__init__ = changed_function(resident.frozen.Planner.__init__,'constructor')
    resident.bind_inputs = changed_function(resident.bind_inputs,'bind')
    selector.ShortLiftPlanner.select_tcp43 = changed_function(
        selector.ShortLiftPlanner.select_tcp43,'selector',{'PROTOCOL':PROTOCOL})
    truncation.templates = changed_function(truncation.templates,'templates')
    changed_pool = changed_function(truncation.make_pool,'pool')

    def make_pool(base):
        pool = changed_pool(base)
        original_init = pool.__init__

        def initialize(self, root, warmup_trial):
            original_init(self,root,warmup_trial)
            assert self.contexts['open'].closed_checker is None
            for role,filename in [('open','state.json'),('closed','closed_state.json')]:
                context = self.contexts[role]
                state = read(Path(warmup_trial)/filename)
                actual = read(context.out/'scene.json')['cuboid']['table']['pose'][:3]
                wanted = context.invbase@np.r_[state['scene_cuboids']['table']['center'],1.]
                assert np.allclose(actual,wanted[:3],atol=1e-9,rtol=0)
                assert not (context.out/'nominal_closed.yml').exists()
            save(Path(root)/'physical_table_policy_ready.json',{
                'passed':True,'protocol':PROTOCOL,'nominal_closed_model_constructed':False,
                'extra_closed_table_margin_m':0.,'open_grasp_collision_preserved':True,
                'approach_path_collision_preserved':True,'measured_closed_lift_collision_preserved':True,
                'model_load_counts':self.loads,'changes':EXPECTED})

        pool.__init__ = initialize
        return pool

    truncation.make_pool = make_pool
    INSTALLED = True


def self_test():
    import numpy as np
    from types import SimpleNamespace
    proofs = {}
    functions = [
        ('arx_pipeline/planner.py','Planner','__init__','constructor'),
        ('resident_planner.py',None,'bind_inputs','bind'),
        ('short_lift_planner.py','ShortLiftPlanner','select_tcp43','selector'),
        ('table_truncation_model.py',None,'templates','templates'),
        ('table_truncation_model.py',None,'make_pool','pool'),
    ]
    trees = {}
    for filename,owner,name,mode in functions:
        original = ast.parse((CODE/filename).read_text())
        body = next(n for n in original.body if isinstance(n,ast.ClassDef) and n.name==owner).body if owner else original.body
        node = next(n for n in body if isinstance(n,ast.FunctionDef) and n.name==name)
        tree = transform_tree(ast.unparse(node),mode)
        trees[mode] = tree
        proofs[mode] = EXPECTED[mode]
    assert 'closed_checker' not in ast.unparse(trees['selector'])
    assert 'closed_checker' not in ast.unparse(trees['templates'])
    assert '.008' not in ast.unparse(trees['constructor'])
    assert '.008' not in ast.unparse(trees['bind'])
    scope = {'np':np,'json':json,'LIFT_DISTANCE_M':.03,'PROTOCOL':PROTOCOL}
    exec(compile(trees['selector'],str(SELF)+':selector_contract','exec'),scope)
    with tempfile.TemporaryDirectory(prefix='arx_physical_table_contract_') as temporary:
        folder = Path(temporary)
        (folder/'graspcl').mkdir();(folder/'planner').mkdir()
        pose = np.eye(4);pose[2,3]=.243
        save(folder/'graspcl/candidate_pool.json',{'candidates':[{'pose':pose.tolist(),'score':1.,'metadata':{}}]})
        def forbidden_nominal(*args):
            raise AssertionError('Nominal closed checker must not be consulted')
        fixture = SimpleNamespace(out=folder/'planner',measured=lambda s:np.zeros(6),
            ik=lambda world,start:np.zeros(6),closed_checker=SimpleNamespace(validate=forbidden_nominal),
            path=lambda a,b:np.stack([a,b]),optimize=lambda name,q,seconds:{'q':q.tolist(),'times':[0,seconds]})
        selected = scope['select_tcp43'](fixture,{},[],None)
        assert selected['success'] and selected['rejected']['closed_collision']==0
        assert abs(selected['selected']['lift_world'][2][3]-selected['selected']['grasp_world'][2][3]-.03)<1e-12
        fixture.path=lambda a,b:None
        assert scope['select_tcp43'](fixture,{},[],None)['rejected']['path']==1
        fixture.ik=lambda a,b:None
        assert scope['select_tcp43'](fixture,{},[],None)['rejected']['grasp_ik']==1
        module = ast.parse((CODE/'table_truncation_model.py').read_text())
        nodes = [n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in ['box_clearance','closing_sweep_points','truncate']]
        local = {'np':np,'deepcopy':deepcopy,'PROTOCOL':PROTOCOL}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SELF)+':truncation_contract','exec'),local)
        spheres=np.array([[0,0,0,.002]]*3);names=np.array(['right_link26','right_link27','right_link28'])
        opened=(spheres,names,0.);nominal=(spheres,names,.008)
        contact=np.diag([1.,-1.,-1.,1.]);contact[2,3]=.006-.043
        candidate={'pose':contact,'width':.04,'metadata':{'source_network_depth_m':.01}}
        table={'center':[0,0,-.25],'dims':[1.,1.,.5]}
        kept,_=local['truncate'](candidate,np.array([[0,0,.006]]),table,[opened],.043,.088)
        assert kept is not None and kept['metadata']['insertion_retraction_m']==0
        old,_=local['truncate'](candidate,np.array([[0,0,.006]]),table,[opened,nominal],.043,.088)
        assert old is None or old['metadata']['insertion_retraction_m']>0
        colliding=deepcopy(candidate);colliding['pose'][2,3]=-.001-.043
        recovered,_=local['truncate'](colliding,np.array([[0,0,.003]]),table,[opened],.043,.088)
        assert recovered is not None and recovered['metadata']['insertion_retraction_m']>0
        from short_lift_validation import verifier_contract
        verified=verifier_contract(folder)
    return {'passed':True,'physical_attempts':0,'ast_changes':proofs,
        'nominal_rejection_removed':True,'real_table_collision_still_retracts':True,
        'ik_and_path_rejections_preserved':True,'lift_target_m':.03,**verified}


def configure_benchmark():
    import numpy as np
    from unittest.mock import patch
    import short_lift_bench as base
    base.__file__ = str(SELF)
    base.report = lambda out:None
    core = base.core

    class Backends(core.Backends):
        def start(self):
            original = core.subprocess.Popen
            def launch(command,*args,**kwargs):
                revised=[]
                for value in command:
                    if str(value)==str(CODE/'native_resident_services.py'):
                        revised.extend([str(SELF),'serve'])
                    else:revised.append(value)
                return original(revised,*args,**kwargs)
            with patch.object(core.subprocess,'Popen',launch):
                super().start()
            assert read(self.campaign/'resident/planner_warmup/physical_table_policy_ready.json')['passed']

    base.Backends = Backends
    original_verify = base.verify
    def verify(out):
        original_verify(out)
        reg=read(out/'registration.json');checked=0
        for arm,_,_ in reg['arms']:
            for claim in (out/arm/'claims').glob('*.json'):
                trial=out/arm/'trials'/claim.stem
                if (trial/'truncation.json').exists():
                    refinement=read(trial/'truncation.json')
                    assert refinement['collision_policy']==PROTOCOL
                    assert refinement['table_closed_margin_m']==0
                    assert refinement['nominal_closed_precheck'] is False
                    checked+=1
                for model in ['refinement_model','plan','approach_plan','lift_plan']:
                    assert not (trial/model/'nominal_closed.yml').exists()
                if (trial/'lift_plan/scene.json').exists():
                    state=read(trial/'closed_state.json')
                    table=read(trial/'lift_plan/scene.json')['cuboid']['table']['pose'][:3]
                    expected=np.linalg.inv(np.asarray(state['links']['body4']))@np.r_[state['scene_cuboids']['table']['center'],1.]
                    assert np.allclose(table,expected[:3],atol=1e-9,rtol=0)
        receipt=read(out/'final_verification.json')
        receipt.update(collision_policy=PROTOCOL,policy_refinement_checks=checked,
            nominal_closed_precheck=False,extra_closed_table_margin_m=0.,measured_closed_lift_collision_preserved=True)
        save(out/'final_verification.json',receipt)
    base.verify=verify
    return base


def prepare(out):
    validation=self_test()
    base=configure_benchmark()
    inputs=CODE/'benchmark_inputs/arx_independent_20261117_01'
    base.prepare(out,inputs,['fixed_orientation'])
    save(out/'component_validation.json',validation)
    baseline=CODE/'docs/resource/arx_lift30_summary_20261118_01/before_after_summary.json'
    shutil.copy2(baseline,out/'report_baseline.json')
    shutil.copy2(SELF,out/'source/physical_table_pair.py')
    for folder in [out,*[out/name for name,_,_ in read(out/'registration.json')['arms']]]:
        path=folder/'registration.json';reg=read(path)
        reg['source_hashes'].update({str(SELF):sha(SELF),str(out/'component_validation.json'):sha(out/'component_validation.json'),
                                     str(out/'report_baseline.json'):sha(out/'report_baseline.json')})
        reg.update(candidate_protocol=PROTOCOL,collision_policy=PROTOCOL,
            nominal_closed_precheck=False,extra_closed_table_margin_m=0.,
            truncation_collision_model='Measured open gripper only, against actual unchanged table surface',
            measured_closed_lift_collision_check=True,
            frozen_algorithm='User-authorized removal of fixed 22mm nominal-closed precheck and +8mm table displacement. Preserve measured open grasp/approach and measured closed 3cm lift IK/path/collision/optimizer, all geometry, physics, initial conditions, cameras, scoring and 1cm/3sec physical success gates.',
            collision_contract_validation=str(out/'component_validation.json'),
            report_path=str(REPORT),new_markdown_reports_created=False,
            user_authorized_rerun='2026-11-18: rerun ZeroGrasp and GraspCL, 200 each, in background; update the existing report only.')
        save(path,reg)
    content=REPORT.read_text()
    lines=content.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('- 截断：'):
            lines[i]='- 截断：按执行时的开爪模型检查抓取位姿与实际桌面；发生桌面碰撞或闭合范围内没有目标点云时，沿接近轴按 2 mm 步长后退，最多退到原插入深度。原位合格则不动，无合格位置则丢弃。取消固定 22 mm 名义闭爪预筛选及额外 8 mm 桌面余量；接近路径和实测闭爪后的 3 cm 抬升路径仍检查碰撞。'
            break
    else:raise AssertionError('Existing report truncation entry missing')
    REPORT.write_text('\n'.join(lines)+'\n')
    update_report(out,{})


FAILURES=['初始化／观察失败','无可用候选','抓取 IK／场景碰撞未通过','名义闭爪碰撞',
          '预抓取或抬升 IK／碰撞未通过','路径搜索失败','测量状态碰撞','动作未收敛',
          '未形成双指接触','未达到当组抬升／保持标准','出现禁止接触','其他失败']


def failure_reason(row,trial):
    reason=row['reason']
    if reason=='no_feasible_native_post_or_recalled_candidate':
        p=read(trial/'plan_result.json');n=p['count'];r=p['rejected']
        if n==0:return '无可用候选'
        if r['grasp_ik']==n:return '抓取 IK／场景碰撞未通过'
        if r['grasp_ik']+r['closed_collision']==n:return '名义闭爪碰撞'
        if r['grasp_ik']+r['closed_collision']+r['pre_or_lift_ik']==n:return '预抓取或抬升 IK／碰撞未通过'
        if r['path']:return '路径搜索失败'
    if reason in ['independent_scene_admission_failed','registered_occlusion_outside_range','registered_wrist_view_not_admitted']:return '初始化／观察失败'
    if reason=='measured_start_state_in_collision':return '测量状态碰撞'
    if reason.endswith('_not_converged'):return '动作未收敛'
    return {'no_bilateral_target_contact':'未形成双指接触','insufficient_hold_height':'未达到当组抬升／保持标准',
            'forbidden_robot_contact':'出现禁止接触'}.get(reason,'其他失败')


def replace_section(content,prefix,section):
    start=content.index(prefix)
    end=content.find('\n## ',start+len(prefix))
    if end<0:end=len(content)
    return content[:start]+section.rstrip()+'\n'+content[end:]


def update_report(out,cache):
    reg=read(out/'registration.json');arms={};details=[]
    for name,profile,_ in reg['arms']:
        rows=[];counts=Counter()
        for file in sorted((out/name/'claims').glob('*.json')):
            try:claim=read(file)
            except json.JSONDecodeError:continue
            if claim['state']!='complete':continue
            key=name+'/'+file.stem;trial=out/name/'trials'/file.stem
            if key not in cache:
                assert sha(trial/'result.json')==claim['result_sha256']
                assert read(trial/'independent_evaluation.json')['passed']
                row=read(trial/'result.json');stages=row.get('stages',{})
                cache[key]={'id':row['id'],'planning_success':row['planning_success'],'success':row['success'],
                    'closed':bool(stages.get('approach',{}).get('settled') and stages.get('close',{}).get('settled')),
                    'failure_category':None if row['success'] else failure_reason(row,trial),'reason':row['reason'],
                    'result_sha256':claim['result_sha256']}
            row=cache[key];assert row['result_sha256']==claim['result_sha256'];rows.append(row)
            if not row['success']:counts[row['failure_category']]+=1
            details.append({'arm':name,**row})
        n=len(rows);success=sum(r['success'] for r in rows)
        assert success+sum(counts.values())==n
        arms[name]={'registered':200,'completed':n,'initial_plan_passes':sum(r['planning_success'] for r in rows),
            'approach_and_closure_completed':sum(r['closed'] for r in rows),'successes':success,
            'grasp_success_rate':success/n if n else None,'failure_distribution':dict(counts)}
    progress=read(out/'progress.json')
    clock=read(out/'suite_timing.json') if (out/'suite_timing.json').exists() else {}
    metrics={'phase':progress['phase'],'registered':400,'completed':sum(v['completed'] for v in arms.values()),
             'arms':arms,'updated_at':datetime.now().astimezone().isoformat(),'timing':clock,'collision_policy':PROTOCOL}
    save(out/'independent_metrics.json',metrics);save(out/'independent_case_metrics.json',details)
    heading='## 调整后（独立场景集，本轮重测）'
    lines=[heading,'','| 链路 | 完成／测试数 | 抓取成功 | 成功率（已完成） | 失败数 |','|---|---:|---:|---:|---:|']
    for name,label in [('fixed_zerograsp','ZeroGrasp＋cuRobo'),('fixed_graspcl','ZeroGrasp＋GraspCL＋cuRobo')]:
        a=arms[name];pct='—' if not a['completed'] else f"{a['grasp_success_rate']:.1%}"
        lines.append(f"| {label} | {a['completed']}/200 | {a['successes'] if a['completed'] else '—'} | {pct} | {sum(a['failure_distribution'].values()) if a['completed'] else '—'} |")
    before=[r for r in read(out/'report_baseline.json')['groups'] if r['phase']=='调整前']
    failures=['## 失败原因分布（已完成例数）','',
        '| 失败原因 | 调整前 ZeroGrasp | 调整前加 GraspCL | 本轮 ZeroGrasp | 本轮加 GraspCL |',
        '|---|---:|---:|---:|---:|']
    after=[arms['fixed_zerograsp'],arms['fixed_graspcl']]
    for key in FAILURES:
        values=[b['failure_distribution'].get(key,0) for b in before]+[a['failure_distribution'].get(key,0) if a['completed'] else '—' for a in after]
        if key=='其他失败' and not any(v for v in values if isinstance(v,int)):continue
        failures.append('| '+key+' | '+' | '.join(map(str,values))+' |')
    failures.append('| 合计 | '+' | '.join(map(str,[b['failures'] for b in before]+[sum(a['failure_distribution'].values()) if a['completed'] else '—' for a in after]))+' |')
    content=REPORT.read_text();content=replace_section(content,'## 调整后','\n'.join(lines))
    content=replace_section(content,'## 失败原因分布','\n'.join(failures))
    REPORT.write_text(content.rstrip()+'\n')
    return metrics


def watch(out):
    cache={}
    while True:
        with (out/'report.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:metrics=update_report(out,cache)
            except json.JSONDecodeError:
                time.sleep(1);continue
        print(json.dumps({'phase':metrics['phase'],'completed':metrics['completed'],'arms':metrics['arms']},ensure_ascii=False),flush=True)
        if metrics['phase'] in ['complete','halted'] and metrics['timing'].get('all_owned_services_exited'):
            return
        time.sleep(30)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','execute','serve','watch'])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--entry',type=Path)
    args,remaining=parser.parse_known_args()
    if args.stage=='serve':
        if remaining[remaining.index('--kind')+1]=='curobo':install_collision_policy()
        from short_lift_services import main as serve
        sys.argv=[str(SELF),*remaining]
        serve();return
    assert not remaining and args.output is not None
    out=args.output.resolve()
    if args.stage=='prepare':prepare(out)
    elif args.stage=='watch':watch(out)
    else:
        base=configure_benchmark()
        if args.stage=='run':base.run(out)
        else:base.execute(out,args.entry.resolve())


if __name__=='__main__':main()
