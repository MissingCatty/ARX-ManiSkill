"""Continue the same 2x200 allocation after fixing a constraint/error boundary."""
from collections import Counter
from copy import deepcopy
from datetime import datetime
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from unittest.mock import patch

import native_pair_bench as core
from native_pair_timing import timing_summary,execute
from graspcl_execute import guard
import graspcl_pipeline as common
from resident_backend import rpc
from benchmark_lifecycle import require_ready

CODE=Path(__file__).resolve().parent
SLOT='B0232_S4_clutter_057_racquetball'


def prepare(out,previous,diagnosis):
    evidence=common.read(diagnosis / 'verification.json')
    if not evidence['passed'] or not evidence['fresh_original_model_reproduces_collision']:
        raise RuntimeError('Original-model collision reproduction is incomplete')
    protocol=common.read(CODE / 'outputs/arx_constraint_protocol_validation_20261117_01/verification.json')
    if not protocol['passed']:raise RuntimeError('Constraint protocol validation is incomplete')
    for path,digest in protocol['validated_source_sha256'].items():
        if common.sha(path)!=digest:raise RuntimeError('Validated protocol changed: '+path)
    reg=common.read(previous / 'registration.json');guard(reg)
    if common.read(previous / 'progress.json')['phase']!='halted' or (previous / 'STOP').exists():
        raise RuntimeError('Only an unexpected halted suite without user STOP can continue')
    old_clock=common.read(previous / 'suite_timing.json')
    if not old_clock['all_owned_services_exited']:raise RuntimeError('Old services are not confirmed stopped')
    for kind in reg['required_components']:
        pid=common.read(previous / 'resident' / (kind+'_stopped.json'))['pid']
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise RuntimeError('Old owned service is alive')
    out.mkdir(parents=True,exist_ok=False)
    for name in ['registration.json']:
        shutil.copy2(previous / name,out / ('previous_'+name))
    carried={};carried_seconds={}
    run,baseline=core.batch.original_modules()
    diagnostic_reg=common.read(diagnosis / 'registration.json')
    for profile in core.PROFILES:
        arm=out / profile;old_arm=previous / profile;arm.mkdir()
        for folder in ['trials','claims','entries','workers','backend_requests','reference','resource_wait']:(arm / folder).mkdir()
        shutil.copy2(old_arm / 'reference/common_initial_conditions.json',arm / 'reference/common_initial_conditions.json')
        shutil.copy2(old_arm / 'storage_policy.json',arm / 'storage_policy.json')
        (arm / 'resident').symlink_to(out / 'resident',target_is_directory=True)
        count=0
        for path in (old_arm / 'claims').glob('*.json'):
            claim=common.read(path);source=old_arm / 'trials' / path.stem;target=arm / 'trials' / path.stem
            if claim['state']=='complete':
                assert common.read(source / 'independent_evaluation.json')['passed']
                assert common.sha(source / 'result.json')==claim['result_sha256']
                target.symlink_to(source,target_is_directory=True)
            else:
                if profile!='zerograsp_200' or path.stem!=SLOT:raise RuntimeError('Unknown slot cannot be replayed or inferred: '+path.stem)
                assert common.sha(source / 'state.json')==diagnostic_reg['state_sha256']
                assert common.sha(source / 'result.json')==diagnostic_reg['original_result_sha256']
                result=common.read(source / 'result.json')
                assert not result['success'] and not result['physical_attempt'] and not result['planning_success']
                assert not any((source / n).exists() for n in ['execution.jsonl','execution.mp4','execution_origin.json'])
                assert common.read(source / 'state.json')['qpos']==common.read(source / 'final_state.json')['qpos']
                shutil.copytree(source,target)
                baseline.verify_trial(target)
                claim.update(state='complete',finished_at=datetime.fromtimestamp((source / 'result.json').stat().st_mtime).astimezone().isoformat(),
                    result_sha256=common.sha(target / 'result.json'),settled_without_replay=True,
                    settlement='Original fresh ARX model reproduced the same initial collision; no commanded motion. Preserve original failed outcome and raw exception.',
                    settlement_verified_at=core.now(),diagnosis=str(diagnosis))
                common.save(target / 'constraint_rejection_classification.json',{'classification':'measured_start_state_in_collision',
                    'original_result_unchanged':True,'original_result_sha256':common.sha(target / 'result.json'),
                    'physical_attempts':0,'replayed':False,'diagnosis':str(diagnosis)})
            common.save(arm / 'claims' / path.name,claim);count+=1
        if any(t.name not in {p.stem for p in (old_arm / 'claims').glob('*.json')} for t in (old_arm / 'trials').iterdir()):
            raise RuntimeError('Old trial without a claim')
        for folder in ['backend_requests','resource_wait']:
            for source in (old_arm / folder).iterdir():shutil.copy2(source,arm / folder / source.name)
        carried[profile]=count
        old_round=old_arm / 'round_timing.json'
        carried_seconds[profile]=common.read(old_round).get('round_wall_seconds',0.) if old_round.exists() else 0.
    new=deepcopy(reg)
    new.update(previous_output=str(previous),previous_registration_sha256=common.sha(previous / 'registration.json'),
        continuation=True,carried_completed=carried,carried_round_seconds=carried_seconds,
        prior_suite_timing=old_clock,collision_diagnosis=str(diagnosis),
        service_protocol_change='Report the unchanged measured-state collision gate as a per-case planning rejection; other assertions and infrastructure errors still halt the suite.',
        allocated_slots_unchanged=400,started_slots_replayed=0)
    for name in ['resume_native_pair.py','checked_native_services.py']:
        file=CODE / name;new['source_hashes'][str(file)]=common.sha(file)
    (out / 'source').mkdir()
    for path,digest in new['source_hashes'].items():
        file=Path(path)
        if file.parent==CODE and file.suffix=='.py':shutil.copy2(file,out / 'source' / file.name)
    common.save(out / 'registration.json',new)
    for profile in core.PROFILES:
        arm=out / profile;armreg=common.read(previous / profile / 'registration.json')
        armreg.update(suite=str(out),previous_output=str(previous / profile),carried_completed=carried[profile],
            continuation=True,collision_diagnosis=str(diagnosis))
        armreg['source_hashes'].update(new['source_hashes'])
        armreg['source_hashes'][str(arm / 'reference/common_initial_conditions.json')]=common.sha(arm / 'reference/common_initial_conditions.json')
        common.save(arm / 'registration.json',armreg)
        rows=[common.read(arm / 'trials' / p.stem / 'result.json') for p in (arm / 'claims').glob('*.json')]
        common.save(arm / 'progress.json',{'phase':'prepared','completed':carried[profile],'successes':sum(r['success'] for r in rows),'total':200})
    core.suite_progress(out,phase='prepared')
    common.save(out / 'continuation.json',{'original_total_slots':400,'carried_completed':carried,
        'remaining':{p:200-carried[p] for p in core.PROFILES},'started_slots_replayed':0,
        'collision_failure_settled_without_execution':SLOT,'previous':str(previous),'diagnosis':str(diagnosis)})
    print('CONTINUATION PREPARED',carried,flush=True)


class Backends(core.Backends):
    def start(self):
        original=core.subprocess.Popen
        def launch(command,*args,**kwargs):
            command=[str(CODE / 'checked_native_services.py') if str(v)==str(CODE / 'native_resident_services.py') else v for v in command]
            return original(command,*args,**kwargs)
        with patch.object(core.subprocess,'Popen',launch):super().start()
        self.config['service_entrypoint']=str(CODE / 'checked_native_services.py')
        common.save(self.campaign / 'resident/backend_config.json',self.config)


class Campaign(core.Campaign):
    def progress(self,worker,**state):
        with self.mutex:
            self.states[worker]={**self.states.get(worker,{}),**state,'updated_at':core.now()}
            common.save(self.out / 'workers' / (worker+'.json'),self.states[worker])
            rows=[]
            for file in (self.out / 'claims').glob('*.json'):
                claim=common.read(file)
                if claim['state']!='complete':continue
                trial=self.out / 'trials' / file.stem
                assert common.read(trial / 'independent_evaluation.json')['passed']
                rows.append(common.read(trial / 'result.json'))
            groups={layout:{'registered':40,'completed':sum(r['layout']==layout for r in rows),
                'physical_attempts':sum(r['physical_attempt'] for r in rows if r['layout']==layout),
                'successes':sum(r['success'] for r in rows if r['layout']==layout)} for layout in self.reg['layouts']}
            summary={'registered':200,'completed':len(rows),'planning_passes':sum(r['planning_success'] for r in rows),
                'physical_attempts':sum(r['physical_attempt'] for r in rows),'successes':sum(r['success'] for r in rows),
                'by_layout':groups,'failure_reasons':dict(Counter(r['reason'] for r in rows if not r['success'])),
                'overall_success_fraction':sum(r['success'] for r in rows)/200,'results':rows}
            common.save(self.out / 'summary.json',summary)
            common.save(self.out / 'progress.json',{'phase':'draining_after_error' if self.failed.is_set() else 'running',
                'completed':len(rows),'successes':summary['successes'],'total':200,'workers':self.states,
                'scene_workers':2,'backend_concurrency':1,'updated_at':core.now()})
            timing_summary(self.out)
            core.suite_progress(self.suite,phase='draining_after_error' if self.failed.is_set() else 'running',active=self.reg['pipeline_profile'])


def run(out):
    reg=common.read(out / 'registration.json');guard(reg)
    core.batch.__file__=str(Path(__file__).resolve())
    with (out / 'suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out / 'STOP').exists() or (out / 'suite_timing.json').exists():raise RuntimeError('Suite stopped or already started')
        backends=Backends(out);started=time.perf_counter();segment_start=core.now()
        prior=reg['prior_suite_timing'];gap=(datetime.fromisoformat(segment_start)-datetime.fromisoformat(prior['ended_at'])).total_seconds()
        clock={'started_at':prior['started_at'],'segment_started_at':segment_start,'phase':'starting_components',
            'prior_segments_total_seconds':prior['total_suite_wall_seconds'],'maintenance_gap_seconds':gap,
            'shared_startup_seconds':prior['shared_startup_seconds'],'rounds':{}}
        common.save(out / 'suite_timing.json',clock)
        try:
            core.suite_progress(out,phase='starting_components');backends.start()
            core.startup_validation(out,backends,reg)
            directory=out / 'startup_validation/known_collision';directory.mkdir()
            failed=Path(reg['previous_output']) / 'zerograsp_200/trials' / SLOT
            shutil.copy2(failed / 'state.json',directory / 'state.json')
            response=rpc(Path(backends.config['sockets']['curobo']),{'directory':str(directory),'observe':True},timeout=900)
            assert response['status']=='ok' and response['returncode']==2
            assert response['result']['reason']=='measured_start_state_in_collision'
            common.save(out / 'constraint_protocol_validation.json',{'passed':True,'response':response,
                'physical_attempts':0,'original_collision_gate_preserved':True,'startup_checks_before_new_claims':True})
            require_ready(out / 'resident',reg['required_components'])
            clock['shared_startup_seconds']=prior['shared_startup_seconds']+time.perf_counter()-started
            clock['phase']='running';common.save(out / 'suite_timing.json',clock)
            for profile in core.PROFILES:
                arm=out / profile;carried=reg['carried_round_seconds'][profile]
                if (out / 'STOP').exists():break
                phase_clock={'started_at':core.now(),'segment_started_monotonic':time.perf_counter(),
                    'started_monotonic':time.perf_counter()-carried,'carried_active_round_seconds':carried,
                    'phase':'running','shared_suite_startup_seconds':clock['shared_startup_seconds'],
                    'time_scope':'Sum of active round segments; maintenance gap is recorded separately at suite level.'}
                common.save(arm / 'round_timing.json',phase_clock)
                try:Campaign(arm).start()
                finally:
                    phase_clock.update(phase=common.read(arm / 'progress.json')['phase'],ended_at=core.now(),
                        round_wall_seconds=time.perf_counter()-phase_clock['started_monotonic'],
                        current_segment_wall_seconds=time.perf_counter()-phase_clock['segment_started_monotonic'])
                    common.save(arm / 'round_timing.json',phase_clock);timing_summary(arm)
                    clock['rounds'][profile]=phase_clock;common.save(out / 'suite_timing.json',clock)
                if common.read(arm / 'progress.json')['phase']!='complete':break
                health={k:rpc(Path(backends.config['sockets'][k]),{'command':'health'}) for k in reg['required_components']}
                common.save(arm / 'resident_health_after_round.json',health)
                if profile=='zerograsp_200':assert health['graspcl']['requests_completed']==0
            clock['phase']='complete' if all(common.read(out / p / 'progress.json')['phase']=='complete' for p in core.PROFILES) else 'stopped'
            core.suite_progress(out,phase=clock['phase'])
        except BaseException:
            clock['phase']='halted';common.save(out / 'halt.json',{'time':core.now(),'error':traceback.format_exc(),'replay_allowed':False})
            core.suite_progress(out,phase='halted');raise
        finally:
            shutdown=time.perf_counter();backends.stop()
            active=prior['total_suite_wall_seconds']+time.perf_counter()-started
            clock.update(ended_at=core.now(),shutdown_seconds=time.perf_counter()-shutdown,
                total_active_suite_wall_seconds=active,total_suite_wall_seconds=active+gap,
                all_owned_services_exited=all(p.poll() is not None for p in backends.children.values()))
            common.save(out / 'suite_timing.json',clock)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','execute'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous',type=Path)
    parser.add_argument('--diagnosis',type=Path)
    parser.add_argument('--entry',type=Path)
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.output.resolve(),args.previous.resolve(),args.diagnosis.resolve())
    elif args.stage=='run':run(args.output.resolve())
    else:execute(args.output.resolve(),args.entry.resolve())
