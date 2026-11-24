"""Offline sidecar fault injection; no ARX services, GPU jobs or grasp trials."""
import copy
import fcntl
import json
import os
import signal
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch, Mock

from . import watchdog as wd
from . import watchdog_recovery as recovery
from .common import ROOT
from .ledger import Ledger


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.parent = self.root / 'arx_parent'; self.parent.mkdir()
        self.home = self.root / 'sidecar'; self.home.mkdir()
        self.cfg = {'project_root': str(ROOT), 'initial_campaign': str(self.parent),
            'campaign_prefix': 'offline_arx_only', 'compute_uuid': 'compute', 'sim_uuids': ['sim0','sim3'],
            'collector_python': sys.executable, 'bulk_parent': str(self.root / 'bulk'),
            'poll_seconds': 1, 'backoff_seconds': 0, 'resource_poll_seconds': 1}
        wd.write(self.home / 'config.json', self.cfg)
        wd.write(self.parent / 'registration.json', {'candidate_backend': 'graspness', 'deadline_unix': time.time()+3600})
        wd.write(self.parent / 'launch.json', {'started_unix': time.time()-1000,'stop_unix': time.time()+3600})
        wd.write(self.parent / 'state.json', {'status':'running'})
        wd.write(self.parent / 'coordinator_process.json', {'pid':-10})
        for kind in ('planner','graspness'):
            wd.write(self.parent / 'workers' / kind / 'process.json', {'pid':-11})
        self.guard = wd.Watchdog(self.home / 'config.json')

    def tearDown(self):
        self.tmp.cleanup()

    def test_live_owned_process_then_actual_exit_is_detected(self):
        # A disposable Python sentinel with ARX-shaped argv, NOT a robot process.
        command=[sys.executable,'-c','import time; time.sleep(30)',
                 'scripts.curobo_arx.data_factory.launch', str(self.parent)]
        process=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            info=wd.owned_process({'pid':process.pid},self.parent,'coordinator')
            self.assertIsNotNone(info)
            self.assertIsNone(wd.owned_process({'pid':process.pid,'start_ticks':info['start_ticks']+1},self.parent,'coordinator'))
            self.assertIsNone(wd.owned_process({'pid':process.pid},self.root/'foreign_campaign','coordinator'))
            process.terminate();process.wait(timeout=5)
            self.assertIsNone(wd.owned_process({'pid':process.pid},self.parent,'coordinator'))
        finally:
            if process.poll() is None: process.kill();process.wait()

    def test_foreign_uid_never_matches_even_with_identical_arguments(self):
        info={'pid':10,'uid':os.getuid()+1,'argv':['scripts.curobo_arx.data_factory.launch',str(self.parent)],'boot_id':'x','start_ticks':5}
        with patch.object(wd,'process_info',return_value=info):
            self.assertIsNone(wd.owned_process({'pid':10},self.parent,'coordinator'))

    def test_manual_stop_does_not_restart_or_overwrite_marker(self):
        wd.write(self.parent/'STOP',{'reason':'user requested stop'})
        before=wd.sha(self.parent/'STOP')
        with patch.object(wd.subprocess,'Popen') as spawn:
            self.assertFalse(self.guard.step())
            spawn.assert_not_called()
        self.assertEqual(self.guard.state['phase'],'manual_stop')
        self.assertEqual(wd.sha(self.parent/'STOP'),before)

    def test_normal_task_completion_never_restarts(self):
        for status in ('quota_complete','budget_exhausted'):
            wd.write(self.parent/'state.json',{'status':status})
            self.assertEqual(self.guard.health()[0],'complete')

    def test_expired_original_deadline_never_restarts(self):
        wd.write(self.parent/'launch.json',{'started_unix':0,'stop_unix':1})
        self.assertEqual(self.guard.health()[0],'complete')

    def test_dead_coordinator_requires_two_checks_then_creates_incident(self):
        self.guard.step();self.assertEqual(self.guard.state['phase'],'watching')
        self.guard.step();self.assertEqual(self.guard.state['phase'],'draining')
        self.assertEqual(wd.read(self.parent/'STOP')['reason'],wd.AUTO_STOP)
        self.assertTrue((Path(self.guard.state['incident'])/'trigger.json').exists())

    def test_preexisting_automatic_stop_is_preserved(self):
        wd.write(self.parent/'STOP',{'reason':'infrastructure/record/resource fault','original':True})
        before=wd.sha(self.parent/'STOP')
        self.guard.begin_recovery('offline process death')
        self.assertEqual(wd.sha(self.parent/'STOP'),before)

    def test_slow_initialization_without_heartbeat_is_not_death(self):
        with patch.object(self.guard,'alive',return_value={'pid':123}):
            self.assertEqual(self.guard.health()[0],'healthy')

    def test_dead_worker_is_detected(self):
        worker=self.parent/'workers/planner/process.json';wd.write(worker,{'pid':-11})
        with patch.object(self.guard,'alive',side_effect=[{'pid':123},None]):
            self.assertEqual(self.guard.health()[0],'failed')

    def make_group(self):
        wd.write(self.parent/'group_registrations/group_offline.json',{'group_id':'group_offline',
            'registration_sha256':wd.sha(self.parent/'registration.json')})
        wd.write(self.parent/'group_registrations/group_offline.process.json',{'pid':-12,'container':'arx-data-factory-group_offline'})

    def test_normal_renderer_rotation_is_not_failed_simulator(self):
        self.make_group()
        wd.write(self.parent/'groups/group_offline/block_result.json',{'idle':True})
        with patch.object(self.guard,'alive',return_value={'pid':123}), patch.object(wd,'container_running') as probe:
            self.assertEqual(self.guard.health()[0],'healthy');probe.assert_not_called()

    def test_unexpected_simulator_exit_is_detected(self):
        self.make_group()
        with patch.object(self.guard,'alive',side_effect=lambda path,role: None if role=='group' else {'pid':123}):
            action,reason=self.guard.health()
            self.assertEqual(action,'failed');self.assertIn('simulator',reason)

    def test_failed_container_inventory_cannot_prove_absence(self):
        with patch.object(wd.subprocess,'run',return_value=subprocess.CompletedProcess([],125)):
            with self.assertRaisesRegex(RuntimeError,'unverified'):wd.container_running('arx-offline')

    def test_integrity_fault_blocks_instead_of_restart(self):
        wd.write(self.parent/'FAULT.json',{'error':'Frozen physical configuration mismatch: offline fixture'})
        self.guard.step();self.assertEqual(self.guard.state['phase'],'blocked')
        self.assertFalse((self.parent/'STOP').exists())

    def test_unconfirmed_old_world_blocks_after_drain_limit(self):
        self.guard.begin_recovery('offline process death')
        self.guard.state['drain_started']=0
        self.guard.drain();self.assertEqual(self.guard.state['phase'],'blocked')

    def test_live_old_container_is_only_asked_to_drain_not_restarted(self):
        self.make_group();self.guard.begin_recovery('offline process death')
        with patch.object(wd,'container_running',return_value=True),patch.object(wd.subprocess,'run') as command:
            self.guard.drain()
            self.assertEqual(self.guard.state['phase'],'draining')
            self.assertEqual(command.call_args.args[0][:4],['podman','kill','--signal','TERM'])

    def test_resource_wait_keeps_same_parent_and_does_not_spawn(self):
        self.guard.begin_recovery('offline process death');self.guard.state['phase']='waiting_resources'
        self.guard.state['retry_after']=0
        def helper(*args,**kwargs):
            result=Path(self.guard.state['incident'])/'recovery_result.json'
            wd.write(result,{'status':'resource_wait','reason':'offline GPU busy'})
            return subprocess.CompletedProcess([],0)
        with patch.object(wd.subprocess,'run',side_effect=helper),patch.object(wd.subprocess,'Popen') as spawn:
            self.guard.recover();spawn.assert_not_called()
        self.assertEqual(self.guard.state['phase'],'waiting_resources')
        self.assertEqual(self.guard.state['campaign'],str(self.parent))

    def test_recovery_preparation_timeout_does_not_launch_or_blindly_retry(self):
        self.guard.begin_recovery('offline process death');self.guard.state['phase']='waiting_resources'
        self.guard.state['retry_after']=0
        with patch.object(wd.subprocess,'run',side_effect=subprocess.TimeoutExpired(['offline_helper'],1800)), \
             patch.object(wd.subprocess,'Popen') as spawn:
            self.guard.recover();spawn.assert_not_called()
        self.assertEqual(self.guard.state['phase'],'blocked')

    def test_second_watchdog_cannot_attach_to_same_state_directory(self):
        with (self.home/'watchdog.lock').open('a+') as held:
            fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):self.guard.run()

    def test_restart_rate_limit_waits_without_resetting_parent_budget(self):
        self.guard.begin_recovery('offline process death');self.guard.state['phase']='waiting_resources'
        self.guard.state['restart_times']=[time.time()-1]*4
        with patch.object(wd.subprocess,'run') as helper:
            self.guard.recover();helper.assert_not_called()
        self.assertGreater(self.guard.state['retry_after'],time.time())

    def test_prepared_result_is_adopted_after_watchdog_itself_restarts(self):
        self.guard.begin_recovery('offline process death');self.guard.state['phase']='waiting_resources'
        self.guard.state['retry_after']=0
        out=self.root/'child';out.mkdir()
        wd.write(Path(self.guard.state['incident'])/'recovery_result.json',{'status':'prepared','campaign':str(out)})
        with patch.object(wd.subprocess,'run') as helper:
            self.guard.recover();helper.assert_not_called()
        self.assertEqual(self.guard.state['prepared_campaign'],str(out))

    def test_existing_launched_child_is_adopted_without_duplicate_spawn(self):
        out=self.root/'child';out.mkdir();wd.write(out/'launch.json',{'offline':True})
        self.guard.state.update(phase='prepared',prepared_campaign=str(out))
        with patch.object(wd.subprocess,'Popen') as spawn:
            self.guard.start_prepared();spawn.assert_not_called()
        self.assertEqual(self.guard.state['campaign'],str(out))
        self.assertEqual(self.guard.state['phase'],'watching')

    def test_user_stop_during_preparation_prevents_spawn(self):
        out=self.root/'child';out.mkdir()
        self.guard.state.update(phase='prepared',prepared_campaign=str(out))
        wd.write(self.parent/'STOP',{'reason':'user requested stop'})
        with patch.object(wd.subprocess,'Popen') as spawn:
            self.guard.start_prepared();spawn.assert_not_called()
        self.assertEqual(self.guard.state['phase'],'manual_stop')

    def test_crash_before_launch_record_is_still_detected(self):
        (self.parent/'launch.json').unlink()
        wd.write(self.parent/'state.json',{'status':'resource_wait'})
        self.assertEqual(self.guard.health()[0],'failed')

    def test_actual_bootstrap_respawn_and_no_duplicate_without_any_robot_service(self):
        project=self.root/'software_sentinel_only'
        module=project/'scripts/curobo_arx/data_factory/launch.py'
        module.parent.mkdir(parents=True)
        module.write_text("import json,sys,time\nfrom pathlib import Path\n"
            "p=Path(sys.argv[1]); (p/'launch.json').write_text(json.dumps({'started_unix':time.time(),'stop_unix':time.time()+1000}))\n"
            "time.sleep(60)\n")
        out=self.root/'sentinel_child';out.mkdir()
        wd.write(out/'registration.json',{'candidate_backend':'graspness','resume':{'parent':str(self.parent)}})
        wd.write(out/'state.json',{'status':'prepared'})
        self.cfg['project_root']=str(project);wd.write(self.home/'config.json',self.cfg)
        self.guard.cfg=self.cfg;self.guard.state.update(phase='prepared',prepared_campaign=str(out))
        spawned=[];real_popen=subprocess.Popen
        def spawn(*args,**kwargs):
            process=real_popen(*args,**kwargs);spawned.append(process);return process
        try:
            with patch.object(wd.subprocess,'Popen',side_effect=spawn):
                self.guard.start_prepared();self.guard.start_prepared()
                end=time.monotonic()+20
                while time.monotonic()<end and self.guard.state['phase']=='prepared':
                    self.guard.start_prepared();time.sleep(.05)
            self.assertEqual(len(spawned),1)
            self.assertEqual(self.guard.state['phase'],'watching')
            meta=wd.read(out/'coordinator_process.json')
            self.assertIsNotNone(wd.owned_process(meta,out,'coordinator'))
            spawned[0].terminate();spawned[0].wait(timeout=5)
            self.guard.step();self.guard.step()
            self.assertEqual(self.guard.state['phase'],'draining')
            self.assertEqual(wd.read(out/'STOP')['reason'],wd.AUTO_STOP)
        finally:
            for process in spawned:
                if process.poll() is None:process.kill();process.wait()

    def test_SIGTERM_exits_without_creating_persistent_user_stop_and_can_resume(self):
        callbacks={};original_step=self.guard.step
        def tick():
            callbacks[signal.SIGTERM](None,None)
            return original_step()
        with patch.object(wd.signal,'signal',side_effect=lambda code,handler:callbacks.update({code:handler})), \
             patch.object(self.guard,'step',side_effect=tick):
            self.guard._run_locked()
        self.assertFalse((self.home/'STOP').exists())
        self.assertEqual(self.guard.state['phase'],'watchdog_exit')
        resumed=wd.Watchdog(self.home/'config.json')
        self.assertEqual(resumed.state['phase'],'watching')

    def test_service_exit_preserves_inflight_recovery_phase_for_next_start(self):
        self.guard.state.update(phase='waiting_resources',incident='offline_incident',retry_after=123)
        self.guard.terminate=True;self.assertFalse(self.guard.step())
        resumed=wd.Watchdog(self.home/'config.json')
        self.assertEqual(resumed.state['phase'],'waiting_resources')
        self.assertEqual(resumed.state['incident'],'offline_incident')
        self.assertEqual(resumed.state['retry_after'],123)

    def test_explicit_watchdog_STOP_remains_persistent_after_service_restart(self):
        wd.write(self.home/'STOP',{'reason':'User explicitly stopped watchdog'})
        self.assertFalse(self.guard.step())
        resumed=wd.Watchdog(self.home/'config.json')
        self.assertFalse(resumed.step())
        self.assertEqual(resumed.state['phase'],'manual_stop')

    def test_missing_data_mount_waits_without_claiming_or_stopping_parent(self):
        with patch.object(wd,'storage_ready',return_value=False):
            self.guard.step();self.guard.step()
        self.assertEqual(self.guard.state['phase'],'watching')
        self.assertEqual(self.guard.state['health'],'resource_wait')
        self.assertFalse((self.parent/'STOP').exists())

    def test_expected_ceph_mount_does_not_accept_underlying_root_filesystem(self):
        mounts=self.root/'mountinfo'
        mounts.write_text('1 0 0:1 / / rw - ext4 /dev/root rw\n')
        cfg={'required_bulk_mount':{'target':'/coolas/data','fstype':'ceph'}}
        self.assertFalse(wd.storage_ready(cfg,mounts))
        mounts.write_text(mounts.read_text()+'2 1 0:2 / /coolas/data rw - ceph server:/data rw\n')
        self.assertTrue(wd.storage_ready(cfg,mounts))

    def test_missing_or_wrong_storage_marker_is_not_ready(self):
        marker=self.root/'marker';cfg={'storage_readiness_marker':{'path':str(marker),'contents':'ready'}}
        self.assertFalse(wd.storage_ready(cfg))
        marker.write_text('wrong');self.assertFalse(wd.storage_ready(cfg))
        marker.write_text('ready');self.assertTrue(wd.storage_ready(cfg))

    def test_recovery_waits_for_mount_before_examining_missing_episode_files(self):
        request=self.home/'request.json'
        wd.write(request,{'parent':str(self.parent),'out':str(self.root/'child'),'watchdog_config':str(self.home/'config.json')})
        with patch.object(recovery,'storage_ready',return_value=False), \
             patch.object(recovery,'inspect_records') as inspect:
            result=recovery.prepare(request);inspect.assert_not_called()
        self.assertEqual(result['status'],'resource_wait')


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.episode=self.root/'episode';self.episode.mkdir()
        self.db=sqlite3.connect(self.root/'ledger.sqlite3')
        self.db.execute('CREATE TABLE attempts(id,state,episode,receipt)')

    def tearDown(self):
        self.db.close();self.tmp.cleanup()

    def row(self,state='executing'):
        self.db.execute('INSERT INTO attempts VALUES (?,?,?,?)',('arx_offline',state,str(self.episode),None));self.db.commit()

    def test_execution_receipt_without_command_blocks_recovery(self):
        self.row();wd.write(self.episode/'requests.json',[{'label':'close'}])
        wd.write(self.episode/'close_execution.json',{'offline':True})
        with self.assertRaisesRegex(ValueError,'lost its command'):recovery.inspect_records(self.root)

    def test_execution_receipt_without_request_blocks_recovery(self):
        self.row();wd.write(self.episode/'close_execution.json',{'offline':True})
        with self.assertRaisesRegex(ValueError,'no command request'):recovery.inspect_records(self.root)

    def test_phase_report_without_request_or_execution_still_blocks_recovery(self):
        self.row();wd.write(self.episode/'grasp_report.json',{'stage_timing':{'close':{}}})
        with self.assertRaisesRegex(ValueError,'Stage report'):recovery.inspect_records(self.root)

    def test_physical_report_without_request_or_execution_still_blocks_recovery(self):
        self.row();wd.write(self.episode/'physical_receipt.json',{'executed_stages':['close']})
        with self.assertRaisesRegex(ValueError,'Stage report'):recovery.inspect_records(self.root)

    def test_reported_stage_with_lost_command_file_blocks_recovery(self):
        self.row();wd.write(self.episode/'requests.json',[{'label':'close'}])
        wd.write(self.episode/'grasp_report.json',{'stage_timing':{'close':{}}})
        with self.assertRaisesRegex(ValueError,'lost its command'):recovery.inspect_records(self.root)

    def test_interrupted_command_without_execution_stays_unknown_and_can_be_copied(self):
        self.row();wd.write(self.episode/'requests.json',[{'label':'close'}])
        wd.write(self.episode/'close_command.json',{'offline':True})
        rows=recovery.inspect_records(self.root)
        self.assertEqual(rows[0]['state'],'executing')

    def test_terminal_failure_missing_execution_receipt_blocks_recovery(self):
        self.row('failed');wd.write(self.episode/'requests.json',[{'label':'close'}])
        wd.write(self.episode/'close_command.json',{'offline':True})
        with self.assertRaisesRegex(ValueError,'execution evidence'):recovery.inspect_records(self.root)

    def test_terminal_failure_missing_outcome_receipt_blocks_recovery(self):
        self.row('failed');wd.write(self.episode/'requests.json',[])
        with self.assertRaisesRegex(ValueError,'outcome evidence'):recovery.inspect_records(self.root)


class RecoveryCloneTests(unittest.TestCase):
    def test_real_queue_copy_retains_original_unknown_attempt_budget_and_deadline(self):
        # Actual ARX queue code, with synthetic records; never starts a service.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'parent';parent.mkdir();home=root/'watchdog';home.mkdir();out=root/'child'
            q={'max_total_dispatches':100,'max_physical_attempts':4,'candidate_reserve':8,
               'goals_per_object':2,'distances_mm':[100],'successes_per_bin':1,
               'physical_per_bin':32,'planning_rejections_per_bin':64}
            catalog={'objects':[{'object_id':'arx_offline_fixture'}]}
            ledger=Ledger(parent/'ledger.sqlite3');ledger.initialize(catalog,q)
            import numpy as np
            ledger.add_goals('arx_offline_fixture',{'candidates':[{'goal_id':'g','grasp_world':np.eye(4).tolist()}]})
            attempt=ledger.claim('arx_offline_fixture','old','env0')
            ledger.transition(attempt['id'],'old','executing',physical=True);ledger.close()
            deadline=time.time()+3600
            files={str(ROOT/('scripts/curobo_arx/data_factory/'+name+'.py')):wd.sha(ROOT/('scripts/curobo_arx/data_factory/'+name+'.py'))
                   for name in ('launch','common','ledger','report','storage')}
            reg={'stage':'graspness_teacher','source_snapshot':str(ROOT),'source_files':files,
                'execution_source_files':{},'dependencies':{},'envs_per_gpu':2,'catalog':catalog,'quotas':q,
                'wall_hours':168,'deadline_unix':deadline,'campaign_started_unix':time.time()-1000}
            wd.write(parent/'registration.json',reg);wd.write(parent/'state.json',{'status':'running'})
            wd.write(parent/'launch.json',{'started_unix':time.time()-1000,'stop_unix':deadline})
            wd.write(parent/'coordinator_process.json',{'pid':-101});wd.write(parent/'STOP',{'reason':wd.AUTO_STOP})
            for name in ['storage_policy.json','method_receipt.json']:wd.write(parent/name,{})
            cfg={'project_root':str(ROOT),'compute_uuid':'compute','sim_uuids':['sim'], 'bulk_parent':str(root/'bulk')}
            wd.write(home/'config.json',cfg)
            wd.write(home/'request.json',{'parent':str(parent),'out':str(out),'watchdog_config':str(home/'config.json')})
            original=wd.sha(parent/'ledger.sqlite3');original_state=wd.sha(parent/'state.json')
            inventory=[{'index':2,'uuid':'compute','free_mb':20000,'arch':'8.9'},
                       {'index':3,'uuid':'sim','free_mb':20000,'arch':'8.9'}]
            with patch('scripts.curobo_arx.data_factory.launch.devices',return_value=inventory), \
                 patch('scripts.curobo_arx.data_factory.storage.register_bulk_storage',return_value={'root':str(root/'bulk')}), \
                 patch('scripts.curobo_arx.data_factory.storage.container_storage_mounts',return_value=[]), \
                 patch.object(recovery,'probe_storage',return_value={'pass':True,'offline_synthetic':True}):
                result=recovery.prepare(home/'request.json')
            self.assertEqual(result['status'],'prepared')
            child=Ledger(out/'ledger.sqlite3')
            try:
                self.assertEqual(child.rows()[0]['state'],'recovered_interrupted')
                self.assertEqual(child.rows()[0]['physical'],1)
                self.assertEqual(child.quotas,q)
            finally:child.close()
            self.assertEqual(wd.sha(parent/'ledger.sqlite3'),original)
            self.assertEqual(wd.sha(parent/'state.json'),original_state)
            self.assertEqual(wd.read(out/'registration.json')['deadline_unix'],deadline)


if __name__=='__main__':
    unittest.main()
