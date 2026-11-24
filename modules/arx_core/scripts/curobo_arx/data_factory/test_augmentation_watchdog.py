"""Offline monitor faults plus retained actual ARX seed/ledger evidence."""
import copy
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from . import watchdog as wd
from . import augmentation_watchdog as aug
from .common import ROOT
from scripts.curobo_arx.seed_augmentation import production as prod
from scripts.curobo_arx.seed_augmentation.contract import controls_ready


class AugmentationMonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.home=self.root/'augmentation';self.home.mkdir()
        self.cfg={'home':str(self.home),'project_root':str(ROOT),'runs_root':str(self.root),
            'campaign_prefix':'arx_aug','max_slots':24,'deadline_unix':time.time()+3600,
            'monitor_python':sys.executable,'preferred_gpu_uuids':['spare'],
            'backoff_seconds':0,'max_restarts_per_hour':2,'failure_confirmations':2}
        wd.write(self.home/'config.json',self.cfg)
        self.guard=aug.AugmentationWatchdog(self.home/'config.json',self.root/'watch/config.json',lambda:False)

    def tearDown(self):self.tmp.cleanup()

    def campaign(self):
        path=self.root/'arx_aug_01';path.mkdir()
        wd.write(path/'registration.json',{'seed_trials':[{'trial_id':'arx_slot'}]})
        wd.write(path/'state.json',{'status':'running'})
        self.guard.save(phase='watching',campaign=str(path),launch_started=time.time()-1000)
        return path

    def test_owned_python_exit_pid_reuse_and_foreign_scope(self):
        campaign=self.campaign()
        proc=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)',
            'scripts.curobo_arx.seed_augmentation.entrypoint',str(campaign)],stdout=subprocess.DEVNULL)
        try:
            meta=wd.process_info(proc.pid)
            self.assertTrue(aug.owned(meta,campaign,'coordinator'))
            self.assertIsNone(aug.owned({**meta,'start_ticks':meta['start_ticks']+1},campaign,'coordinator'))
            self.assertIsNone(aug.owned(meta,self.root/'other','coordinator'))
            self.assertIsNone(aug.prove_exited(campaign))
            proc.terminate();proc.wait(timeout=5)
            self.assertIsNone(aug.owned(meta,campaign,'coordinator'))
            self.assertTrue(aug.prove_exited(campaign)['all_virtual_Worlds_absent'])
        finally:
            if proc.poll() is None:proc.kill();proc.wait()

    def test_group_hash_is_bound_to_registration(self):
        campaign=self.campaign()
        wd.write(campaign/'group_registrations/seed_raytracedlighting.json',
            {'group_id':'seed_raytracedlighting','registration_sha256':'altered',
             'renderer':'RayTracedLighting','trial_ids':['arx_slot'],'env_count':1})
        with self.assertRaisesRegex(ValueError,'identity mismatch'):list(aug.group_records(campaign))

    def test_live_world_prevents_any_successor(self):
        campaign=self.campaign()
        group=(dict(group_id='seed',renderer='RayTracedLighting'),campaign/'absent.json','owned-arx-world')
        with patch.object(aug,'group_records',return_value=iter([group])),patch.object(wd,'container_running',return_value=True):
            self.assertIsNone(aug.prove_exited(campaign))

    def test_unavailable_container_inventory_cannot_prove_absence(self):
        campaign=self.campaign()
        group=({},campaign/'absent.json','owned-arx-world')
        with patch.object(aug,'group_records',return_value=iter([group])),patch.object(wd,'container_running',side_effect=RuntimeError('unavailable')):
            with self.assertRaises(RuntimeError):aug.prove_exited(campaign)

    def test_automatic_exit_requires_confirmation_and_keeps_old_stop(self):
        campaign=self.campaign()
        wd.write(campaign/'STOP',{'reason':aug.AUTO,'original':True})
        before=wd.sha(campaign/'STOP')
        self.guard.monitor();self.assertEqual(self.guard.state['phase'],'watching')
        self.guard.monitor();self.assertEqual(self.guard.state['phase'],'draining')
        self.assertEqual(wd.sha(campaign/'STOP'),before)
        self.assertTrue(self.guard.state['resume'])

    def test_manual_stop_never_restarts(self):
        campaign=self.campaign();wd.write(campaign/'STOP',{'reason':'user stopped augmentation'})
        with patch.object(self.guard,'spawn') as spawn:
            self.guard.monitor();spawn.assert_not_called()
        self.assertEqual(self.guard.state['phase'],'manual_stop')

    def test_caps_and_original_deadline_end_queue(self):
        for allocated,deadline in ((24,time.time()+3600),(0,1)):
            self.guard.cfg['deadline_unix']=deadline;self.guard.save(phase='idle',allocated_slots=allocated)
            with patch.object(self.guard,'spawn_job') as spawn:
                self.guard.step();spawn.assert_not_called()
            self.assertEqual(self.guard.state['phase'],'complete')

    def test_mount_not_ready_does_not_interpret_missing_records(self):
        self.guard.save(phase='helper',job='/missing')
        with patch.object(wd,'storage_ready',return_value=False),patch.object(self.guard,'helper') as helper:
            self.guard.step();helper.assert_not_called()
        self.assertEqual(self.guard.state['health'],'waiting_storage')

    def test_dead_readonly_helper_is_recreated_without_replaying_trials(self):
        request=self.home/'jobs/old.json'
        wd.write(request,{'kind':'finish','config':str(self.home/'config.json'),'result':str(self.home/'missing.result.json'),
            'process_metadata':str(self.home/'missing.process.json'),'campaign':'original_arx_batch','out':'old_partial','resume':True})
        self.guard.save(phase='helper',job=str(request),job_started=time.time()-1000)
        with patch.object(self.guard,'spawn_job') as spawn:
            self.guard.helper()
            args=spawn.call_args
            self.assertEqual(args.args,('finish',));self.assertEqual(args.kwargs['campaign'],'original_arx_batch')
            self.assertNotEqual(args.kwargs['out'],'old_partial')

    def test_repeated_process_failure_waits_after_rate_limit(self):
        self.assertTrue(self.guard.recovery_allowed('one'))
        self.assertTrue(self.guard.recovery_allowed('two'))
        self.assertFalse(self.guard.recovery_allowed('three'))
        self.assertEqual(self.guard.state['health'],'recovery_backoff')

    def test_finished_batch_rolls_to_new_seeds_without_resetting_budget(self):
        request=self.home/'jobs/finish.json';result=self.home/'jobs/result.json'
        wd.write(request,{'result':str(result)})
        wd.write(result,{'status':'finished','campaign':'old','physical_attempts':3,'admitted':2,
            'admitted_by_policy':{'stitch':2}})
        self.guard.save(phase='helper',job=str(request),allocated_slots=12,
            batches=[{'campaign':'old','accounted':False}],reserved_seed_ids=['used_seed'])
        self.guard.helper()
        self.assertEqual(self.guard.state['phase'],'idle')
        self.assertEqual(self.guard.state['allocated_slots'],12)
        self.assertEqual(self.guard.state['reserved_seed_ids'],['used_seed'])
        self.assertEqual(self.guard.state['admitted_by_policy'],{'stitch':2})

    def test_native_wait_does_not_block_augmentation_monitor(self):
        config=self.root/'watch/config.json'
        wd.write(config,{'initial_campaign':str(self.root/'native'),
            'augmentation':{'config':str(self.home/'config.json')}})
        native=wd.Watchdog(config);observed=threading.Event()
        def native_step():
            self.assertTrue(observed.wait(2));native.terminate=True;return False
        with patch.object(aug.AugmentationWatchdog,'run',lambda _:observed.set()),patch.object(native,'step',native_step):
            native._run_locked()

    def test_inherited_unknown_or_duplicate_controls_cannot_activate(self):
        controls=[{'seed_id':'arx_seed','policy':'control','trial_id':str(i),'completed':True,'admitted':True} for i in range(2)]
        self.assertTrue(controls_ready('arx_seed',controls))
        self.assertFalse(controls_ready('arx_seed',[controls[0],controls[0]]))
        self.assertFalse(controls_ready('arx_seed',[controls[0],{**controls[1],'completed':False,'admitted':False}]))
        self.assertFalse(controls_ready('other_seed',controls))


class ActualArxRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=ROOT/'runs/seed_augmentation_20261104_03'
        cls.reg=wd.read(cls.original/'registration.json')
        cls.trial=cls.reg['seed_trials'][0]
        cls.db=cls.original/'trial_ledgers'/(cls.trial['trial_id']+'.sqlite3')
        cls.row=prod.trial_rows(cls.db)[0]

    def make_campaign(self,root):
        reg=copy.deepcopy(self.reg);reg['seed_trials']=[copy.deepcopy(self.trial)]
        wd.write(root/'registration.json',reg)
        target=root/'trial_ledgers'/self.db.name;target.parent.mkdir();shutil.copy2(self.db,target)
        episode=root/'groups/seed/trials'/self.trial['trial_id']/'episode';episode.mkdir(parents=True)
        for p in Path(self.row['episode']).glob('*.json'):shutil.copy2(p,episode/p.name)
        db=sqlite3.connect(target);db.execute('UPDATE attempts SET episode=?',(str(episode),));db.commit();db.close()
        return target,episode

    def test_real_arx_receipt_without_command_blocks_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,episode=self.make_campaign(root)
            self.assertTrue((episode/'close_execution.json').exists())
            (episode/'close_command.json').unlink()
            with self.assertRaisesRegex(ValueError,'lost its command file'):prod.gather(root)

    def test_interrupted_real_arx_attempt_is_unknown_and_never_unused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target,_=self.make_campaign(root)
            db=sqlite3.connect(target);db.execute("UPDATE attempts SET state='executing'");db.commit();db.close()
            _,done,started,physical,unknown=prod.gather(root)
            self.assertEqual(done,{})
            self.assertIn(self.trial['trial_id'],started)
            self.assertEqual(unknown,[self.trial['trial_id']]);self.assertEqual(physical,1)

    def test_planned_perturbations_preserve_actual_arx_g_and_placement(self):
        seed_id=self.trial['seed_id'];seed=self.reg['seeds'][seed_id];before=copy.deepcopy(seed)
        trials=prod.plan_trials({seed_id:seed})
        self.assertEqual(seed,before);self.assertEqual(len(trials),6)
        self.assertEqual([t['policy'] for t in trials],['control','control','stitch','direct','direct','stitch'])
        self.assertTrue(all(t['placement']==seed['placement'] for t in trials))
        self.assertEqual(len({t['trial_id'] for t in trials}),6)

    def test_missing_accepted_source_record_blocks_selection_instead_of_skipping(self):
        parent=ROOT/'runs/graspness_household_20261104_resume_01'
        with patch.object(prod,'source_seed',side_effect=ValueError('Successful seed is missing command/execution or provenance files')):
            with self.assertRaisesRegex(RuntimeError,'integrity fault'):prod.select_seeds(parent,set(),1)

    def test_pre_dispatch_rejection_is_known_without_post_execution_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target,episode=self.make_campaign(root)
            # Reuse the ARX ledger schema, with a rejected non-dispatch slot.
            db=sqlite3.connect(target)
            db.execute("UPDATE attempts SET state='rejected',physical=0,episode=NULL,outcome='no_valid_G'")
            db.commit();db.close()
            _,done,started,physical,unknown=prod.gather(root)
            self.assertEqual(physical,0);self.assertEqual(unknown,[])
            self.assertFalse(done[self.trial['trial_id']]['admitted'])
            self.assertIn(self.trial['trial_id'],started)


if __name__=='__main__':unittest.main()
