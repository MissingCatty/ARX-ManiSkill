"""Isolated one-environment success-seed trials with fresh wrist recordings."""
import argparse
import copy
from pathlib import Path
import time
import traceback
import uuid
import numpy as np
from scripts.curobo_arx.data_factory.common import read, write, space, verify_files
from scripts.curobo_arx.data_factory.frozen import fingerprint
from scripts.curobo_arx.data_factory.ledger import Ledger
from scripts.curobo_arx.data_factory.runtime import Group, Environment
from .contract import join_check, replay_command, audit_episode, anchor_readiness, controls_ready


class SeedEnvironment(Environment):
    def record(self):
        stage = self.stage
        if stage == 'approach_connect':
            # The raw command retains its real connector label. Both moving
            # pre-close segments have approach semantics for the shared codec.
            self.stage = 'approach'
        try:
            return super().record()
        finally:
            self.stage = stage

    def dispatch(self, command, desired):
        saved = self.actual_entry
        if command['label'] == 'approach_connect' and saved is None:
            saved = self.last_observation
        super().dispatch(command, desired)
        if saved is not None:
            self.actual_entry = saved

    def join(self):
        result = join_check(self.group.active_seed, self.bridge.snapshot())
        write(self.episode/'join_state_check.json', result)
        return result['pass']

    def submit_plan(self, action, phase, label, goal):
        snapshot = self.bridge.snapshot(); scene = self.bridge.export_scene()
        path = self.episode/(label+'_scene.json'); write(path, scene)
        self.future = self.group.planner.submit(action, phase=phase, label=label,
            seed_id=self.group.active_seed['seed_id'], snapshot=snapshot, scene=str(path),
            goal_world=np.asarray(goal).tolist(), grasp_world=self.task['goal']['grasp_world'],
            output=str(self.episode), visual_context=self.visual_context(),
            command_id=uuid.uuid4().hex, minimum_duration=6. if label == 'pregrasp' else 3.)
        self.state = 'planning_'+phase

    def plan(self, phase, world):
        trial = self.group.active_trial; seed = self.group.active_seed; mode = trial['policy']
        if phase == 'reach' and mode == 'control':
            return self.submit_plan('seed_connect', phase, 'pregrasp', seed['anchor_world'])
        if phase == 'approach' and mode == 'stitch' and not getattr(self, 'connected', False):
            return self.submit_plan('seed_connect', 'connect', 'approach_connect', seed['anchor_world'])
        if phase == 'approach' and mode == 'control' and not getattr(self, 'anchor_ready', False):
            self.anchor_arrival = self.group.world.current_time
            self.state = 'settling_anchor'
            return
        if mode in ('control', 'stitch') and phase in ('approach', 'lift'):
            return self.submit_plan('seed_replay', phase, phase, world)
        return super().plan(phase, world)

    def dwell(self, label, desired):
        if self.group.active_trial['policy'] in ('control', 'stitch'):
            self.dispatch(replay_command(self.group.active_seed, label), desired)
        else:
            super().dwell(label, desired)

    def tick(self):
        if self.state == 'settling_anchor':
            if self.bridge.active or self.bridge.fault:
                raise RuntimeError('Anchor settling requires an idle, healthy original controller')
            if time.monotonic()-self.started > self.group.registration['attempt_seconds']:
                raise TimeoutError('Anchor settling exceeded the existing attempt wall budget')
            result = anchor_readiness(self.group.active_seed, self.bridge.snapshot(),
                                      self.group.world.current_time-self.anchor_arrival)
            status = result['anchor_wait']['status']
            if status == 'waiting':return
            write(self.episode/'join_state_check.json', result)
            if not result['pass']:return self.finish('seed_join_rejected')
            self.anchor_ready = True
            return self.plan('approach', self.task['goal']['grasp_world'])
        if self.state != 'executing' or self.last_command['label'] != 'approach_connect':
            return super().tick()
        if self.bridge.fault:
            raise RuntimeError(str(self.bridge.fault))
        if time.monotonic()-self.started > self.group.registration['attempt_seconds']:
            raise TimeoutError('Seed connector attempt budget exceeded')
        result = self.bridge.result()
        if result is None:
            return
        from scripts.curobo_arx.settling import assess_settling
        write(self.episode/'approach_connect_execution.json', result)
        assessment = assess_settling(result, self.desired)
        write(self.episode/'approach_connect_settling.json', assessment)
        self.stages.append('approach_connect')
        target = result['final']['objects']['test_object']['path']
        touched = any(c['num_contacts'] > 0 and target in c['a']+c['b'] for c in result['contacts'])
        forbidden = [c for c in result['contacts'] if c['num_contacts'] > 0
            and not ('/World/Robot/' in c['a'] and '/World/Robot/' in c['b'])
            and not (target in c['a']+c['b'] and any(n in c['a']+c['b'] for n in ('right_link27', 'right_link28')))]
        if forbidden:
            return self.finish('forbidden_contact')
        if not assessment['accepted']:
            return self.finish('original_convergence_rejected')
        if touched:
            return self.finish('seed_connector_touched_target')
        self.connected = True
        self.anchor_arrival = self.group.world.current_time
        self.state = 'settling_anchor'

    def finish(self, reason=None):
        if self.bridge.active:
            raise RuntimeError('Never finalize a seed attempt while execution is active')
        audit = audit_episode(self.episode, self.group.active_seed, self.group.active_trial['policy'])
        # Shared quality hashes every *_optimization.json, thereby binding the
        # source hashes, actual connector and join receipt into the admission.
        write(self.episode/'seed_augmentation_optimization.json', audit)
        return super().finish(reason)


class SeedGroup(Group):
    def __init__(self, campaign, launch):
        super().__init__(campaign, launch)
        self.envs = [SeedEnvironment(i, env.bridge, self) for i, env in enumerate(self.envs)]
        if len(self.envs) != 1:
            raise ValueError('The feasibility pilot uses exactly one isolated ARX environment')

    def sample_handoff(self, task):
        return {'world': self.active_trial['entry_world'], 'distance_mm': 100, 'sample': 0,
                'seed': self.registration['seed'], 'source_seed_id': self.active_seed['seed_id'],
                'policy': self.active_trial['policy'], 'perturbation': self.active_trial['perturbation'],
                'formula': 'Registered perturbation around actual source approach entry; source G unchanged'}

    def run_seed_trial(self, asset):
        started = time.monotonic(); env = self.envs[0]; env.connected = False; env.anchor_ready = False
        self.block_dispatches = 0
        block = self.out/'blocks'/('00000__'+asset['object_id']); block.mkdir(parents=True)
        write(block/'collection_placement.json', asset['collection_placement'])
        env.block_count = 0; env.reset(asset)
        for _ in range(15):
            self.step(False)
        check = fingerprint(env.bridge, read(self.registration['reference_files'][self.renderer]))
        write(block/'frozen_0.json', check)
        if not check['pass']:
            raise RuntimeError('Frozen physical configuration mismatch in seed trial')
        env.state = 'idle'
        verify_files(self.active_seed['files'])
        goal = copy.deepcopy(self.active_seed['goal'])
        goal.update(placement_id=self.active_seed['placement']['placement_id'],
                    seed_augmentation={'seed_id': self.active_seed['seed_id'], 'source_files': self.active_seed['files'],
                    'policy': self.active_trial['policy'], 'same_G_and_object_placement': True})
        scene_path = block/'qualification_scene.json'; write(scene_path, env.bridge.export_scene())
        job = self.planner.submit('qualify_pool', pool={'candidates': [goal], 'candidate_source': 'Previously executed successful seed, not new neural inference'},
            snapshot=env.bridge.snapshot(), scene=str(scene_path), visual_context=env.visual_context())
        wait_started = time.monotonic()
        while True:
            reply = self.planner.poll(job)
            if reply is not None:
                break
            self.step(False)
            if self.stop or (self.campaign/'STOP').exists() or time.monotonic()-wait_started > 900:
                raise RuntimeError('Seed qualification stopped or exceeded its resource/time budget')
        if not reply['ok']:
            raise RuntimeError(reply['error'])
        pool = reply['result']
        unchanged = [g for g in pool['candidates'] if np.allclose(g['grasp_world'], goal['grasp_world'], atol=1e-10, rtol=0)]
        pool['candidates'] = unchanged
        write(block/'qualified_pool.json', pool)
        self.pool = self.ledger.add_goals(asset['object_id'], pool)
        if not self.pool['candidates']:
            return {'status': 'seed_goal_rejected', 'object_id': asset['object_id'], 'dispatches': 0, 'idle': True,
                    'wall_seconds': time.monotonic()-started}
        while self.app.is_running():
            if time.time() >= self.launch['stop_unix'] or (self.campaign/'STOP').exists():
                self.stop = True
            space(self.out); self.writer.check(); env.tick()
            if env.task is None and (self.stop or self.block_dispatches >= 1 or env.block_count >= 1):
                break
            self.step()
            write(self.out/'heartbeat.json', {'wall': time.time(), 'trial_id': self.active_trial['trial_id'],
                'object_id': asset['object_id'], 'state': env.state, 'active': env.task['id'] if env.task else None})
        self.writer.drain()
        if env.task or env.bridge.active:
            raise RuntimeError('Seed trial did not end with confirmed idle')
        return {'status': 'trial_complete', 'object_id': asset['object_id'], 'dispatches': self.block_dispatches,
                'idle': True, 'wall_seconds': time.monotonic()-started}

    def run(self):
        base = self.out; results = []; self.ledger.close()
        try:
            for ident in self.launch['trial_ids']:
                if self.stop or (self.campaign/'STOP').exists() or time.time() >= self.launch['stop_unix']:
                    break
                trial = next(t for t in self.registration['seed_trials'] if t['trial_id'] == ident)
                self.active_trial = trial; self.active_seed = self.registration['seeds'][trial['seed_id']]
                prior = [r for r in self.registration.get('inherited_control_results',[])+results
                         if r['seed_id'] == trial['seed_id'] and r['policy'] == 'control']
                if trial['policy'] != 'control' and not controls_ready(trial['seed_id'],prior):
                    receipt = {**trial, 'completed': False, 'admitted': False, 'success': False,
                               'physical_dispatches': 0, 'status': 'not_activated_seed_control_failed'}
                else:
                    if not self.app.is_running():
                        raise RuntimeError('ARX application closed before a registered seed trial')
                    self.out = base/'trials'/ident; self.out.mkdir(parents=True, exist_ok=False)
                    self.ledger = Ledger(self.campaign/'trial_ledgers'/(ident+'.sqlite3'))
                    asset = self.ledger.acquire_object(self.owner, object_id=trial['object_id'], renderer=self.renderer)
                    if asset is None:
                        raise RuntimeError('Seed trial did not obtain its exact registered placement')
                    began = time.time(); result = self.run_seed_trial(asset)
                    rows = self.ledger.rows()
                    if len(rows) > 1:
                        raise RuntimeError('Seed feasibility trial dispatched more than one attempt')
                    evidence = [read(Path(row['episode'])/'physical_receipt.json') for row in rows]
                    receipt = {**trial, 'completed': True, 'started_unix': began, 'finished_unix': time.time(),
                        'success': bool(evidence and evidence[0]['pass']),
                        'admitted': bool(rows and rows[0]['state'] == 'committed'),
                        'physical_dispatches': sum(row['physical'] for row in rows),
                        'outcome': rows[0]['outcome'] if rows else result['status'],
                        'attempts': rows, 'physical_evidence': evidence, 'block': result}
                    write(self.out/'trial_result.json', receipt)
                    self.ledger.release_object(self.owner); self.ledger.close()
                results.append(receipt)
                write(base/'seed_results.json', {'status': 'running', 'trials': results})
                print('SEED_TRIAL', ident, receipt.get('outcome', receipt.get('status')), receipt['admitted'], flush=True)
            write(base/'seed_results.json', {'status': 'complete' if len(results) == len(self.launch['trial_ids']) else 'stopped', 'trials': results})
        except BaseException:
            write(base/'FAULT.json', {'wall': time.time(), 'error': traceback.format_exc(),
                                     'completed_trial_ids': [r['trial_id'] for r in results]})
            raise
        finally:
            self.writer.close(); self.ledger.close(); self.diagnostics.close()
            idle = not any(env.bridge.active for env in self.envs)
            write(base/'shutdown_receipt.json', {'wall': time.time(), 'all_trajectories_idle': idle,
                  'raw_writer_drained_and_closed': True, 'ledger_closed': True})
            self.app.close(wait_for_replicator=False, skip_cleanup=True, exit_code=1 if (base/'FAULT.json').exists() else 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--campaign', required=True); parser.add_argument('--launch', required=True)
    args = parser.parse_args(); SeedGroup(args.campaign, args.launch).run()
