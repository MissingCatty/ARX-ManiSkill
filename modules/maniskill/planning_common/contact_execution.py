"""Execution and independent readback for the registered distal contact policy."""
import json
from pathlib import Path

import numpy as np

from arx_project.layout import ROOT, read_json, write_json
from modules.curobo.fingertip_contact.physical_contact import classify_contact, ReleaseEnvelope
from modules.curobo.fingertip_contact.geometry import FINGERS
from modules.arx_core.arms import get_arm


def execution_class(parent, profile, snapshot):
    class ContactExecution(parent):
        def __init__(self, env, out):
            self.profile = profile
            self.release = ReleaseEnvelope(profile)
            self.frame_depths = []
            self.original_forbidden = []
            self.tip_contacts = []
            self.frame_eef = None
            calibration = read_json(ROOT/'simulation/robots/arx/runtime/action_tcp.json')
            arm = get_arm(getattr(env, 'grasp_arm', 'right'))
            self.tip_planes = dict(zip(arm.finger_links,
                [calibration['authored_tip_axial_position_eef_z_m'][name] for name in FINGERS]))
            self.table_box = snapshot(env)['scene_cuboids']['table']
            self.table_entity = env.table._objs[0]
            self.finger_entities = {env.agent.robot.links_map[name]._objs[0].entity: name for name in arm.finger_links}
            super().__init__(env, out)
            mapping = {'target': env.obj._objs[0].name, 'table': self.table_entity.name,
                       'fingers': {entity.name: name for entity, name in self.finger_entities.items()},
                       'table_geometry': self.table_box, 'tip_planes': self.tip_planes,
                       'profile_id': profile.profile_id, 'arm': arm.name}
            write_json(self.out/'contact_entities.json', mapping)

        def contact_event(self, robot_link, other, contact, original_allowed):
            points = [{'world': np.asarray(p.position).tolist(), 'normal': np.asarray(p.normal).tolist(),
                       'separation_m': float(p.separation), 'impulse_ns': float(np.linalg.norm(p.impulse))}
                      for p in contact.points]
            for point in points:
                if not np.isfinite([*point['world'], *point['normal'], point['separation_m'], point['impulse_ns']]).all():
                    raise ValueError('Nonfinite measured contact')
            finger = self.finger_entities.get(robot_link)
            decision = classify_contact(self.profile, finger=finger, is_table=other == self.table_entity,
                stage=self.label, points=points, eef=self.frame_eef, table=self.table_box,
                tip_plane=self.tip_planes.get(finger, 0.), arm=self.arm.name)
            local = decision['allowed']
            depth = decision['contact_depth_m']
            if local:
                self.frame_depths.append(depth)
                if self.label == 'lift' and (self.release.ceiling is None or
                        depth > self.release.ceiling + self.profile.release_depth_noise_m):
                    decision.update(allowed=False, reason='contact_depth_increased_during_release')
            allowed = bool(original_allowed or decision['allowed'])
            event = {'allowed': allowed, 'original_allowed': original_allowed,
                     'robot_link': finger or robot_link.name, 'other_name': other.name,
                     'tip_table_contact': local, 'points': points,
                     'policy_reason': 'original_target_contact' if original_allowed else decision['reason'],
                     'contact_depth_m': depth, 'release_ceiling_m': self.release.ceiling}
            if not original_allowed:
                self.original_forbidden.append({'tick': self.tick, 'stage': self.label, **event})
            if local and allowed:
                self.tip_contacts.append({'tick': self.tick, 'stage': self.label, **event})
            return event

        def record(self):
            self.frame_depths = []
            self.frame_eef = self.env.agent.tcp.pose.to_transformation_matrix().cpu().numpy()[0]
            super().record()
            depth = max(self.frame_depths, default=0.)
            if self.label == 'close':
                self.release.anchor(depth)
            elif self.label == 'lift':
                self.release.accept(depth)
            self.env.fingertip_contact_state = {'profile_id': self.profile.profile_id,
                'stage': self.label, 'tick': self.tick, 'depth_m': depth,
                'qpos': self.env.agent.robot.qpos.cpu().numpy()[0].tolist(),
                'valid': not self.forbidden, 'arm': self.arm.name}

        def extra_trace_fields(self):
            return {'eef_world': self.frame_eef.tolist(), 'execution_arm': self.arm.name,
                    'contact_policy_id': self.profile.profile_id,
                    'release_ceiling_m': self.release.ceiling}

        def segment(self, label, *args, **kwargs):
            if label == 'lift' and self.release.ceiling is None:
                raise RuntimeError('Missing measured closure contact anchor')
            return super().segment(label, *args, **kwargs)

        def finish(self):
            write_json(self.out/'original_forbidden_contacts.json', self.original_forbidden)
            write_json(self.out/'allowed_tip_table_contacts.json', self.tip_contacts)
            super().finish()
    return ContactExecution


def verify_trial(directory, profile):
    """Recompute contact decisions and grasp gates from recorded raw points."""
    directory = Path(directory)
    result = read_json(directory/'result.json')
    if not (directory/'execution.jsonl').exists():
        if result['success']:
            raise RuntimeError('Reported success without an execution trace')
        write_json(directory/'independent_evaluation.json', {'passed': True,
            'independently_successful': False, 'physical_attempt': False})
        return
    entities = read_json(directory/'contact_entities.json')
    if entities['profile_id'] != profile.profile_id:
        raise ValueError('Execution contact profile changed')
    if result.get('selected_arm') and entities.get('arm') != result['selected_arm']:
        raise ValueError('Execution contact entities belong to the other arm')
    box = entities['table_geometry']; top = box['center'][2]+box['dims'][2]/2
    center, half = np.asarray(box['center']), np.asarray(box['dims'])/2
    origin = read_json(directory/'execution_origin.json')['object_initial_z_m']
    bad = strict_bad = tip_events = 0
    max_depth = 0.; release = None; bilateral = set(); holds = []
    for line in (directory/'execution.jsonl').open():
        row = json.loads(line); stage = row['stage']; eef = np.asarray(row['eef_world'])
        if result.get('selected_arm') and row.get('execution_arm') != result['selected_arm']:
            raise ValueError('Trace arm changed after selection')
        if eef.shape != (4, 4) or not np.isfinite(eef).all():
            raise ValueError('Invalid recorded tool pose')
        depths = []
        for event in row['contacts']:
            name = event['a'] if event['a'] in entities['fingers'] else event['b']
            finger = entities['fingers'].get(name)
            other = event['b'] if name == event['a'] else event['a']
            original = bool(finger and other == entities['target'] and stage != 'observe')
            if original and stage == 'close':
                bilateral.add(finger)
            allowed = original
            if finger and other == entities['table'] and stage in ['approach', 'close', 'lift']:
                points = event['points']
                if any(not np.isfinite([*p['world'], *p['normal'], p['separation_m'], p['impulse_ns']]).all() for p in points):
                    raise ValueError('Nonfinite recorded contact')
                points = [p for p in points if p['impulse_ns'] > 1e-8]
                depth = max([max(0., -p['separation_m']) for p in points], default=0.)
                plane = entities['tip_planes'][finger]
                local = [(np.asarray(p['world'])-eef[:3, 3]) @ eef[:3, :3] for p in points]
                allowed = bool(points) and depth <= profile.maximum_penetration_m
                allowed &= all(v[2] >= plane-profile.distal_length_m-1e-6 and
                               v[2] <= plane+profile.maximum_penetration_m for v in local)
                allowed &= all(abs(np.linalg.norm(p['normal'])-1.) <= 1e-4 and
                    abs(p['normal'][2]) >= profile.minimum_normal_z and
                    np.all(np.abs(np.asarray(p['world'])[:2]-center[:2]) <= half[:2]) and
                    abs(p['world'][2]-top) <= profile.maximum_penetration_m+.001+profile.surface_position_noise_m
                    for p in points)
                if allowed:
                    depths.append(depth)
                    if stage == 'lift' and (release is None or depth > release+profile.release_depth_noise_m):
                        allowed = False
                    if allowed:
                        tip_events += 1; max_depth = max(max_depth, depth)
            strict_bad += int(not original); bad += int(not allowed)
            if bool(allowed) != event['allowed'] or original != event['original_allowed']:
                raise RuntimeError('Contact decision disagrees with independent geometry readback')
        depth = max(depths, default=0.)
        if stage == 'close':
            release = depth
        elif stage == 'lift' and release is not None and depth <= release+profile.release_depth_noise_m:
            release = min(release, depth)
        if stage == 'hold':
            holds.append(float(row['object_pose'][2])-origin)
    stages = result.get('stages', {})
    settled = all(name in stages and stages[name]['settled'] and stages[name]['joint_error_rad'] < .005
                  for name in ['pregrasp', 'approach', 'close', 'lift', 'hold'])
    success = bool(holds and min(holds) >= .01 and len(holds)/120 >= 3 and settled and bad == 0 and len(bilateral) == 2)
    if result.get('selected_arm'):
        success &= bool(stages.get('open', {}).get('finger_opening_passed'))
    if success != result['success']:
        raise RuntimeError('Independent physical success disagrees')
    result.update(success_v1=bool(success and strict_bad == 0), success_v2=success,
                  contact_policy_id=profile.profile_id)
    write_json(directory/'result.json', result)
    write_json(directory/'independent_evaluation.json', {'passed': True,
        'independently_successful': success, 'success_v1': result['success_v1'], 'success_v2': success,
        'minimum_lift_m': min(holds) if holds else None, 'hold_seconds': len(holds)/120,
        'bilateral': sorted(bilateral), 'forbidden': bad, 'strict_forbidden': strict_bad,
        'allowed_tip_table_events': tip_events, 'maximum_allowed_depth_m': max_depth,
        'raw_contact_points_recomputed': True, 'all_required_stages_settled': settled})


def install(run, baseline, profile):
    if getattr(run.Execution, 'contact_profile_id', None) == profile.profile_id:
        return
    snapshot = run.snapshot
    run.Execution = execution_class(run.Execution, profile, snapshot)
    run.Execution.contact_profile_id = profile.profile_id

    def measured_snapshot(env, obs=None):
        state = snapshot(env, obs)
        if hasattr(env, 'fingertip_contact_state'):
            state['fingertip_table_contact'] = dict(env.fingertip_contact_state)
        return state
    run.snapshot = measured_snapshot
    baseline.verify_trial = lambda directory: verify_trial(directory, profile)
