"""Change obstacle poses in fixed storage; replace geometry only when needed."""
from copy import deepcopy
import json


def separate(scene):
    topology, poses = {}, {}
    for kind, values in scene.items():
        if kind not in ['cuboid', 'mesh']:
            raise ValueError('Unexpected measured obstacle kind: ' + kind)
        topology[kind] = {}
        for name, geometry in values.items():
            poses[name] = list(geometry['pose'])
            topology[kind][name] = {k: v for k, v in geometry.items() if k != 'pose'}
    json.dumps(poses, allow_nan=False)
    return json.dumps(topology, sort_keys=True, allow_nan=False), poses


class SceneUpdates:
    def __init__(self, config_factory, pose_factory):
        self.config_factory, self.pose_factory = config_factory, pose_factory
        self.reset()

    def reset(self):
        self.records = {}
        self.events = []

    def update(self, owner, scene):
        checker = getattr(owner, 'scene_collision_checker', None)
        if checker is None:
            checker = owner.scene_model
        identity = id(checker)
        key, poses = separate(scene)
        previous = self.records.get(identity)
        # Invalidate before a mutation: a failed update can never become a hit.
        self.records.pop(identity, None)
        if previous is None or previous['topology'] != key:
            owner.clear_scene_cache()
            owner.update_world(self.config_factory(scene))
            mode, count = 'replace_geometry', len(poses)
        else:
            changed = {n: p for n, p in poses.items() if previous['poses'][n] != p}
            for name, pose in changed.items():
                checker.update_obstacle_pose(name, self.pose_factory(pose))
            mode, count = ('update_poses' if changed else 'reuse_scene'), len(changed)
        self.records[identity] = {'topology': key, 'poses': poses}
        event = {'operation': mode, 'updated_obstacles': count, 'obstacle_count': len(poses)}
        self.events.append(event)
        return event

    def report(self):
        return {'calls': len(self.events), 'geometry_replacements': sum(
            e['operation'] == 'replace_geometry' for e in self.events), 'pose_updates': sum(
            e['updated_obstacles'] for e in self.events if e['operation'] == 'update_poses'),
            'unchanged_scene_reuses': sum(e['operation'] == 'reuse_scene' for e in self.events)}


class UnionSceneUpdates(SceneUpdates):
    """Keep a bounded union of geometries; switch exact collision enable flags."""
    def update(self, owner, scene):
        checker = getattr(owner, 'scene_collision_checker', None)
        if checker is None:
            checker = owner.scene_model
        identity = id(checker)
        separate(scene)
        old = self.records.pop(identity, None)
        bank = deepcopy(old['bank']) if old else {}
        changed_geometry = old is None
        for kind, values in scene.items():
            for name, geometry in values.items():
                previous = bank.get(kind, {}).get(name)
                if previous is None:
                    changed_geometry = True
                elif {k: v for k, v in previous.items() if k != 'pose'} != {
                        k: v for k, v in geometry.items() if k != 'pose'}:
                    # A same-name size/file change invalidates the whole bank.
                    bank = {}
                    changed_geometry = True
                    break
        for kind, values in scene.items():
            bank.setdefault(kind, {}).update(deepcopy(values))
        if any(len(v) > 8 for v in bank.values()):
            bank = deepcopy(scene)
            changed_geometry = True
        active = {n for values in scene.values() for n in values}
        bank_poses = {n: g['pose'] for values in bank.values() for n, g in values.items()}
        count, toggles = 0, 0
        if changed_geometry:
            owner.clear_scene_cache()
            owner.update_world(self.config_factory(bank))
            mode, count = 'replace_geometry', len(bank_poses)
            enabled = set(bank_poses)
        else:
            before = {n: g['pose'] for values in old['bank'].values() for n, g in values.items()}
            for name in active:
                if before[name] != bank_poses[name]:
                    checker.update_obstacle_pose(name, self.pose_factory(bank_poses[name]))
                    count += 1
            enabled = old['active']
            mode = 'update_poses' if count else 'reuse_scene'
        for name in set(bank_poses):
            if (name in enabled) != (name in active):
                checker.enable_obstacle(name, name in active)
                toggles += 1
        if mode == 'reuse_scene' and toggles:
            mode = 'update_enabled'
        self.records[identity] = {'bank': bank, 'active': active}
        event = {'operation': mode, 'updated_obstacles': count, 'obstacle_count': len(active),
                 'resident_obstacles': len(bank_poses), 'enable_flag_updates': toggles}
        self.events.append(event)
        return event

    def report(self):
        return {**super().report(),
                'enable_flag_updates': sum(e['enable_flag_updates'] for e in self.events)}
