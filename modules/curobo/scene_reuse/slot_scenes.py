"""Update bounded native obstacle slots by logical role and verified geometry."""
from copy import deepcopy
import hashlib
import json

from modules.curobo.state_updates.scenes import separate


def logical_name(name):
    prefix, _, suffix = name.rpartition('_')
    return prefix if len(suffix) == 20 and all(c in '0123456789abcdef' for c in suffix) else name


class SlotSceneUpdates:
    def __init__(self, config_factory, pose_factory, digest):
        self.config_factory, self.pose_factory, self.digest = config_factory, pose_factory, digest
        self.records = {}; self.events = []; self.invalidations = 0; self.bounds = {}

    @staticmethod
    def checker(owner):
        value = getattr(owner, 'scene_collision_checker', None)
        return value if value is not None else owner.scene_model

    @staticmethod
    def fingerprint(checker):
        """Check mutable state at request boundaries, outside the binding loop."""
        import torch
        digest = hashlib.sha256()
        arrays = []
        for data in [checker.data.cuboids, checker.data.meshes]:
            if data is None: continue
            digest.update(json.dumps(data.names, sort_keys=True).encode())
            for name in ['dims','inv_pose','enable','count','mesh_ids','bounds_lower','bounds_upper']:
                tensor = getattr(data, name, None)
                if tensor is not None:
                    arrays.append(tensor.detach().contiguous().view(torch.uint8).reshape(-1))
            if hasattr(data, 'wp_cache'):
                loaded = {name: int(data.wp_cache[name].mesh_id) if name in data.wp_cache else None
                          for name in data.names[0] if name is not None}
                digest.update(json.dumps(loaded, sort_keys=True).encode())
        if arrays:
            digest.update(torch.cat(arrays).cpu().numpy().tobytes())
        return digest.hexdigest()

    def reset(self):
        self.events = []; self.invalidations = 0; self.bounds = {}
        for identity, record in list(self.records.items()):
            if record.get('fingerprint') != self.fingerprint(record['checker']):
                del self.records[identity]; self.invalidations += 1

    def shape(self, kind, name, spec):
        value = {k:v for k,v in spec.items() if k not in ['pose','file_path']}
        if kind == 'mesh':
            digest = self.digest(spec['file_path'])
            if logical_name(name) != name and name.rsplit('_',1)[1] != digest[:20]:
                raise ValueError('Qualified obstacle name and full content digest disagree')
            value['content_sha256'] = digest
        return json.dumps(value, sort_keys=True, allow_nan=False)

    def initialize(self, owner, checker, scene):
        owner.clear_scene_cache(); owner.update_world(self.config_factory(scene))
        rows = {'cuboid':{}, 'mesh':{}}
        for kind, values in scene.items():
            store = checker.data.cuboids if kind == 'cuboid' else checker.data.meshes
            for name, spec in values.items():
                logical = logical_name(name) if kind == 'mesh' else name
                if logical in rows[kind]: raise ValueError('Duplicate physical obstacle role')
                rows[kind][logical] = {'index':store.names[0].index(name), 'name':name,
                    'spec':deepcopy(spec), 'shape':self.shape(kind,name,spec), 'enabled':True}
        return {'checker':checker,'rows':rows}

    def replace_slot(self, store, kind, index, name, spec, shape, count):
        pose = self.pose_factory(spec['pose'])
        if kind == 'mesh':
            mesh = self.config_factory({'mesh':{name:spec}}).mesh[0]
            geometry = store._load_mesh_into_cache(mesh)
            store.update_from_warp_id(geometry.mesh_id, name, w_obj_pose=pose, mesh_idx=index)
            key = (shape, int(geometry.mesh_id))
            if key not in self.bounds: self.bounds[key] = geometry.get_bounds()
            lower, upper = self.bounds[key]
            store.dims[0,index,:3] = upper-lower
            store.bounds_lower[0,index] = lower; store.bounds_upper[0,index] = upper
        else:
            store.dims[0,index,:3] = store.device_cfg.to_device(spec['dims'])
            store.inv_pose[0,index,:7] = pose.inverse().get_pose_vector()
            store.names[0][index] = name; store.enable[0,index] = 1
        store.count[0] = count

    def update(self, owner, scene):
        separate(scene)
        scene = deepcopy(scene)
        qualified = {}
        for name, spec in scene.get('mesh',{}).items():
            digest = self.digest(spec['file_path'])
            logical = logical_name(name)
            if logical != name and name.rsplit('_',1)[1] != digest[:20]:
                raise ValueError('Qualified obstacle name and full content digest disagree')
            key = logical+'_'+digest[:20]
            if key in qualified: raise ValueError('Duplicate physical obstacle role')
            qualified[key] = spec
        if 'mesh' in scene: scene['mesh'] = qualified
        checker = self.checker(owner); identity = id(checker)
        old = self.records.pop(identity, None)
        if old is None:
            record = self.initialize(owner, checker, scene)
            event = {'operation':'replace_geometry','updated_obstacles':sum(len(v) for v in scene.values()),
                     'slot_replacements':0,'enable_flag_updates':0}
        else:
            record = old; changed = toggles = replacements = 0
            for kind in ['cuboid','mesh']:
                values = scene.get(kind,{})
                store = checker.data.cuboids if kind == 'cuboid' else checker.data.meshes
                if store is None:
                    if values: raise ValueError('Obstacle storage is not allocated')
                    continue
                desired = {(logical_name(n) if kind=='mesh' else n):(n,s) for n,s in values.items()}
                if len(desired) != len(values) or len(desired)>store.max_n:
                    raise ValueError('Duplicate obstacle roles or original slot capacity exceeded')
                rows = record['rows'][kind]
                for logical, (name,spec) in desired.items():
                    shape = self.shape(kind,name,spec); row = rows.get(logical)
                    if row is None:
                        used = {r['index'] for r in rows.values()}
                        available = [i for i in range(store.max_n) if i not in used]
                        if not available:
                            evict = next((n for n in rows if n not in desired),None)
                            if evict is None: raise ValueError('No free native obstacle slot')
                            available = [rows.pop(evict)['index']]
                        index = available[0]
                        row = {'index':index,'enabled':False}; rows[logical] = row
                    if row.get('shape') != shape or row.get('name') != name:
                        count = max(r['index'] for r in rows.values())+1
                        self.replace_slot(store,kind,row['index'],name,spec,shape,count)
                        row.update(name=name,shape=shape,enabled=True); replacements += 1
                    else:
                        if row['spec']['pose'] != spec['pose']:
                            checker.update_obstacle_pose(name,self.pose_factory(spec['pose'])); changed += 1
                        if not row['enabled']:
                            checker.enable_obstacle(name,True); row['enabled']=True; toggles += 1
                    row['spec'] = deepcopy(spec)
                for logical,row in rows.items():
                    if logical not in desired and row['enabled']:
                        checker.enable_obstacle(row['name'],False); row['enabled']=False; toggles += 1
            bank = {kind:{r['name']:r['spec'] for r in rows.values()} for kind,rows in record['rows'].items()}
            cfg = self.config_factory(bank)
            checker.scene_model = cfg; checker.data.scene_model = cfg
            event = {'operation':'replace_slots' if replacements else 'update_poses' if changed else 'update_enabled' if toggles else 'reuse_scene',
                     'updated_obstacles':changed,'slot_replacements':replacements,'enable_flag_updates':toggles}
        event.update(obstacle_count=sum(len(v) for v in scene.values()),
                     resident_obstacles=sum(len(v) for v in record['rows'].values()))
        self.records[identity] = record; self.events.append(event)
        return event

    def report(self):
        for record in self.records.values():
            record['fingerprint'] = self.fingerprint(record['checker'])
        return {'calls':len(self.events), 'geometry_replacements':sum(e['operation']=='replace_geometry' for e in self.events),
            'pose_updates':sum(e['updated_obstacles'] for e in self.events if e['operation']=='update_poses'),
            'unchanged_scene_reuses':sum(e['operation']=='reuse_scene' for e in self.events),
            'enable_flag_updates':sum(e['enable_flag_updates'] for e in self.events),
            'slot_replacements':sum(e['slot_replacements'] for e in self.events),
            'request_boundary_invalidations':self.invalidations,
            'original_mesh_capacity':8,'mutable_state_checked_at_request_boundaries':True}
