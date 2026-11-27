"""Bounded cache of exact native sweep FK; collision costs are never cached."""
from collections import OrderedDict
import hashlib
import json

import numpy as np


def geometry_key(config,q,fractions,target,observed,device):
    q = np.asarray(q)
    digest = hashlib.sha256(json.dumps([config,float(target),observed,str(device)],
                                     sort_keys=True,allow_nan=False).encode())
    for array in [q,np.asarray(fractions)]:
        digest.update(str((array.shape,array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class SphereMemo:
    def __init__(self,max_bytes=64*1024**2):
        if max_bytes < 1:
            raise ValueError('Sphere cache must have a positive memory bound')
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0
        self.reset()

    def reset(self):
        self.hits = self.misses = self.evictions = 0

    def get(self,key,builder):
        if key in self.entries:
            self.hits += 1
            value = self.entries.pop(key)
            self.entries[key] = value
            return value,True
        self.misses += 1
        value = builder()
        size = value.numel()*value.element_size()
        while self.entries and self.bytes+size > self.max_bytes:
            _,old = self.entries.popitem(last=False)
            self.bytes -= old.numel()*old.element_size()
            self.evictions += 1
        if size <= self.max_bytes:
            self.entries[key] = value
            self.bytes += size
        return value,False

    def report(self):
        return {name:getattr(self,name) for name in ['hits','misses','evictions','bytes','max_bytes']}


def attach(pool):
    from .cached_sweep import native_opening_spheres
    if hasattr(pool,'sphere_memo'):
        raise RuntimeError('Sphere memo already attached')
    pool.sphere_memo = SphereMemo()
    pool.sphere_reuse_enabled = False
    pool.fast_sweep_enabled = False

    def make(context,config,q,fractions,target,observed):
        def build():
            if pool.fast_sweep_enabled:
                from .direct_sweep import native_opening_spheres as direct
                return direct(pool,context,config,q,fractions,target,observed)
            return native_opening_spheres(pool,context,config,q,fractions,target,observed)
        if not pool.sphere_reuse_enabled:
            return build()
        key = geometry_key(config,q.detach().cpu().numpy(),fractions,target,observed,q.device)
        batch,hit = pool.sphere_memo.get(key,build)
        if hit:
            # Match the original builder's final native lock state as well.
            for name,value in observed.items():
                config['kinematics']['lock_joints'][name] = float(value+fractions[-1]*(target-value))
            robot = pool._measured_robot_view(config,context.planner.device_cfg)
            context.checker.kinematics.kinematics_config.copy_(robot.kinematics.kinematics_config)
        return batch
    pool.make_opening_spheres = make
    return pool
