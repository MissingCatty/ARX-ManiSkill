"""Cheap sufficient collision proof for a specific FK witness, not a new rule.

Only reject vertices definitely inside the unchanged table. Inconclusive meshes
continue to the original native sphere and exact local-contact validators.
"""
from copy import deepcopy

import numpy as np

from modules.arx_core.arms import get_arm


class HandTablePrecheck:
    def __init__(self, model, state, maximum_tip_depth_m):
        self.local = {}
        table = state['scene_cuboids']['table']
        self.center = np.asarray(table['center'], dtype=float)
        self.half = np.asarray(table['dims'], dtype=float)/2
        self.maximum_tip_depth_m = maximum_tip_depth_m
        opened = deepcopy(state)
        opened['links'] = {'base_link': state['links']['base_link']}
        for side in ['right', 'left']:
            for joint in get_arm(side).finger_joints:
                opened['qpos'][state['joint_names'].index(joint)] = .044
        frames = model.frames(opened)
        for side in ['right', 'left']:
            arm = get_arm(side)
            inv = np.linalg.inv(frames[arm.eef])
            parts = []
            for link in arm.links[-3:]:
                transform = inv@frames[link]
                for vertices, _, _ in model.parts(link):
                    local = vertices@transform[:3, :3].T+transform[:3, 3]
                    parts.append((link, local, link in arm.finger_links))
            self.local[side] = parts

    def check(self, proposal):
        actual = np.asarray(proposal['witness']['achieved_eef_world'])
        side = proposal['arm']
        if not hasattr(self, '_bounds'):
            self._bounds = {}
        key = (side, self.maximum_tip_depth_m)
        if key not in self._bounds:
            self._bounds[key] = [(link, points, (points.min(0)+points.max(0))/2,
                (points.max(0)-points.min(0))/2, self.maximum_tip_depth_m if finger else 0.)
                for link, points, finger in self.local[side]]
        lo, hi = self.center-self.half, self.center+self.half
        rotation, translation = actual[:3, :3], actual[:3, 3]
        for link, local, center, half, allowance in self._bounds[key]:
            world_center = rotation@center+translation
            world_half = np.abs(rotation)@half
            # Conservative box rejection. Close boundaries go to exact vertices.
            if ((world_center-world_half >= hi-1e-7+1e-12).any() or
                    (world_center+world_half <= lo+1e-7-1e-12).any() or
                    world_center[2]-world_half[2] >= hi[2]-allowance-1e-7+1e-12):
                continue
            z = local@rotation[2]+translation[2]
            eligible = np.flatnonzero((z > lo[2]+1e-7) & (z < hi[2]-allowance-1e-7))
            if not len(eligible):
                continue
            lowest = eligible[np.argmin(z[eligible])]
            xy = rotation[:2]@local[lowest]+translation[:2]
            if (xy > lo[:2]+1e-7).all() and (xy < hi[:2]-1e-7).all():
                depth = float(hi[2]-z[lowest])
            else:
                # A table-edge case needs only the potentially deep vertices.
                xy = local[eligible]@rotation[:2].T+translation[:2]
                inside = ((xy > lo[:2]+1e-7) & (xy < hi[:2]-1e-7)).all(axis=1)
                if not inside.any():
                    continue
                depth = float((hi[2]-z[eligible[inside]]).max())
            return {'passed': False, 'reason': 'known_hand_vertex_inside_table',
                    'link': link, 'depth_m': depth, 'allowance_m': allowance,
                    'new_IK_calls': 0, 'native_gate_equivalent_rejection': True}
        return {'passed': True, 'reason': 'inconclusive_continue_original_native_gate', 'new_IK_calls': 0}
