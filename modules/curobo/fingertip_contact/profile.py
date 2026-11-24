"""Validate the shared planning/execution contract without importing a GPU SDK."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from arx_project.layout import ROOT

POLICY_FILE = ROOT / 'configs/policies/fingertip_table.json'


@dataclass(frozen=True)
class ContactProfile:
    profile_id: str = 'arx_871_distal_table_contact_v1'
    enabled: bool = True
    collision_profile_id: str = 'arx_refined_871_v1'
    maximum_penetration_m: float = .002
    distal_length_m: float = .010
    release_depth_noise_m: float = .0001
    minimum_normal_z: float = .95
    surface_position_noise_m: float = .0001
    allowed_stages: tuple = ('final_approach', 'closure')
    release_policy: str = 'nonincreasing_existing_contact'
    ik_fallback_seeds: int = 32
    ik_return_seeds: int = 8
    position_tolerance_m: float = .002
    rotation_tolerance_rad: float = .03
    actual_geometry_resource: str = 'arx_robot'

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('Contact policy enable flag must be boolean')
        object.__setattr__(self, 'allowed_stages', tuple(self.allowed_stages))
        values = [self.maximum_penetration_m, self.distal_length_m, self.release_depth_noise_m,
                  self.minimum_normal_z, self.surface_position_noise_m,
                  self.position_tolerance_m, self.rotation_tolerance_rad]
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite contact policy')
        if self.collision_profile_id != 'arx_refined_871_v1':
            raise ValueError('The contact policy requires the current 871-sphere model')
        if not (0 <= self.maximum_penetration_m <= .002 and 0 < self.distal_length_m <= .010):
            raise ValueError('Only the distal 10mm and up to 2mm contact are registered')
        if not (0 <= self.release_depth_noise_m <= .0001 and 0 <= self.surface_position_noise_m <= .0001):
            raise ValueError('Excessive contact measurement noise allowance')
        if not .95 <= self.minimum_normal_z <= 1.:
            raise ValueError('Table contact must have a vertical normal')
        if tuple(self.allowed_stages) != ('final_approach', 'closure') or self.release_policy != 'nonincreasing_existing_contact':
            raise ValueError('Unregistered contact stage or release policy')
        if (self.ik_fallback_seeds, self.ik_return_seeds) != (32, 8):
            raise ValueError('Unregistered IK budget')
        if self.position_tolerance_m != .002 or self.rotation_tolerance_rad != .03:
            raise ValueError('The original endpoint tolerances must remain unchanged')


def load_profile(path=POLICY_FILE):
    return ContactProfile(**json.loads(Path(path).read_text()))
