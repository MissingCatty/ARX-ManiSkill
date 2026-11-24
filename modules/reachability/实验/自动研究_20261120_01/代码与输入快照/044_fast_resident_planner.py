"""Refresh measured prismatic lock state without rebuilding the ARX robot model."""
from copy import deepcopy
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch
import xml.etree.ElementTree as ET

import torch
import yaml

import resident_planner as original


class FastPlannerPool(original.PlannerPool):
    def __init__(self, root, warmup_trial):
        super().__init__(root, warmup_trial)
        self.views = {}
        self.yaml_cache = {}
        reports = {}
        for role, context in self.contexts.items():
            cfg = yaml.safe_load(context.model.read_text())
            urdf = Path(cfg['robot_cfg']['kinematics']['urdf_path'])
            joints = {j.get('name'): j for j in ET.parse(urdf).getroot().findall('joint')}
            names = original.frozen.FINGERS
            assert all(joints[n].get('type') == 'prismatic' for n in names)
            upper = [float(joints[n].find('limit').get('upper')) for n in names]
            assert set(cfg['robot_cfg']['kinematics']['lock_joints']) == set(names)

            def generate(values):
                candidate = deepcopy(cfg['robot_cfg'])
                candidate['kinematics']['lock_joints'] = dict(zip(names, values))
                return original.RobotCfg.create(candidate, context.planner.device_cfg).kinematics.kinematics_config

            started = time.perf_counter()
            zero = generate([0., 0.])
            anchors = [generate([upper[0], 0.]), generate([0., upper[1]])]
            dynamic_fields = ['fixed_transforms', 'link_spheres', 'reference_link_spheres']
            slopes = {field: [(getattr(a, field)-getattr(zero, field))/u for a, u in zip(anchors, upper)]
                      for field in dynamic_fields if getattr(zero, field) is not None}
            # Prismatic locks change translations, never authored rotations or radii.
            for delta in slopes['fixed_transforms']:
                assert bool((delta[..., :3] == 0).all())
            for field in ['link_spheres', 'reference_link_spheres']:
                for delta in slopes.get(field, []):
                    assert bool((delta[..., 3] == 0).all())
            scratch = deepcopy(zero)
            probe_values = [upper[0] * .31, upper[1] * .77]
            probe = generate(probe_values)
            errors = {}
            for field, delta in slopes.items():
                expected = getattr(zero, field) + probe_values[0]*delta[0] + probe_values[1]*delta[1]
                error = float((expected-getattr(probe, field)).abs().max())
                assert error < 2e-7, (role, field, error)
                errors[field] = error
            self.views[role] = {'cfg': cfg['robot_cfg'], 'zero': zero, 'scratch': scratch,
                                'slopes': slopes, 'names': names}
            reports[role] = {'startup_parameter_constructions': 4,
                'startup_state_refresh_preparation_seconds': time.perf_counter()-started,
                'prismatic_joint_names': names, 'source_urdf': str(urdf),
                'probe_max_errors': errors, 'authored_rotations_and_sphere_radii_unchanged': True,
                'runtime_robot_configuration_rebuilds': 0}
        for kind in ['model', 'model_closed']:
            path = original.frozen.ARX / 'scripts/curobo_arx/static_pick/reference' / kind / 'arx.yml'
            text = path.read_text()
            self.yaml_cache[text] = yaml.safe_load(text)
        self.state_refresh_proof = reports
        original.save(root / 'state_refresh_validation.json', reports)

    def _measured_robot_view(self, cfg, device):
        urdf = cfg['kinematics']['urdf_path']
        role = 'closed' if Path(urdf).parent.name == 'model_closed' else 'open'
        view = self.views[role]
        expected = deepcopy(view['cfg'])
        expected['kinematics']['lock_joints'] = deepcopy(cfg['kinematics']['lock_joints'])
        assert expected == cfg, 'A frozen robot property other than measured lock state changed'
        values = [float(cfg['kinematics']['lock_joints'][name]) for name in view['names']]
        scratch = view['scratch']
        for field, slopes in view['slopes'].items():
            getattr(scratch, field).copy_(getattr(view['zero'], field) + values[0]*slopes[0] + values[1]*slopes[1])
        locked = scratch.lock_jointstate
        position = torch.tensor([cfg['kinematics']['lock_joints'][name] for name in locked.joint_names],
                                device=locked.position.device, dtype=locked.position.dtype)
        locked.position.copy_(position.reshape_as(locked.position))
        return SimpleNamespace(kinematics=SimpleNamespace(kinematics_config=scratch))

    def bind(self, directory, state, closed=False, include_target=False):
        old_load = yaml.safe_load

        def cached_yaml(text):
            if isinstance(text, str) and text in self.yaml_cache:
                return deepcopy(self.yaml_cache[text])
            return old_load(text)

        started = time.perf_counter()
        with patch.object(original, 'RobotCfg', SimpleNamespace(create=self._measured_robot_view)), patch.object(yaml, 'safe_load', cached_yaml):
            context = super().bind(directory, state, closed=closed, include_target=include_target)
        receipt = __import__('json').loads((directory / 'resident_rebind.json').read_text())
        receipt.update(runtime_robot_configuration_rebuilds=0,
            measured_prismatic_state_refresh='Exact affine state dependence derived from the unchanged URDF factory; source rotations, radii, limits and collision rules retained',
            fast_bind_wall_seconds=time.perf_counter()-started)
        original.save(directory / 'resident_rebind.json', receipt)
        return context
