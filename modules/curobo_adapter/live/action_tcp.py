"""Approved robot action frame for independent planning diagnostics."""
from pathlib import Path
import numpy as np
from arx_project.layout import ROOT, module_dir, read_json as read, sha256 as sha
from modules.arx_core.scripts.curobo_arx.action_frame import EEF_TCP, action_frame, target_pose


class ActionTCP:
    def __init__(self, config=None):
        if config is None:
            path = ROOT/'simulation/robots/arx/runtime/action_tcp.json'
            config = {'action_tcp_profile': str(path), 'action_tcp_profile_sha256': sha(path)}
        path = Path(config['action_tcp_profile']).expanduser()
        self.path = path if path.is_absolute() else module_dir('curobo_adapter')/path
        self.digest = sha(self.path)
        if self.digest != config['action_tcp_profile_sha256']:
            raise RuntimeError('Action TCP profile hash mismatch')
        profile = read(self.path)
        self.frame = action_frame(profile['T_eef_action'])
        if (profile['profile_id'] != 'arx_action_tcp_43mm_v1'
                or profile['relative_to'] != 'right_eef_link'
                or not np.array_equal(self.frame, EEF_TCP)):
            raise ValueError('Independent route requires the approved 43 mm action TCP')
        self.profile_id = profile['profile_id']
        self.frame.setflags(write=False)

    def metadata(self):
        return {'profile_id': self.profile_id, 'profile_path': str(self.path),
                'profile_sha256': self.digest, 'T_eef_action': self.frame.tolist()}

    def target_pose(self, ee_world, action):
        return target_pose(ee_world, action, eef_tcp=self.frame)
