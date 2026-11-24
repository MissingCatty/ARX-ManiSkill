"""Frozen shared inference plus the continuous MPC pilot, loaded once per batch."""
import json
from pathlib import Path
import shutil
import sys

from arx_quiet_diagnostics_20261119 import install as quiet
quiet()
import arx_runtime_imports
arx_runtime_imports.install()
import graspcl_pipeline as common
import short_lift_services as services
import graspcl_unseen_services as original
import native_resident_services as native
from postprocess_backend import NativePostNeural

old_guard = services.InputGuard


class PhaseGuard(old_guard):
    def __init__(self, root):
        super().__init__(root)
        self.experiment = self.campaign.parent
        self.owner_bank = self.reg['bank']
        self.groups = set(common.read(self.experiment/'registration.json')['test_arms'])

    def check(self, directory, kind, **flags):
        directory = Path(directory).resolve()
        if directory.parent.name == 'trials':
            group = directory.parent.parent.parent
            if group.parent != self.experiment or group.name not in self.groups:
                raise RuntimeError('Unregistered pilot input')
            reg = common.read(group/'registration.json')
            if kind == 'graspcl' and reg['bank'] != self.owner_bank:
                raise RuntimeError('Frozen pilot bank differs from loaded bank')
            self.campaign = group;self.reg = reg
            self.allowed = set(reg['test_object_ids']);self.excluded = set(reg['excluded_old_object_ids'])
            self.entries = {e['id']: e for e in reg['schedule']}
            self.arms = {name for name, _, _ in reg['arms']}
        return super().check(directory, kind, **flags)


services.InputGuard = PhaseGuard
old_main = services.main


def main():
    kind = sys.argv[sys.argv.index('--kind')+1]
    if kind == 'curobo':
        from arx_direct_grasp_20261120 import install_resident
        install_resident()
    elif kind == 'zerograsp':
        request = NativePostNeural.request
        def shared(self, directory, **kwargs):
            directory = Path(directory).resolve();root = self.root
            if directory.parent.name == 'trials':
                self.root = directory.parent.parent.parent/'resident'
            try:
                return request(self, directory, **kwargs)
            finally:
                self.root = root
        NativePostNeural.request = shared
    elif kind == 'graspcl':
        request = native.GraspCLWorker.request
        def cached(self, directory, policy):
            directory = Path(directory)
            if directory.parent.name != 'trials':
                return request(self, directory, policy)
            group = directory.parent.parent.parent
            reg = common.read(group/'registration.json')
            record = reg['cached_graspcl_pools'][directory.name]
            source = Path(record['path'])
            assert common.sha(source) == record['sha256']
            destination = directory/'graspcl';destination.mkdir(exist_ok=False)
            common.save(destination/'candidate_pool.json', common.read(source))
            for name in ['state.json', 'camera/params.json', 'camera/color.png', 'camera/depth.png', 'camera/mask.png']:
                target = destination/name;target.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy2(directory/name, target)
            exclusion = {'excluded_feedback_ids': [], 'retained_feedback_ids': list(self.ids)}
            common.save(directory/'self_case_exclusion.json', exclusion)
            common.save(directory/'cached_graspcl_input.json', {**record, 'shared_pre_refinement_predictions': True,
                        'execution_outcomes_used': False, 'test_feedback_added': 0})
            return {'success': True, 'self_case_exclusion': exclusion, 'bank_updates': 0,
                    'cached_pre_refinement_pool': True}
        native.GraspCLWorker.request = cached
    old_main()


services.main = main
original.install()
