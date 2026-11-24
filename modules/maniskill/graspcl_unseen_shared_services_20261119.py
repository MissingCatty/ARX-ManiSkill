"""Original frozen services with audited routing across the four paired phases."""
import faulthandler
from pathlib import Path
import signal
import sys
import time

faulthandler.enable()
faulthandler.register(signal.SIGUSR1,all_threads=True)
faulthandler.dump_traceback_later(60,repeat=True)
started=time.monotonic()
print('SERVICE_BOOTSTRAP importing modules',flush=True)
CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE))
import arx_runtime_imports
arx_runtime_imports.install()
import graspcl_pipeline as common
import short_lift_services as services
import graspcl_unseen_services as original
from postprocess_backend import NativePostNeural

PHASES={'collection','A_zerograsp','B_graspcl_no_experience','C_graspcl_experience50'}
old_guard=services.InputGuard


class PhaseGuard(old_guard):
    def __init__(self,root):
        super().__init__(root)
        self.experiment=self.campaign.parent
        self.owner_bank=self.reg['bank']

    def check(self,directory,kind,**flags):
        directory=Path(directory).resolve()
        if directory.parent.name=='trials':
            group=directory.parent.parent.parent
            if group.parent!=self.experiment or group.name not in PHASES:
                raise RuntimeError('Unregistered cross-phase service input')
            reg=common.read(group/'registration.json')
            if kind=='graspcl' and reg['bank']!=self.owner_bank:
                raise RuntimeError('Requested scoring bank differs from loaded frozen bank')
            self.campaign=group;self.reg=reg
            self.allowed=set(reg['test_object_ids']);self.excluded=set(reg['excluded_old_object_ids'])
            self.entries={e['id']:e for e in reg['schedule']}
            self.arms={name for name,_,_ in reg['arms']}
            assert not self.allowed&self.excluded
        return super().check(directory,kind,**flags)


services.InputGuard=PhaseGuard
old_main=services.main


def main_with_shared_perception():
    shared=NativePostNeural.request
    def request(self,directory,**kwargs):
        directory=Path(directory).resolve()
        root=self.root
        if directory.parent.name=='trials':
            group=directory.parent.parent.parent
            if group.parent!=root.parent.parent or group.name not in PHASES:
                raise RuntimeError('Unknown shared perception phase')
            self.root=group/'resident'
        try:return shared(self,directory,**kwargs)
        finally:self.root=root
    NativePostNeural.request=request
    print('SERVICE_BOOTSTRAP modules imported',time.monotonic()-started,flush=True)
    old_main()


services.main=main_with_shared_perception
original.install()
