"""Main-thread diagnostic snapshots, without asynchronous Python frame walking."""
import faulthandler
from pathlib import Path
import threading
import time
import traceback


class FaultDiagnostics:
    def __init__(self,path,*,interval=120.,backend=None,clock=None):
        self.backend=backend or faulthandler
        self.clock=clock or time.monotonic
        self.owner=threading.get_ident();self.interval=interval
        self.last=self.clock();self.closed=False
        self.stream=Path(path).open('w')
        # A watchdog reads other threads' frames without their execution lock.
        # Keep fatal-signal diagnostics, and sample our own stack only at safe
        # points reached by the simulator's main thread.
        self.backend.cancel_dump_traceback_later()
        self.backend.enable(file=self.stream,all_threads=False)
        self.stream.write('Fatal signal handler enabled; asynchronous timed stack sampling disabled.\n')
        self.stream.flush()

    def tick(self):
        if threading.get_ident()!=self.owner:
            raise RuntimeError('Diagnostic stack sampling must run on the simulator main thread')
        if self.closed:raise RuntimeError('Diagnostic stream is closed')
        now=self.clock()
        if now-self.last<self.interval:return False
        self.last=now
        self.stream.write(f'\nSAFE_MAIN_THREAD_STACK wall={time.time():.6f}\n')
        traceback.print_stack(file=self.stream)
        self.stream.flush()
        return True

    def close(self):
        if self.closed:return
        # Python's handler retains the FD; disable it before closing that FD.
        self.backend.cancel_dump_traceback_later()
        self.backend.disable()
        self.stream.close();self.closed=True
