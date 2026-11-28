"""Use the frozen orchestrator with a schedule-based final-cohort reviewer."""
import argparse
from pathlib import Path
from threading import Event,Thread
import traceback
from types import FunctionType

from arx_project.layout import ROOT,ensure_run_path,now,read_json,write_json
from modules.pipeline.experiments.endpoint_parallel.shared_runner import record_load
from modules.pipeline.experiments.parallel_planning.run import run as original_run
from modules.pipeline.experiments.graspgenx_native_postprocessing.planning import review as review_schedule
from modules.pipeline.experiments.parallel_planning.audit import audit
from .slot_run import analyze


def run(root):
    reg=read_json(root/'registration.json')
    if reg['planning_requests']!=136 or reg['config']['repeats']!=1:
        raise RuntimeError('Expected the single final paired schedule')
    # The original pilot reviewer assumes two repeats. A private namespace keeps
    # the unchanged runner while selecting its schedule-driven independent review.
    namespace={**original_run.__globals__,'review':review_schedule}
    runner=FunctionType(original_run.__code__,namespace,argdefs=original_run.__defaults__,closure=original_run.__closure__)
    stop=Event(); thread=Thread(target=record_load,args=(root,reg['gpu'],stop)); thread.start()
    try: runner(root)
    finally: stop.set(); thread.join(timeout=15)
    if thread.is_alive() or not read_json(root/'gpu_load_summary.json')['planning_samples']:
        raise RuntimeError('Incomplete final GPU load records')
    audit(root); analyze(root)
    write_json(ROOT/'tmp/state_updates/scene_reuse_active.json',{'at':now(),'run':str(root.relative_to(ROOT)),
        'stage':reg['config']['stage'],'phase':'complete'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    root=ensure_run_path(parser.parse_args().root)
    try: run(root)
    except BaseException:
        write_json(root/'STOP',{'at':now(),'error':traceback.format_exc(),'automatic_retry':False}); raise
