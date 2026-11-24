"""Reproduce the saved ARX state in the original fresh planner; no execution."""
import json
import sys
import time
import traceback
from pathlib import Path

CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE / 'arx_pipeline'))
import planner
from graspcl_execute import guard
from graspcl_pipeline import read,save


root=Path(sys.argv[1]).resolve()
reg=read(root / 'registration.json')
guard(reg)
reports={}
for label,path in [('failed',reg['failed_state']),('clear_control',reg['clear_control_state'])]:
    started=time.perf_counter()
    directory=root / label
    directory.mkdir()
    try:
        context=planner.Planner(directory,read(path),include_target=True)
        reports[label]={'initial_state_collision':False,'original_fresh_constructor_passed':True}
        del context
    except AssertionError as error:
        if str(error)!='Initial ARX state collides in cuRobo model':raise
        reports[label]={'initial_state_collision':True,'original_fresh_constructor_passed':False,'original_gate':str(error)}
    except BaseException:
        save(root / 'diagnosis_error.json',{'error':traceback.format_exc()})
        raise
    reports[label]['wall_seconds']=time.perf_counter()-started
    save(root / 'reproduction.json',reports)
    print(label,reports[label],flush=True)
assert reports['failed']['initial_state_collision'] and not reports['clear_control']['initial_state_collision']
save(root / 'verification.json',{'passed':True,'fresh_original_model_reproduces_collision':True,
    'clear_control_passed':True,'physical_attempts':0,'failed_slot_no_commanded_motion':True,
    'conclusion':'A planning validity rejection was routed as a service failure; retain the original collision gate and report the case as failed.'})
