"""Run unchanged baseline algorithms with isolated side-task IO paths.

Archive the exact adapted source and diff. Only path routing is substituted.
"""
import sys,os,difflib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
name,stage=sys.argv[1:3]
assert name.isidentifier()
source=ROOT/'scripts/curobo_arx'/stage
assert stage in ('grasp_input.py','prepare.py','grasp_execute.py','zerograsp_request.py')
out=ROOT/'runs/arx_dynamic_pick'/name
text=source.read_text()
adapted=text.replace('runs/arx_grasp','runs/arx_dynamic_pick/'+name).replace('runs/arx_curobo/mailbox','runs/arx_dynamic_pick/mailbox')
(out/('adapted_'+stage)).write_text(adapted)
(out/(stage+'.diff')).write_text(''.join(difflib.unified_diff(text.splitlines(True),adapted.splitlines(True),fromfile=str(source),tofile='isolated IO')))
sys.path.insert(0,str(source.parent));sys.argv=[str(source),*sys.argv[3:]]
import curobo._src.runtime as runtime
runtime.cache_dir=str((ROOT/'runs/arx_dynamic_pick').resolve().parent/'curobo-cache')
# Register kernels before importing checker-first baseline prepare.py.
from curobo.motion_planner import MotionPlanner
exec(compile(adapted,str(source),'exec'),{'__file__':str(source),'__name__':'__main__'})
