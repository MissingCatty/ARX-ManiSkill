"""Keep SAPIEN's renderer context alive across original ARX environments."""
from pathlib import Path
import runpy
import sys

CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE))
import arx_runtime_imports
arx_runtime_imports.install()
import graspcl_unseen_bench as bench

original_modules=bench.core.batch.original_modules
keeper=None
installed=False


def modules():
    global installed
    run,baseline=original_modules()
    if not installed:
        original_make=run.make
        assert baseline.make is original_make
        def make(*args,**kwargs):
            global keeper
            env=original_make(*args,**kwargs)
            if keeper is None:
                import sapien
                keeper=sapien.render.SapienRenderer(env._render_device)
                root=Path(sys.argv[sys.argv.index('--root')+1])
                bench.common.save(root/'renderer_lifetime.json',{'context_owner_count':1,
                    'render_device':str(env._render_device),'pci':env._render_device.pci_string,
                    'created_after_original_first_environment':True,'physics_and_camera_changes':0})
            return env
        run.make=make;baseline.make=make
        installed=True
    return run,baseline


bench.core.batch.original_modules=modules
runpy.run_path(str(CODE/'graspcl_unseen_simulator_20261119.py'),run_name='__main__')
