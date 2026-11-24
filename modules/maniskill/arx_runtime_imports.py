"""Skip only unrelated RoboCasa auto-registration in an ARX-only process.

The upstream file, simulator, robot and task implementation stay unchanged.
This loader executes the real mobile_manipulation package initializer with its
single RoboCasa wildcard import omitted. Explicit RoboCasa use is out of scope.
"""
import ast
import hashlib
import importlib.abc
import importlib.machinery
import sys

TARGET='mani_skill.envs.tasks.mobile_manipulation'
_installed=False
PROOF={}


class Loader(importlib.abc.Loader):
    def __init__(self,original,origin):
        self.original=original;self.origin=origin

    def create_module(self,spec):return None

    def exec_module(self,module):
        source=self.original.get_source(TARGET)
        tree=ast.parse(source,self.origin)
        removed=[node for node in tree.body if isinstance(node,ast.ImportFrom) and node.level==1 and node.module=='robocasa']
        assert len(removed)==1 and len(removed[0].names)==1 and removed[0].names[0].name=='*'
        tree.body=[node for node in tree.body if node not in removed]
        assert any(isinstance(node,ast.ImportFrom) and node.module=='open_cabinet_drawer' for node in tree.body)
        PROOF.update(upstream_file=self.origin,upstream_sha256=hashlib.sha256(source.encode()).hexdigest(),
            omitted_import='from .robocasa import *',omitted_import_count=1,
            physics_robot_ycb_and_arx_task_changes=0)
        exec(compile(tree,self.origin,'exec'),module.__dict__)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!=TARGET:return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        assert spec is not None and spec.loader is not None
        spec.loader=Loader(spec.loader,spec.origin)
        return spec


def install():
    global _installed
    if _installed:return
    if TARGET in sys.modules:raise RuntimeError('ARX import scope must be installed before ManiSkill imports')
    sys.meta_path.insert(0,Finder())
    _installed=True
