"""Reusable fixed-capacity CUDA Graph API for candidate endpoint checks."""
from threading import Lock

import numpy as np
import torch

from .executable import ExecutableAnalyticIK
from .collision import BranchCollisionChecker


def scene_storage(root):
    """Fingerprint tensor storage used by the native captured scene queries."""
    stack=[('scene',root)];seen=set();result={}
    while stack:
        name,value=stack.pop()
        if id(value) in seen:continue
        seen.add(id(value))
        if isinstance(value,torch.Tensor):
            result[name]=(value.data_ptr(),tuple(value.shape),value.dtype)
        elif isinstance(value,dict):stack.extend((name+'.'+str(k),v) for k,v in value.items())
        elif isinstance(value,(list,tuple)):stack.extend((name+'.'+str(i),v) for i,v in enumerate(value))
        elif type(value).__module__.startswith('curobo.') and hasattr(value,'__dict__'):
            stack.extend((name+'.'+k,v) for k,v in vars(value).items())
        if len(seen)>100000:raise RuntimeError('Unexpected native scene object graph')
    return result


class AnalyticBatchSession:
    def __init__(self,robot_config,scene,*,capacity=256,device='cuda:0',include_self=False,native_scene=None):
        if capacity!=256:raise ValueError('This validated endpoint session has fixed capacity 256')
        self.capacity=capacity;self.device=torch.device(device);self.include_self=bool(include_self)
        if self.device!=torch.device('cuda:0'):
            raise ValueError('Select one visible GPU and use cuda:0 for the native collision model')
        self.stream=torch.cuda.current_stream(self.device)
        self._solver=ExecutableAnalyticIK(device=device)
        # This native checker owns captured scratch buffers. Never resize it or
        # share it with eager diagnostics after graph capture.
        self._collision=BranchCollisionChecker(robot_config,scene,native_scene=native_scene)
        self._shared_scene=native_scene
        self._scene_storage=scene_storage(self._collision.checker.scene_model.data)
        self.targets=torch.eye(4,device=device,dtype=torch.float64)[None].repeat(capacity,1,1)
        self.widths=torch.full((capacity,),.088,device=device,dtype=torch.float32)
        self.real=torch.zeros(capacity,dtype=torch.bool,device=device)
        self.lock=Lock()
        for _ in range(3):self.answer,self.checked=self._compute()
        torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.answer,self.checked=self._compute()
        self.graph.replay();torch.cuda.synchronize()

    def update_scene(self,scene_config):
        """Replace obstacles in the fixed native cache without rebuilding IK/Graph."""
        if self._shared_scene is not None:
            raise RuntimeError('The parent planner owns this shared scene; update it through the parent')
        if torch.cuda.current_stream(self.device)!=self.stream:
            raise ValueError('Update the scene on the registered CUDA stream')
        with self.lock:
            torch.cuda.synchronize()
            checker=self._collision.checker
            before=scene_storage(checker.scene_model.data)
            checker.clear_scene_cache();checker.update_world(scene_config)
            torch.cuda.synchronize()
            if scene_storage(checker.scene_model.data)!=before:
                raise RuntimeError('Native scene update changed captured tensor storage')

    def verify_shared_scene(self,native_scene):
        if self._shared_scene is not native_scene or self._collision.checker.scene_model is not native_scene:
            raise RuntimeError('Analytic graph no longer refers to its parent scene')
        if scene_storage(native_scene.data)!=self._scene_storage:
            raise RuntimeError('Parent scene replaced captured tensor storage')

    def _compute(self):
        answer=self._solver.solve(self.targets)
        collision=self._collision.check(answer['q'],answer['valid']&self.real[:,None],self.widths,
            include_self=self.include_self)
        return answer,collision

    def device_query(self,targets,widths):
        """Return owned GPU outputs; subsequent calls never overwrite them."""
        n=len(targets)
        if targets.shape!=(n,4,4) or widths.shape!=(n,) or n>self.capacity:
            raise ValueError('Expected at most 256 original poses and opening widths')
        if targets.device!=self.device or widths.device!=self.device or targets.dtype!=torch.float64 or widths.dtype!=torch.float32:
            raise ValueError('Use double targets and float32 widths on the registered GPU')
        if torch.cuda.current_stream(self.device)!=self.stream:
            raise ValueError('Use this session on its registered CUDA stream')
        with self.lock:
            if n:
                self.targets.copy_(targets[:1].expand_as(self.targets));self.widths.copy_(widths[:1].expand_as(self.widths))
                self.targets[:n].copy_(targets);self.widths[:n].copy_(widths)
            self.real.zero_();self.real[:n]=True;self.graph.replay()
            return {'q':self.answer['q'][:n].clone(),'valid_ik':self.answer['valid'][:n].clone(),
                'valid_world':self.checked['valid_world_only'][:n].clone(),
                'valid_collision':self.checked['valid_world_and_self'][:n].clone(),
                'self_collision_checked':torch.full((n,),self.include_self,device=self.device,dtype=torch.bool),
                'needs_fallback':self.answer['needs_fallback'][:n].clone(),
                'valid_input':self.answer['valid_input'][:n].clone()}

    def host_query(self,targets,widths):
        targets=np.asarray(targets,dtype=np.float64);widths=np.asarray(widths,dtype=np.float32)
        result=self.device_query(torch.as_tensor(targets,device=self.device),torch.as_tensor(widths,device=self.device))
        return {name:value.cpu().numpy() for name,value in result.items()}
