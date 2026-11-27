"""Double-precision branch algebra with independently checked float32 commands."""
import torch

from .solver import BatchAnalyticIK


class ExecutableAnalyticIK:
    def __init__(self,arm='right',*,device='cpu'):
        self.algebra=BatchAnalyticIK(arm,device=device,dtype=torch.float64)
        self.command_limits=BatchAnalyticIK(arm,device=device,dtype=torch.float32)
        self.device=self.algebra.device
        self.model=self.algebra.model

    def solve(self,targets):
        if targets.dtype!=torch.float64 or targets.device!=self.device:
            raise ValueError('The precise API takes original double-precision target matrices')
        original=targets
        identity=torch.eye(4,dtype=torch.float64,device=self.device)
        finite=torch.isfinite(targets).all((-1,-2))
        safe=torch.where(finite[:,None,None],targets,identity)
        rotation=safe[:,:3,:3]
        orthogonal=((rotation.transpose(-1,-2)@rotation-identity[:3,:3]).square().sum((-1,-2))<4e-10)
        determinant=(torch.linalg.cross(rotation[:,:,0],rotation[:,:,1])*rotation[:,:,2]).sum(-1)
        input_valid=finite&orthogonal&(determinant>.99998)&((safe[:,3]-identity[3]).abs().amax(-1)<1e-6)
        # Recorded float32 rotation matrices may not be exactly orthogonal.
        # Normalize only the internal IK query; verify against the raw target.
        first=rotation[:,:,0];first=first/torch.linalg.vector_norm(first,dim=-1,keepdim=True).clamp_min(1e-12)
        second=rotation[:,:,1]-(rotation[:,:,1]*first).sum(-1,keepdim=True)*first
        second=second/torch.linalg.vector_norm(second,dim=-1,keepdim=True).clamp_min(1e-12)
        query=safe.clone();query[:,:3,:3]=torch.stack([first,second,torch.linalg.cross(first,second)],-1)
        answer=self.algebra.solve(query)
        q=answer['q'].to(torch.float32)
        # Only account for the float32 boundary rounding, never widen bounds.
        q=torch.maximum(torch.minimum(q,self.command_limits.upper),self.command_limits.lower)
        executable=q.to(torch.float64)
        position,actual_rotation=self.algebra.fk(executable)
        pe=torch.linalg.vector_norm(position-safe[:,None,:3,3],dim=-1)
        relative=actual_rotation@rotation[:,None].transpose(-1,-2)
        skew=torch.stack([relative[...,2,1]-relative[...,1,2],relative[...,0,2]-relative[...,2,0],
            relative[...,1,0]-relative[...,0,1]],-1)/2
        re=torch.atan2(torch.linalg.vector_norm(skew,dim=-1),(relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2)
        mask=answer['valid']&input_valid[:,None]&(pe<=.002)&(re<=.03)
        pair=(q[:,:,None]-q[:,None,:]).abs().amax(-1)
        previous=torch.tril(torch.ones((8,8),device=self.device,dtype=torch.bool),diagonal=-1)
        duplicate=((pair<1e-6)&mask[:,None,:]&previous).any(-1)
        mask=mask&~duplicate
        return {'q':q,'valid':mask,'position_error_m':pe,'rotation_error_rad':re,
            'singular':answer['singular'],'needs_fallback':~mask.any(-1)|answer['singular'],
            'valid_input':input_valid}
