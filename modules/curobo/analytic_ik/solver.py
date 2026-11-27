"""Tensorized IK-Geo subproblem decomposition, fixed eight regular branches.

Adapted from the mathematical decomposition in rpiRobotics/ik-geo.
Copyright (c) 2022, rpiRobotics. BSD-3-Clause; see reference/LICENSE.
The implementation below batches every branch and supports CUDA Graph replay.
"""
import numpy as np
import torch

from .model import AnalyticModel


def rotate(axis, angle):
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero]).reshape(3, 3)
    identity = torch.eye(3, dtype=axis.dtype, device=axis.device)
    return identity + angle.sin()[..., None, None]*skew + (1-angle.cos())[..., None, None]*(skew@skew)


def mv(matrix, vector):
    return (matrix@vector[..., None]).squeeze(-1)


def circle_plane(h, p, axis, d, eps):
    parallel = (p*axis).sum(-1, keepdim=True)*axis
    perpendicular = p-parallel
    cosine = (h*perpendicular).sum(-1)
    sine = (h*torch.linalg.cross(axis.expand_as(p), p)).sum(-1)
    constant = (h*parallel).sum(-1)
    scale = (cosine.square()+sine.square()).sqrt()
    residual = d-constant
    ratio = residual/scale.clamp_min(eps)
    phase = torch.atan2(sine, cosine)
    delta = torch.acos(ratio.clamp(-1, 1))
    angle = torch.stack([phase+delta, phase-delta], dim=-1)
    valid = (residual.abs() <= scale+eps) & (scale > eps)
    return angle, valid, (scale+torch.zeros_like(residual)) <= eps


def circle_point(a, b, axis, eps):
    a = a-(a*axis).sum(-1, keepdim=True)*axis
    b = b-(b*axis).sum(-1, keepdim=True)*axis
    cosine = (a*b).sum(-1)
    sine = (axis*torch.linalg.cross(a, b)).sum(-1)
    scale = torch.linalg.vector_norm(a, dim=-1)*torch.linalg.vector_norm(b, dim=-1)
    return torch.atan2(sine, cosine), scale <= eps


class BatchAnalyticIK:
    def __init__(self, arm='right', *, device='cpu', dtype=torch.float64):
        self.model = AnalyticModel(arm)
        self.device, self.dtype = torch.device(device), dtype
        to = lambda values: torch.as_tensor(values, device=self.device, dtype=dtype)
        self.H = to(self.model.H.T);self.P = to(self.model.P.T)
        self.original_H = to(self.model.chain.H.T);self.original_P = to(self.model.chain.P.T)
        self.tool = to(self.model.chain.R_tool);self.signs = to(self.model.signs)
        limits = self.model.chain.limits
        if dtype == torch.float32:
            lower, upper = limits[:, 0].astype(np.float32), limits[:, 1].astype(np.float32)
            lower = np.where(lower.astype(float) < limits[:, 0], np.nextafter(lower, np.float32(np.inf)), lower)
            upper = np.where(upper.astype(float) > limits[:, 1], np.nextafter(upper, np.float32(-np.inf)), upper)
        else:
            lower, upper = limits[:, 0], limits[:, 1]
        self.lower, self.upper = to(lower), to(upper)
        self.eps = 2e-7 if dtype == torch.float32 else 1e-12
        self.angular_eps = 2e-6 if dtype == torch.float32 else 1e-11

    def fk(self, q):
        identity = torch.eye(3, device=q.device, dtype=q.dtype)
        rotation = identity.expand(*q.shape[:-1], 3, 3)
        position = self.original_P[0].expand(*q.shape[:-1], 3)
        for index in range(6):
            rotation = rotation@rotate(self.original_H[index], q[..., index])
            position = position+mv(rotation, self.original_P[index+1])
        return position, rotation@self.tool

    def solve(self, targets):
        """Return finite padded q[N,8,6], masks, errors, and fallback flags."""
        if targets.ndim != 3 or targets.shape[1:] != (4, 4) or targets.device != self.device or targets.dtype != self.dtype:
            raise ValueError('Targets must have shape [N,4,4] and the registered device/dtype')
        h, p, eps = self.H, self.P, self.eps
        finite = torch.isfinite(targets).all(dim=(-1, -2))
        identity = torch.eye(4, device=self.device, dtype=self.dtype)
        safe = torch.where(finite[:, None, None], targets, identity)
        input_rotation = safe[:, :3, :3]
        orthogonal = ((input_rotation.transpose(-1, -2)@input_rotation - identity[:3, :3]).square().sum((-1,-2)) < 4e-10)
        determinant = (torch.linalg.cross(input_rotation[:,:,0],input_rotation[:,:,1])*input_rotation[:,:,2]).sum(-1)
        valid_input = finite & orthogonal & (determinant > .99998) & ((safe[:,3]-identity[3]).abs().amax(-1)<1e-6)
        rotation = input_rotation@self.tool.T
        position = safe[:, :3, 3]-p[0]-mv(rotation, p[6])
        first, first_valid, first_singular = circle_plane(h[1], position, -h[0],
            (h[1]*p[1:5].sum(0)).sum(), eps)
        r1 = rotate(h[0], first)
        r16 = r1.transpose(-1, -2)@rotation[:, None]
        wrist, wrist_valid, wrist_singular = circle_plane(h[1], h[5], h[4],
            (mv(r16,h[5])*h[1]).sum(-1), eps)
        r5 = rotate(h[4], wrist)
        total, total_singular = circle_point(mv(r5,h[5]), mv(r16,h[5])[:, :, None], h[1], eps)
        last, last_singular = circle_point(mv(r5.transpose(-1,-2),h[1]),
            mv(r16.transpose(-1,-2),h[1])[:, :, None], -h[5], eps)
        inner = mv(r1.transpose(-1,-2),position[:,None])[:, :, None]-p[1]-mv(rotate(h[1],total),p[4])
        d = (inner.square().sum(-1)-p[2].square().sum()-p[3].square().sum())/2
        elbow, elbow_valid, elbow_singular = circle_plane(p[2],p[3],h[1],d,eps*0.1)
        second, second_singular = circle_point(p[2]+mv(rotate(h[1],elbow),p[3]),inner[:,:,:,None],h[1],eps)
        fourth = total[:,:,:,None]-second-elbow
        shape = elbow.shape
        raw = torch.stack([first[:,:,None,None].expand(shape), second, elbow, fourth,
            wrist[:,:,:,None].expand(shape), last[:,:,:,None].expand(shape)],-1).reshape(-1,8,6)*self.signs
        q = torch.atan2(raw.sin(),raw.cos())
        turns = torch.ceil((self.lower-q-self.angular_eps)/(2*torch.pi))
        q = q+turns*(2*torch.pi)
        q = torch.where((q < self.lower)&(q >= self.lower-self.angular_eps), self.lower,q)
        q = torch.where((q > self.upper)&(q <= self.upper+self.angular_eps), self.upper,q)
        bounds = (q >= self.lower).all(-1)&(q <= self.upper).all(-1)
        # Near a singular family, finite representatives can still pass the
        # independent pose residual below. Preserve them while explicitly
        # requesting fallback for branch completeness.
        branch = (first_valid[:,None,None,None] & wrist_valid[:,:,None,None] &
            elbow_valid[:,:,:,None]).expand(shape).reshape(-1,8)
        actual_position, actual_rotation = self.fk(q)
        position_error = torch.linalg.vector_norm(actual_position-safe[:,None,:3,3],dim=-1)
        relative = actual_rotation@input_rotation[:,None].transpose(-1,-2)
        skew = torch.stack([relative[...,2,1]-relative[...,1,2],relative[...,0,2]-relative[...,2,0],
            relative[...,1,0]-relative[...,0,1]],-1)/2
        rotation_error = torch.atan2(torch.linalg.vector_norm(skew,dim=-1),
            (relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2)
        valid = valid_input[:,None]&branch&bounds&(position_error<=.002)&(rotation_error<=.03)&torch.isfinite(q).all(-1)
        pair_distance = (q[:,:,None]-q[:,None,:]).abs().amax(-1)
        earlier = torch.tril(torch.ones((8,8),device=self.device,dtype=torch.bool),diagonal=-1)
        duplicates = ((pair_distance < self.angular_eps*4)&valid[:,None,:]&earlier).any(-1)
        valid = valid&~duplicates
        singular = first_singular | wrist_singular.any(-1) | total_singular.any((-1,-2)) | last_singular.any((-1,-2)) | second_singular.any((-1,-2,-3)) | elbow_singular.any((-1,-2))
        return {'q':q,'valid':valid,'position_error_m':position_error,'rotation_error_rad':rotation_error,
            'singular':singular,'needs_fallback':~valid.any(-1)|singular,'valid_input':valid_input}
