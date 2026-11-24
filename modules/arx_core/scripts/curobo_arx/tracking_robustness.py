"""H011: query the unchanged ARX checker at sampled tracking states."""
import torch


def probes(q):
    q = q.reshape(-1)
    if q.numel() != 6:
        raise ValueError('This preregistered sampler is for the six-joint ARX right arm')
    indices = torch.arange(64, device=q.device, dtype=torch.long)[:, None]
    bits = ((indices >> torch.arange(6, device=q.device)) & 1) * 2 - 1
    corners = q[None] + bits.to(q.dtype) * .003
    return torch.cat([q[None], corners], dim=0)


def check_goal(checker, q):
    samples = probes(q)
    masks = torch.cat([checker.validate(chunk[:, None]).flatten()
                       for chunk in samples.split(64)])
    return {'pass': bool(masks.all()), 'samples': len(samples),
            'invalid_count': int((~masks).sum()),
            'valid_mask': masks.tolist(), 'joint_delta_rad': .003,
            'scope': 'Nominal plus64 joint-box corners; unchanged robot/world checker. '
                     'Sampled planning test, not a guarantee over all physical tracking errors.'}
