"""Batch candidate endpoints once and retain their original score order."""
import numpy as np


STAGES=('grasp','pregrasp','lift')


def endpoint_poses(candidates):
    poses=[];widths=[]
    for candidate in candidates:
        grasp=np.asarray(candidate['pose'],dtype=np.float64).copy()
        grasp[:3,3]-=.043*grasp[:3,2]
        pre=grasp.copy();pre[:3,3]-=.08*grasp[:3,2]
        lift=grasp.copy();lift[2,3]+=.03
        poses.extend([grasp,pre,lift]);widths.extend([candidate['width']]*3)
    return np.asarray(poses).reshape(-1,4,4),np.asarray(widths,dtype=np.float32)


class EndpointBank:
    def __init__(self,session,candidates,invbase):
        self.poses,widths=endpoint_poses(candidates)
        targets=np.asarray(invbase,dtype=np.float64)[None]@self.poses
        blocks=[]
        for begin in range(0,len(targets),256):
            value=session.host_query(targets[begin:begin+256],widths[begin:begin+256])
            if not value['valid_input'].all():raise ValueError('Malformed original endpoint target')
            blocks.append(value)
        self.results=({name:np.concatenate([b[name] for b in blocks]) for name in blocks[0]}
                      if blocks else {})
        self.calls=len(blocks);self.cursor=0;self.used=False;self.enabled=True;self.events=[]

    def branches(self,stage,start):
        row=self.cursor*3+STAGES.index(stage)
        mask=self.results['valid_ik' if stage=='grasp' else 'valid_world'][row]
        q=self.results['q'][row,mask].copy()
        if len(q):q=q[np.argsort(np.square(q-np.asarray(start)).sum(1),kind='stable')]
        return q

    def matches(self,stage,world):
        expected=self.poses[self.cursor*3+STAGES.index(stage)]
        return np.allclose(expected,world,atol=1e-9,rtol=0)

    def summary(self):
        return {'candidates':len(self.poses)//3,'endpoint_targets':len(self.poses),
            'fixed256_batches':self.calls,'regular_branch_capacity':8,
            'analytic_targets_with_legal_branch':int(self.results['valid_ik'].any(-1).sum()) if self.results else 0,
            'native_world_clear_branches':int(self.results['valid_world'].sum()) if self.results else 0,
            'events':self.events,'strict_table_rejection_does_not_delete_grasp':True}
