"""Copy only the fields proven to depend on prismatic finger locks."""


def copy_lock_dependent_fields(target,source):
    for name in ['fixed_transforms','link_spheres','reference_link_spheres']:
        old,new=getattr(target,name),getattr(source,name)
        if (old is None)!=(new is None):raise RuntimeError('Frozen kinematics field layout changed')
        if old is not None:old.copy_(new)
    if (target.lock_jointstate is None)!=(source.lock_jointstate is None):
        raise RuntimeError('Frozen locked-joint layout changed')
    if target.lock_jointstate is not None:target.lock_jointstate.copy_(source.lock_jointstate)
