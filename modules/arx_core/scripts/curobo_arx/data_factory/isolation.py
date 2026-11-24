"""Disjoint clone groups that retain original within-environment filtering."""
from .common import digest


def isolate(stage, roots):
    from pxr import Sdf,Usd,UsdPhysics
    ownership={}
    for i,root in enumerate(roots):
        original=[UsdPhysics.CollisionGroup(p) for p in Usd.PrimRange(stage.GetPrimAtPath(root)) if p.IsA(UsdPhysics.CollisionGroup)]
        queries=[g.GetCollidersCollectionAPI().ComputeMembershipQuery() for g in original]
        unassigned=[]
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():continue
            members=[j for j,q in enumerate(queries) if q.IsPathIncluded(prim.GetPath())]
            if len(members)>1:
                raise RuntimeError('Preexisting overlapping collision-group membership: '+str(prim.GetPath()))
            if not members:unassigned.append(prim.GetPath())
        default=UsdPhysics.CollisionGroup.Define(stage,'/World/CloneCollisionGroups/default_'+str(i))
        collection=default.GetCollidersCollectionAPI()
        collection.CreateExpansionRuleAttr().Set('explicitOnly')
        collection.CreateIncludesRel().SetTargets(unassigned)
        groups=original+[default]
        ownership[root]={'groups':groups,'original_groups':[str(g.GetPath()) for g in original],
            'default_group':str(default.GetPath()),'unassigned_colliders':len(unassigned),
            'original_filters':{str(g.GetPath()):[str(p) for p in g.GetFilteredGroupsRel().GetTargets()] for g in original}}
    for root,record in ownership.items():
        other=[g.GetPath() for key,value in ownership.items() if key!=root for g in value['groups']]
        for group in record['groups']:
            old=group.GetFilteredGroupsRel().GetTargets()
            group.CreateFilteredGroupsRel().SetTargets(old+[p for p in other if p not in old])
        record['cross_environment_groups']=[str(p) for p in other]
    for record in ownership.values():
        record.pop('groups')
    return ownership
