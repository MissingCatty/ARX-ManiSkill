"""Registered XY/yaw diversity for new teacher collection only.

Historical benchmark resets and the catalog's reference placements are retained.
No asset, camera, robot, physics parameter, support height or success rule changes.
"""
import copy
from collections import Counter
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from .common import digest, sha


def validation_subset(catalog, names):
    result=copy.deepcopy(catalog);requested=set(names)
    available={a['object_id'] for a in result['objects'] if a['available'] and a.get('collection_enabled',True)}
    if not requested or requested-available:raise ValueError('Bounded validation objects must be available collection targets')
    for asset in result['objects']:
        if asset.get('collection_enabled',True) and asset['object_id'] not in requested:
            asset.update(collection_enabled=False,collection_exclusion_reason='Outside explicitly registered bounded validation subset')
    result.setdefault('collection_selection',{})['selected_object_ids']=sorted(requested)
    result['collection_selection']['excluded_object_ids']=[a['object_id'] for a in result['objects'] if not a.get('collection_enabled',True)]
    result['catalog_sha256']=digest(result['objects'])
    return result


def support_surface(environment, suffix):
    from pxr import Usd,UsdGeom
    stage=Usd.Stage.Open(str(environment))
    matches=[p for p in stage.Traverse() if str(p.GetPath()).endswith(suffix)]
    if len(matches)!=1 or not matches[0].IsA(UsdGeom.Mesh):
        raise ValueError('Registered historical table support must resolve to one mesh')
    prim=matches[0];mesh=UsdGeom.Mesh(prim)
    points=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
    matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
    points=(np.c_[points,np.ones(len(points))]@matrix.T)[:,:3]
    top=points[points[:,2]>=points[:,2].max()-.003]
    if len(top)<3:raise ValueError('Missing table top geometry')
    hull=ConvexHull(top[:,:2])
    return {'polygon_xy':top[hull.vertices,:2].tolist(),'top_z_range_m':[float(top[:,2].min()),float(top[:,2].max())],
        'environment':str(environment),'prim':str(prim.GetPath()),
        'dependencies':{layer.realPath:sha(layer.realPath) for layer in stage.GetUsedLayers() if layer.realPath}}


def placement_bank(asset, vertices, profile, support, seed):
    """Yaw about the original root; retain Z and the original resting tilt."""
    if profile['version']!='tabletop_xy_yaw_v1':raise ValueError('Unknown collection placement profile')
    offsets=np.asarray(profile['xy_offsets_m'],dtype=float)
    yaws=np.asarray(profile['yaw_degrees'],dtype=float)
    if (offsets.ndim!=2 or offsets.shape[1]!=2 or not len(offsets) or not len(yaws)
            or not np.isfinite(offsets).all() or not np.isfinite(yaws).all()):
        raise ValueError('Invalid bounded XY/yaw strata')
    base=copy.deepcopy(asset['placement']);xyz=np.asarray(base['position'],dtype=float)
    wxyz=np.asarray(base['orientation'],dtype=float)
    rotation=Rotation.from_quat(np.r_[wxyz[1:],wxyz[0]])
    hull=ConvexHull(np.asarray(support['polygon_xy']))
    planes=hull.equations;normal=np.linalg.norm(planes[:,:2],axis=1)
    accepted=[];rejected=[];seen=set()
    for xy_index,offset in enumerate(offsets):
        for yaw_index,degrees in enumerate(yaws):
            rot=Rotation.from_euler('z',float(degrees),degrees=True)*rotation
            position=xyz+np.r_[offset,0.]
            points=np.asarray(vertices)@rot.as_matrix().T+position
            # Every actual visual vertex must be inside the support polygon
            # with the declared data-placement border. This is not a collider margin.
            clearance=-(points[:,:2]@planes[:,:2].T+planes[:,2])/normal
            key=digest([np.round(position,9).tolist(),np.round(rot.as_matrix(),9).tolist()])
            if key in seen:raise ValueError('Duplicate collection placement strata')
            seen.add(key)
            metadata={'placement_id':digest([asset['object_id'],key])[:20],
                'xy_bin':xy_index,'yaw_bin':yaw_index,'offset_xy_m':offset.tolist(),'yaw_delta_degrees':float(degrees)}
            if float(clearance.min())<profile['support_border_m']:
                rejected.append({**metadata,'reason':'target_footprint_outside_registered_table_support',
                    'minimum_support_border_m':float(clearance.min())});continue
            quat=rot.as_quat()
            accepted.append({**metadata,'placement':{'position':position.tolist(),
                'orientation':np.r_[quat[3],quat[:3]].tolist()},
                'minimum_support_border_m':float(clearance.min())})
    # Deficient XY/yaw marginal coverage before a repeat of either, followed by
    # a deterministic per-object tie order. No successful-outcome selection.
    ordered=[];xy_counts=Counter();yaw_counts=Counter()
    while accepted:
        index=min(range(len(accepted)),key=lambda i:(
            xy_counts[accepted[i]['xy_bin']]+yaw_counts[accepted[i]['yaw_bin']],
            max(xy_counts[accepted[i]['xy_bin']],yaw_counts[accepted[i]['yaw_bin']]),
            float(np.linalg.norm(accepted[i]['offset_xy_m'])) if profile.get('reference_first') else 0,
            min(accepted[i]['yaw_delta_degrees']%360,(-accepted[i]['yaw_delta_degrees'])%360) if profile.get('reference_first') else 0,
            digest([seed,asset['object_id'],accepted[i]['placement_id']])))
        item=accepted.pop(index);ordered.append(item)
        xy_counts[item['xy_bin']]+=1;yaw_counts[item['yaw_bin']]+=1
    if not ordered:raise ValueError('No supported collection placements for '+asset['object_id'])
    preferred=profile.get('first_placements',{}).get(asset['object_id'])
    if preferred is not None:
        matches=[p for p in ordered if np.allclose(p['offset_xy_m'],preferred['offset_xy_m'],atol=1e-12,rtol=0)
            and abs(p['yaw_delta_degrees']-preferred['yaw_delta_degrees'])<1e-9]
        if len(matches)!=1:raise ValueError('Preferred first placement is not a supported registered bin')
        ordered=[matches[0]]+[p for p in ordered if p['placement_id']!=matches[0]['placement_id']]
    return {'version':profile['version'],'placements':ordered,'rejected':rejected,
        'reference_placement':base,'requested_bins':len(offsets)*len(yaws),
        'scope':'New teacher episodes only; XY/yaw variations, unchanged Z/resting tilt; settling and original full ARX checks still required'}


def active_asset(asset, placement):
    result=copy.deepcopy(asset)
    result['reference_placement']=copy.deepcopy(asset['placement'])
    result['placement']=copy.deepcopy(placement['placement'])
    result['collection_placement']=copy.deepcopy(placement)
    return result


def tag_pool(pool, asset):
    """Do not merge nearby Gs or successes from distinct object initial poses."""
    result=copy.deepcopy(pool);placement=asset.get('collection_placement')
    if placement is None:return result
    for goal in result['candidates']:
        goal['unscoped_goal_id']=goal['goal_id']
        goal['placement_id']=placement['placement_id']
        goal['goal_id']=digest([placement['placement_id'],goal['goal_id']])[:20]
    result['collection_placement']=placement
    return result


def available_placement(asset, goals, rows, events, quotas):
    if not asset.get('collection_placements'):return None
    visits=Counter();exhausted=set()
    for event in events:
        value=event['value']
        if value.get('object_id')!=asset['object_id']:continue
        ident=value.get('placement_id')
        if event['event']=='object_acquired':visits[ident]+=1
        if event['event']=='placement_exhausted':exhausted.add(ident)
    successes=Counter((r['goal_id'],r['distance']) for r in rows if r['state']=='committed')
    options=[]
    for index,placement in enumerate(asset['collection_placements']):
        ident=placement['placement_id']
        if ident in exhausted:continue
        scoped=[g for g in goals if g.get('placement_id')==ident]
        full=sum(all(successes[g['goal_id'],d]>=quotas['successes_per_bin'] for d in quotas['distances_mm']) for g in scoped)
        if full>=quotas['goals_per_object']:continue
        options.append((visits[ident],index,placement))
    return min(options,key=lambda p:p[:2])[2] if options else None
