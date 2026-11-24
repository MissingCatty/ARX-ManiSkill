"""UNREGISTERED D3 segmented full trajectory optimization contact-drag preparation; never edits scene properties.

Geometry reads are proposal data only. Actual actions require a separate
registered ARX diagnostic and the unchanged planner/checker/controller API.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh
import yaml
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from optimize_path import optimize_path
from settling import assess_settling
from curobo.types import JointState


def masked_points(folder):
    folder = Path(folder)
    params = json.loads((folder/'camera/params.json').read_text())
    k = params['intrinsics']
    depth = np.asarray(Image.open(folder/'camera/depth.png'), dtype=float)*params['depth_scale']
    mask = binary_erosion(np.asarray(Image.open(folder/'camera/mask.png')) > 0, iterations=2) & (depth > 0)
    y, x = np.where(mask); z = depth[y, x]
    if len(z) < 150:
        raise ValueError('Insufficient original masked RGB-D')
    points = np.column_stack([(x-k['ppx'])*z/k['fx'], (y-k['ppy'])*z/k['fy'], z])
    camera = np.asarray(json.loads((folder/'camera_world_optical.json').read_text()))
    return points @ camera[:3, :3].T+camera[:3, 3]


def geometry(folder, fit, snapshot, output=None):
    """Read the ORIGINAL table mesh; no inflated plane or object placement."""
    folder = Path(folder)
    scene_folder = folder if (folder/'scene.yml').exists() else folder/'model_evidence'
    scene = yaml.safe_load((scene_folder/'scene.yml').read_text())
    sources = json.loads((scene_folder/'scene_sources.json').read_text())
    base = np.asarray(json.loads((scene_folder/'snapshot.json').read_text())['link_world']['body4'])
    center = np.asarray(fit['center_world'])
    triangles = []
    for key, data in scene['mesh'].items():
        if 'breakfast_table' not in sources[key]:
            continue
        path = scene_folder/'meshes'/Path(data['file_path']).name
        if not path.exists():
            raise ValueError('Missing archive-local original table geometry')
        mesh = trimesh.load(path, force='mesh', process=False)
        pose = np.asarray(data['pose'])
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(np.r_[pose[4:7], pose[3]]).as_matrix()
        transform[:3, 3] = pose[:3]
        world = base @ transform
        vertices = np.asarray(mesh.vertices) @ world[:3, :3].T+world[:3, 3]
        tri = vertices[np.asarray(mesh.faces)]
        cross = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        area2 = np.linalg.norm(cross, axis=1)
        horizontal = (np.abs(cross[:, 2]) > .98*area2) & (area2 > 1e-7)
        z = tri[:, :, 2].mean(1)
        horizontal &= (z < center[2]+.003) & (z > center[2]-.08)
        triangles.extend(tri[horizontal])
    if not triangles:
        raise ValueError('Original table surface not found')
    triangles = np.asarray(triangles)
    z = triangles[:, :, 2].mean(1)
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)/2
    levels = np.round(z, 4)
    broad = [level for level in np.unique(levels) if area[levels == level].sum() > .10]
    if not broad:
        raise ValueError('No broad unchanged table plane')
    top = max(broad)
    chosen = triangles[np.abs(z-top) < .001]
    xy = chosen[:, :, :2].reshape(-1, 2)
    hull = ConvexHull(xy)
    polygon = xy[hull.vertices]
    points = masked_points(folder)
    robot = np.asarray(snapshot['link_world']['body4'])[:2, 3]
    options = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = b-a
        if np.linalg.norm(edge) < .12:
            continue
        normal = np.array([edge[1], -edge[0]])/np.linalg.norm(edge)
        boundary = float(normal @ a)
        inside = boundary-float(normal @ center[:2])
        facing = float(normal @ (robot-center[:2]))
        support = float(np.quantile(points[:, :2] @ normal, .99)-center[:2] @ normal)
        stroke = inside-support+.030
        predicted_inside = inside-stroke
        if facing <= .02 or inside <= 0 or not .020 <= stroke <= .120 or predicted_inside < .020:
            continue
        options.append({'normal_xy': normal.tolist(), 'boundary_dot_m': boundary,
                        'stroke_m': stroke, 'initial_center_inside_m': inside,
                        'initial_observed_front_gap_m': inside-support,
                        'predicted_center_inside_m': predicted_inside,
                        'planned_observed_overhang_m': .030, 'facing_robot_dot_m': facing})
    if not options:
        raise ValueError('No supported edge-drag geometry under the declared range')
    plan = min(options, key=lambda x: x['stroke_m'])
    plan.update(table_top_z_m=float(np.average(z[np.abs(z-top) < .001], weights=area[np.abs(z-top) < .001])),
                table_polygon_world_xy=polygon.tolist(), fit_center_world=center.tolist(),
                scope='Geometry proposal only; no object move, IK, collision or physical drag proof')
    destination = Path(output) if output is not None else folder
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'drag_geometry.json').write_text(json.dumps(plan, indent=2))
    return plan


def optimize_contact_segments(planner, checker, route, names, limits, duration, out, label):
    """Optimize EVERY original PRM segment; retain contact waypoints, no fallback.

    All segment calls use unchanged optimize_path. A final independent dense
    check also covers the resampled joins and the public command trajectory.
    """
    import torch
    import time
    started = time.monotonic()
    waypoints = json.loads((out/'drag_waypoints.json').read_text())
    assert len(waypoints) >= 2 and waypoints[0]['start_row'] == 0
    assert waypoints[-1]['end_row'] == len(route)-1
    chunks = []; grids = []; elapsed = 0.; reports = []
    for index, point in enumerate(waypoints):
        lo, hi = point['start_row'], point['end_row']
        seed = route[lo:hi+1]
        assert len(seed) >= 2
        if index:
            assert lo == waypoints[index-1]['end_row']
        segment_label = f'{label}_segment_{index:02d}'
        q, times = optimize_path(planner, checker, seed, names, limits,
                                 duration/len(waypoints), out, segment_label)
        if index:
            assert float((q[0]-chunks[-1][-1]).abs().max()) <= 1e-5, 'Optimized segment join mismatch'
        chunks.append(q if index == 0 else q[1:])
        grids.append(times+elapsed if index == 0 else times[1:]+elapsed)
        elapsed += float(times[-1])
        reports.append(json.loads((out/f'{segment_label}_optimization.json').read_text()))
    joined = torch.cat(chunks)
    original_times = np.concatenate(grids)
    assert len(joined) == len(original_times) and (np.diff(original_times)>0).all()
    # Linear resampling preserves joint bounds; validate the resulting command
    # and all joins against the exact same original checker below.
    count = int(np.ceil(elapsed/.025))+1
    times = np.linspace(0., elapsed, count)
    positions = joined.detach().cpu().numpy()
    samples = np.column_stack([np.interp(times, original_times, positions[:, j]) for j in range(len(names))])
    q = torch.tensor(samples, dtype=route.dtype, device=route.device)
    violation = float(torch.maximum(limits[:,0]-q, q-limits[:,1]).clamp_min(0).max())
    assert torch.isfinite(q).all() and violation <= 1e-6, 'Segmented command exceeds unchanged bounds'
    a = torch.arange(10,device=q.device)/10
    dense = torch.cat([(q[:-1,None]*(1-a[None,:,None])+q[1:,None]*a[None,:,None]).reshape(-1,len(names)),q[-1:]])
    invalid = sum(int((~checker.validate(chunk[:,None])).sum()) for chunk in dense.split(64))
    assert invalid == 0, 'Segmented command failed original dense checks'
    # Same finite-difference derivative bounds used by the original optimizer.
    dt = float(times[1]-times[0]); derivative = q; scale = 1.; maxima = {}
    for order, bound in ((1,1.5),(2,5.),(3,100.)):
        derivative = torch.diff(derivative,dim=0)/dt
        peak = float(derivative.abs().max()) if derivative.numel() else 0.
        maxima[str(order)] = peak
        scale = max(scale,(peak/bound)**(1/order))
    times *= scale
    assert times[-1] <= 120, 'Segmented contact trajectory exceeds original 120s command cap'
    report = {'method':'Original full PRM+trajectory optimization for every Cartesian segment; joined/resampled and independently checked',
              'stage':label,'success':True,'segments':len(reports),'segment_reports':reports,
              'max_limit_exceed_rad':violation,'invalid_samples':invalid,'dense_samples':len(dense),
              'samples':len(q),'time_scale':scale,'execution_duration_s':float(times[-1]),
              'start_error_rad':float((q[0]-route[0]).abs().max()),
              'goal_error_rad':float((q[-1]-route[-1]).abs().max()),
              'native_finite_difference_maxima':maxima,'planning_wall_seconds':time.monotonic()-started,
              'physics_or_model_properties_changed':False}
    assert report['start_error_rad'] < .005 and report['goal_error_rad'] < .005
    (out/f'{label}_optimization.json').write_text(json.dumps(report,indent=2))
    (out/f'{label}_prm_seed.json').write_text(json.dumps(route.tolist()))
    return q, times


def execute_extra(label, q, world_goal, gripper, duration, *, planner, checker,
                  names, limits, out, client, target_path, optimize=True, line_info=None):
    """Same optimizer, fixed limits, API and settling rule as original actions."""
    import torch
    if not torch.isfinite(q).all() or not ((q >= limits[:, 0]) & (q <= limits[:, 1])).all():
        raise AssertionError('Preparation command outside unchanged arm bounds')
    if optimize:
        q, times = (optimize_contact_segments(planner, checker, q, names, limits, duration, out, label)
                    if label == "contact_drag" else optimize_path(planner, checker, q, names, limits, duration, out, label))
    else:
        assert len(q) == 2 and torch.equal(q[0], q[1]), 'Nonoptimized extra action must hold a constant arm target'
        assert checker.validate(q[:, None]).all(), 'Original open hold-state check failed'
        times = np.array([0., duration])
    if line_info is not None:
        state = planner.kinematics.compute_kinematics(JointState.from_position(q, joint_names=names))
        positions = state.tool_poses.position.reshape(-1, 3).detach().cpu().numpy()
        base = np.asarray(line_info['base_world'])
        positions = positions @ base[:3, :3].T+base[:3, 3]
        origin = np.asarray(line_info['origin_world'])
        normal = np.r_[line_info['normal_xy'], 0.]
        relative = positions-origin
        progress = relative @ normal
        off_line = relative-progress[:, None]*normal
        report = {'max_off_line_m': float(np.linalg.norm(off_line, axis=1).max()),
                  'min_progress_m': float(progress.min()), 'max_progress_m': float(progress.max()),
                  'tolerance_m': .002, 'scope': 'Additional planned contact-path quality gate; original model/constraints unchanged'}
        report['pass'] = bool(report['max_off_line_m'] <= .002 and progress.min() >= -.002
                              and progress.max() <= line_info['stroke_m']+.002)
        (out/f'{label}_cartesian_check.json').write_text(json.dumps(report, indent=2))
        assert report['pass'], 'Optimized drag left the proposed contact line'
    command = {'joint_names': names, 'position': q.tolist(), 'time_from_start': times.tolist(),
               'gripper_target': float(gripper), 'position_tolerance_rad': .005}
    (out/f'{label}_command.json').write_text(json.dumps(command))
    execution = client.request('trajectory', timeout=300, **command)
    (out/f'{label}_execution.json').write_text(json.dumps(execution))
    assessment = assess_settling(execution, world_goal)
    (out/f'{label}_settling.json').write_text(json.dumps(assessment, indent=2))
    external = [c for c in execution['contacts'] if c['num_contacts'] > 0
                and not ('/World/Robot/' in c['a'] and '/World/Robot/' in c['b'])]
    unwanted = [c for c in external if not (target_path in c['a']+c['b']
                and any(n in c['a']+c['b'] for n in ('right_link27', 'right_link28')))]
    assert not unwanted, f'Unwanted physical contact during {label}'
    assert assessment['accepted'], f'Original convergence gate failed during {label}'
    return execution


def line_route(ik, path, start, world, plan, tracking_valid, out):
    """Original IK+PRM at<=10mm Cartesian steps, original checks unchanged."""
    import torch
    normal = np.r_[plan['normal_xy'], 0.]
    count = int(np.ceil(plan['stroke_m']/.010))+1
    segments = []
    current = start
    goals = []
    offset = 0
    for distance in np.linspace(0., plan['stroke_m'], count):
        goal = np.asarray(world).copy()
        goal[:3, 3] += distance*normal
        q = ik(goal, current)
        if q is None:
            raise AssertionError('Original closed IK rejected a drag waypoint')
        segment = path(current, q)
        if segment is None:
            raise AssertionError('Original closed PRM/dense check rejected a drag segment')
        start_row = max(0, offset-1)
        part = segment if not segments else segment[1:]
        segments.append(part)
        offset += len(part)
        current = q
        goals.append({'distance_m': float(distance), 'world_goal': goal.tolist(), 'q': q.tolist(),
                      'start_row':start_row,'end_row':offset-1})
    assert tracking_valid(current, 'contact_drag_goal'), 'Original65state3mrad drag-end test failed'
    (out/'drag_waypoints.json').write_text(json.dumps(goals, indent=2))
    return torch.cat(segments), goal, current


def outcome(initial, released, fit, plan, target, folder, final_snapshot=None):
    """Measured preparation condition ONLY; never a grasp-success metric."""
    initial_world = np.asarray(initial['objects'][target]['world'])
    center = np.r_[fit['center_world'], 1.]
    local = np.linalg.inv(initial_world) @ center
    final = final_snapshot if final_snapshot is not None else released['final']
    end_world = np.asarray(final['objects'][target]['world'])
    after = (end_world @ local)[:3]
    before = center[:3]
    normal = np.asarray(plan['normal_xy'])
    along = float((after[:2]-before[:2]) @ normal)
    across = float(np.linalg.norm((after[:2]-before[:2])-along*normal))
    points = masked_points(folder)
    moved = (np.c_[points, np.ones(len(points))] @ (end_world @ np.linalg.inv(initial_world)).T)[:, :3]
    overhang = float(np.quantile(moved[:, :2] @ normal, .99)-plan['boundary_dot_m'])
    inside = float(plan['boundary_dot_m']-after[:2] @ normal)
    heights = [(np.asarray(r['objects'][target]) @ local)[2] for r in released['records']]
    passed = bool(along >= .8*plan['stroke_m'] and across <= .015 and overhang >= .020
                  and inside >= .015 and max(abs(np.asarray(heights)-before[2])) <= .005
                  and abs(after[2]-before[2]) <= .005)
    return {'preparation_condition_pass': passed, 'grasp_success_evaluated': False,
            'measured_along_displacement_m': along, 'measured_cross_displacement_m': across,
            'measured_observed_overhang_m': overhang, 'measured_center_inside_m': inside,
            'measured_center_vertical_delta_m': float(after[2]-before[2]),
            'scope': 'Actual rigid target motion and support estimate after release only; no completed grasp or lift claim'}
