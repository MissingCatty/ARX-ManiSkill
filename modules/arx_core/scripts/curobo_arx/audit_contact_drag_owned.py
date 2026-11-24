"""Independent raw-record audit for a preparation-only ARX contact-drag test."""
import json
from pathlib import Path
import numpy as np
import trimesh
import yaml
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation
from settling import assess_settling
from research_guard import invariant_digest


def audit(folder, expected_canonical):
    folder = Path(folder)
    read = lambda n: json.loads((folder/n).read_text())
    guard = []; metric = []; stages = []
    initial = read('capture_snapshot.json')
    names = initial['right_arm']
    limits = np.asarray([initial['joint_limits'][initial['joint_names'].index(n)] for n in names])
    for file in ('initial_fingerprint.json', 'final_fingerprint.json'):
        if not (folder/file).exists() or invariant_digest(read(file)) != expected_canonical:
            guard.append('Missing or changed original configuration: '+file)
    selected = read('selected_grasp.json') if (folder/'selected_grasp.json').exists() else None
    plan = read('drag_geometry.json') if (folder/'drag_geometry.json').exists() else None
    actions = ['pregrasp','approach','close','contact_drag','drag_release','drag_retreat']
    expected_gripper = [-3.2,-3.2,.1,.1,-3.2,-3.2]
    target = read('scenario.json').get('target','test_object')
    target_path = initial['objects'][target]['path']
    for label, gripper in zip(actions, expected_gripper):
        command_file = folder/f'{label}_command.json'
        execution_file = folder/f'{label}_execution.json'
        if not command_file.exists() and not execution_file.exists():
            metric.append('Unexecuted preparation action: '+label); continue
        if not command_file.exists() or not execution_file.exists():
            guard.append('Incomplete issued-action evidence: '+label); continue
        try:
            command = read(command_file.name); execution = read(execution_file.name)
            q = np.asarray(command['position']); times = np.asarray(command['time_from_start'])
            if command['joint_names'] != names or q.ndim != 2 or q.shape[1] != len(names) or len(q) != len(times):
                raise ValueError('Wrong ARX joint order or command shape')
            if not np.isfinite(q).all() or not np.isfinite(times).all() or len(times) < 2 or times[0] != 0 or not (np.diff(times) > 0).all():
                raise ValueError('Nonfinite or unordered command')
            exceed = float(np.maximum(limits[:,0]-q,q-limits[:,1]).clip(0).max())
            if exceed > 1e-6:
                guard.append('Command exceeds unchanged arm bounds: '+label)
            if abs(command['gripper_target']-gripper) > 1e-12 or command.get('position_tolerance_rad') != .005:
                guard.append('Changed original command/API tolerance: '+label)
            targets = np.asarray([r['target'] for r in execution['records']])
            if targets.ndim != 2 or targets.shape[1] != len(names) or not np.isfinite(targets).all():
                raise ValueError('Missing recorded actual targets')
            if np.maximum(limits[:,0]-targets,targets-limits[:,1]).clip(0).max() > 1e-6:
                guard.append('Recorded target exceeds original bounds: '+label)
            if ((targets < q.min(0)-2e-6) | (targets > q.max(0)+2e-6)).any():
                guard.append('Recorded targets outside the issued command range: '+label)
            if label == 'drag_release':
                if q.shape != (2,len(names)) or not np.array_equal(q[0],q[1]) or not np.allclose(targets,q[0],atol=1e-7,rtol=0):
                    guard.append('Release must hold a constant accepted arm target')
                previous = read('contact_drag_execution.json')['records'][-1]['target'] if (folder/'contact_drag_execution.json').exists() else None
                if previous is not None and not np.array_equal(q[0],np.asarray(previous,dtype=np.float32)):
                    guard.append('Release changed the last drag arm command')
                final = execution['final']
                opened = [final['joint_positions'][final['joint_names'].index(n)] for n in ('right_joint27','right_joint28')]
                if max(abs(v-.044) for v in opened) > .0001:
                    metric.append('Release did not reach the original full-open state')
            if label in ('pregrasp','approach','contact_drag','drag_retreat'):
                opt = read(label+'_optimization.json')
                if not opt.get('success') or opt['max_limit_exceed_rad'] > 1e-6 or opt['invalid_samples'] != 0:
                    guard.append('Original optimization/dense checks failed: '+label)
            external = [c for c in execution['contacts'] if c['num_contacts'] > 0
                        and not ('/World/Robot/' in c['a'] and '/World/Robot/' in c['b'])]
            bad = [c for c in external if not (target_path in c['a']+c['b']
                   and any(n in c['a']+c['b'] for n in ('right_link27','right_link28')))]
            if bad:
                guard.append('Unwanted physical contact: '+label)
            if selected is None or plan is None:
                raise ValueError('Missing selected contact/geometry')
            if label == 'pregrasp':
                world = np.asarray(selected['pregrasp_world'])
            elif label in ('approach','close'):
                world = np.asarray(selected['grasp_world'])
            else:
                world = np.asarray(selected['grasp_world']).copy()
                world[:3,3] += plan['stroke_m']*np.r_[plan['normal_xy'],0.]
                if label == 'drag_retreat':world[2,3] += .05
            settling = assess_settling(execution,world)
            if not settling['accepted']:metric.append('Original convergence gate failed: '+label)
            if label == 'close':
                fingers = {n for c in external for n in ('right_link27','right_link28') if n in c['a']+c['b']}
                if len(fingers) != 2:metric.append('No original bilateral contact during acquisition')
            if label == 'contact_drag':
                check = read('contact_drag_cartesian_check.json')
                if not check.get('pass') or check['max_off_line_m'] > .002:
                    guard.append('Optimized contact path left the declared line')
            stages.append({'stage':label,'max_planned_limit_exceed_rad':exceed,
                           'unwanted_contacts':len(bad),'convergence':settling})
        except Exception as exc:
            guard.append(label+': '+repr(exc))
    if (folder/'contact_drag_command.json').exists():
        try:
            waypoints=read('drag_waypoints.json');parts=[];times_parts=[];seed_parts=[];elapsed=0.;previous=None
            integrity=read('contact_drag_original_route_integrity.json')
            if not integrity.get('unchanged') or integrity['max_route_change_rad']!=0:
                guard.append('Original PRM route changed during segment optimization')
            for index,point in enumerate(waypoints):
                label=f'contact_drag_segment_{index:02d}'
                raw=read(label+'_optimized_positions.json');q=np.asarray(raw['position']);t=np.asarray(raw['time_from_start']);seed=np.asarray(read(label+'_prm_seed.json'));opt=read(label+'_optimization.json')
                if q.ndim!=2 or q.shape[1]!=len(names) or len(q)!=len(t) or len(t)<2 or not np.isfinite(q).all() or not np.isfinite(t).all() or t[0]!=0 or not (np.diff(t)>0).all():
                    raise ValueError('Missing/nonfinite optimized segment command evidence')
                if np.maximum(limits[:,0]-q,q-limits[:,1]).clip(0).max()>1e-6:
                    guard.append('Optimized segment exceeds original bounds: '+label)
                if not opt['success'] or opt['max_limit_exceed_rad']>1e-6 or opt['invalid_samples']!=0:
                    guard.append('Segment full original optimization/dense check failed: '+label)
                if np.max(np.abs(q[0]-seed[0]))>=.005 or np.max(np.abs(q[-1]-seed[-1]))>=.005 or np.max(np.abs(seed[-1]-point['q']))>1e-6:
                    guard.append('Optimized segment endpoint does not match original PRM/waypoint: '+label)
                if previous is not None and np.max(np.abs(q[0]-previous))>1e-5:
                    guard.append('Optimized segment join mismatch: '+label)
                previous=q[-1]
                parts.append(q if index==0 else q[1:]);times_parts.append(t+elapsed if index==0 else t[1:]+elapsed)
                seed_parts.append(seed if index==0 else seed[1:]);elapsed+=float(t[-1])
            joined=np.concatenate(parts);original_times=np.concatenate(times_parts);original_seed=np.concatenate(seed_parts)
            if not np.array_equal(original_seed,np.asarray(read('contact_drag_prm_seed.json'))):
                guard.append('Final original PRM evidence differs from original per-segment seeds')
            grid=np.linspace(0.,elapsed,int(np.ceil(elapsed/.025))+1)
            expected=np.column_stack([np.interp(grid,original_times,joined[:,j]) for j in range(len(names))])
            command=read('contact_drag_command.json');actual=np.asarray(command['position']);actual_times=np.asarray(command['time_from_start']);opt=read('contact_drag_optimization.json')
            if expected.shape!=actual.shape or not np.allclose(expected,actual,atol=2e-6,rtol=0):
                guard.append('Actual drag command differs from independently joined optimized segments')
            if not np.allclose(actual_times,grid*opt['time_scale'],atol=1e-8,rtol=0) or actual_times[-1]>120:
                guard.append('Actual drag timing differs from declared stretching/original120s cap')
            # Position rounding on the final float32 command was already included
            # in the controller's original finite-difference time stretch.
            derivative=actual;dt=float(actual_times[1]-actual_times[0])
            for order,bound in ((1,1.5),(2,5.),(3,100.)):
                derivative=np.diff(derivative,axis=0)/dt
                if derivative.size and np.max(np.abs(derivative))>bound*(1+1e-5):
                    guard.append('Actual joined command exceeds original derivative playback bound: '+str(order))
        except Exception as exc:
            guard.append('Independent segment ownership/join evidence: '+repr(exc))
    measured = None
    if plan is not None and (folder/'preparation_final_snapshot.json').exists() and (folder/'drag_release_execution.json').exists():
        try:
            fit = read('thin_fit.json'); final = read('preparation_final_snapshot.json')
            original_world = np.asarray(initial['objects'][target]['world'])
            end_world = np.asarray(final['objects'][target]['world'])
            center = np.r_[fit['center_world'],1.]; local = np.linalg.inv(original_world) @ center
            after = (end_world @ local)[:3]; before = center[:3]; normal = np.asarray(plan['normal_xy'])
            along = float((after[:2]-before[:2]) @ normal)
            across = float(np.linalg.norm(after[:2]-before[:2]-along*normal))
            params = read('camera/params.json'); k = params['intrinsics']
            depth = np.asarray(Image.open(folder/'camera/depth.png'),dtype=float)*params['depth_scale']
            mask = binary_erosion(np.asarray(Image.open(folder/'camera/mask.png'))>0,iterations=2) & (depth>0)
            yy,xx = np.where(mask); zz = depth[yy,xx]
            points = np.c_[(xx-k['ppx'])*zz/k['fx'],(yy-k['ppy'])*zz/k['fy'],zz]
            camera = np.asarray(read('camera_world_optical.json'))
            points = points @ camera[:3,:3].T+camera[:3,3]
            transformed = np.c_[points,np.ones(len(points))] @ (end_world @ np.linalg.inv(original_world)).T
            overhang = float(np.quantile(transformed[:,:2] @ normal,.99)-plan['boundary_dot_m'])
            inside = float(plan['boundary_dot_m']-after[:2] @ normal)
            release = read('drag_release_execution.json')
            heights = [(np.asarray(r['objects'][target]) @ local)[2] for r in release['records']]
            height_error = float(np.abs(np.asarray(heights)-before[2]).max())
            if not .020 <= plan['stroke_m'] <= .120 or abs(np.linalg.norm(normal)-1) > 1e-6 or abs(plan['planned_observed_overhang_m']-.030)>1e-9:
                guard.append('Changed declared geometric preparation range')
            if abs(plan['initial_center_inside_m']-(plan['boundary_dot_m']-before[:2] @ normal)) > 1e-6 or plan['predicted_center_inside_m'] < .020:
                guard.append('Invalid original support forecast')
            if abs(np.quantile(points[:,:2] @ normal,.99)+plan['stroke_m']-plan['boundary_dot_m']-.030) > 1e-6:
                guard.append('Stroke does not match original observed front/declared overhang')
            scene_root = folder/'model_evidence'
            scene = yaml.safe_load((scene_root/'scene.yml').read_text())
            sources = json.loads((scene_root/'scene_sources.json').read_text())
            base = np.asarray(json.loads((scene_root/'snapshot.json').read_text())['link_world']['body4'])
            table = []
            for key,data in scene['mesh'].items():
                if 'breakfast_table' not in sources[key]:continue
                mesh = trimesh.load(scene_root/'meshes'/Path(data['file_path']).name,force='mesh',process=False)
                pose = data['pose']; rotation = Rotation.from_quat([*pose[4:7],pose[3]]).as_matrix()
                vertices = np.asarray(mesh.vertices) @ rotation.T+pose[:3]
                table.extend(vertices @ base[:3,:3].T+base[:3,3])
            table = np.asarray(table)
            allowed = table[table[:,2] <= before[2]+.003]
            actual_top = float(allowed[:,2].max())
            top = allowed[np.abs(allowed[:,2]-actual_top)<.001]
            if abs(actual_top-plan['table_top_z_m']) > .002 or abs(float((top[:,:2] @ normal).max())-plan['boundary_dot_m']) > .002:
                guard.append('Proposed edge does not match unchanged archived table geometry')
            passed = bool(along >= .8*plan['stroke_m'] and across <= .015 and overhang >= .020 and inside >= .015 and height_error <= .005 and abs(after[2]-before[2]) <= .005)
            measured = {'preparation_condition_pass':passed,'along_m':along,'across_m':across,
                        'observed_overhang_m':overhang,'center_inside_m':inside,'release_max_vertical_change_m':height_error}
            if not passed:metric.append('Measured preparation displacement/support condition failed')
            controller = read('preparation_report.json')
            if controller.get('status') == 'completed' and controller.get('preparation_condition_pass') is not passed:
                guard.append('Controller preparation report disagrees with raw measurements')
        except Exception as exc:
            guard.append('Measured condition evidence: '+repr(exc))
    else:
        metric.append('Preparation did not reach a measured release/final state')
    result = {'preparation_pass':not guard and not metric,'guard_pass':not guard,
              'guard_violations':guard,'metric_failures':metric,'stages':stages,'measured':measured,
              'grasp_success_evaluated':False,'scope':'Independent extra-action/preparation diagnostic only. Original five-stage grasp evaluator unchanged and no completed grasp claimed.'}
    (folder/'preparation_independent_evaluation.json').write_text(json.dumps(result,indent=2))
    return result
