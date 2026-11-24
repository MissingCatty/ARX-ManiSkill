"""CPU-only advisor35 diagnostics of immutable ARX records.

No simulator, service, neural inference, training, or physical command entry
points. Counterfactual gate outcomes are decisions, never grasp successes.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / 'runs/nearfield_advisor34_20261111_01'
TRAIN = ROOT / 'runs/nearfield_mask_repair_20261111_03/training'
AXES = ('dx_mm', 'dy_mm', 'dz_mm', 'rx_deg', 'ry_deg', 'rz_deg')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Evidence:
    def __init__(self):
        self.files = {}

    def pin(self, path, expected=None):
        p = Path(path).resolve()
        h = sha(p)
        if expected is not None and h != expected:
            raise ValueError('Evidence hash mismatch: ' + str(p))
        if str(p) in self.files and self.files[str(p)] != h:
            raise ValueError('Evidence changed during analysis: ' + str(p))
        self.files[str(p)] = h
        return p

    def read(self, path):
        return json.loads(self.pin(path).read_text())

    def lines(self, path):
        return [json.loads(s) for s in self.pin(path).read_text().splitlines()]

    def array(self, path):
        return np.load(self.pin(path), allow_pickle=False)


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n')


def csv_file(path, rows):
    if not rows:
        return
    with Path(path).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                        for k, v in r.items()})


def vector(text):
    p = text.split()
    if p == ['GRASP']:
        return None
    if len(p) != 7 or p[0] != 'REFINE':
        raise ValueError('Malformed registered D label: ' + text)
    a = np.asarray(list(map(float, p[1:])))
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite D label')
    return a


def stats(values, weights):
    a, w = np.asarray(values), np.asarray(weights)
    return {'mean': a.mean(0).tolist(), 'std_population': a.std(0).tolist(),
            'weighted_mean': np.average(a, axis=0, weights=w).tolist(),
            'negative': (a < 0).sum(0).tolist(), 'zero': (a == 0).sum(0).tolist(),
            'positive': (a > 0).sum(0).tolist(),
            'negative_beyond_0p5': (a < -.5).sum(0).tolist(),
            'within_0p5': (abs(a) <= .5).sum(0).tolist(),
            'positive_beyond_0p5': (a > .5).sum(0).tolist(),
            'min': a.min(0).tolist(), 'max': a.max(0).tolist()}


def training_distribution(ev, out):
    reg = ev.read(TRAIN / 'registration.json')
    samples = ev.lines(TRAIN / 'dataset/paired_samples.jsonl')
    model = ev.lines(TRAIN / 'model_rows/D/train.jsonl')
    train = [s for s in samples if s['split'] == 'train']
    assert len(train) == len(model) == 435
    byid = {s['id']: s for s in train}
    assert set(byid) == {r['sample_id'] for r in model}
    for p in (TRAIN / 'dataset/paired_samples.jsonl', TRAIN / 'model_rows/D/train.jsonl'):
        ev.pin(p, reg['data_files'][str(p)])
    grouped = defaultdict(list)
    lookup = defaultdict(list)
    for row in model:
        s = byid[row['sample_id']]
        assert row['output'] == s['labels']['D']
        grouped[s['object_id']].append((s, row))
        lookup[row['output']].append({'object_id': s['object_id'], 'sample_id': s['id'],
            'family': s['family'], 'group': s['group'], 'sample_weight': row['sample_weight']})
    results, axes, phases = [], [], []
    for obj, pairs in sorted(grouped.items()):
        motion = [(s, r) for s, r in pairs if vector(r['output']) is not None]
        a = [vector(r['output']) for s, r in motion]
        w = [r['sample_weight'] for s, r in motion]
        dist = stats(a, w)
        record = {'object_id': obj, 'total': len(pairs), 'motion': len(motion),
                  'GRASP': len(pairs)-len(motion), 'families': len({s['family'] for s, r in pairs}),
                  'episodes': len({s['episode_id'] for s, r in pairs}), 'axes': AXES, **dist}
        results.append(record)
        for j, axis in enumerate(AXES):
            axes.append({'object_id': obj, 'axis': axis, 'motion_labels': len(motion),
                         **{k: v[j] for k, v in dist.items()}})
        for group in sorted({s['group'] for s, r in motion}):
            subset = [(s, r) for s, r in motion if s['group'] == group]
            phases.append({'object_id': obj, 'group': group, 'motion_labels': len(subset),
                **stats([vector(r['output']) for s, r in subset], [r['sample_weight'] for s, r in subset])})
    log = ev.lines(TRAIN / 'D/training.jsonl')
    aug = ev.read(TRAIN / 'rgb_augmentation.json')
    variants = [v for sample in aug.values() for v in sample]
    timing = {'updates': len(log), 'samples_seen': log[-1]['samples_seen'],
              'training_seconds': log[-1]['elapsed_seconds'],
              'steady_update_seconds_median': float(np.median([r['update_seconds'] for r in log[10:]])),
              'registration_to_finished_seconds': ev.read(TRAIN / 'D/state.json')['finished_unix']-reg['created_unix']}
    result = {'train': 435, 'motion': sum(r['motion'] for r in results),
              'GRASP': sum(r['GRASP'] for r in results), 'objects': results,
              'phase_distributions': phases, 'denominator': 'Motion labels only; GRASP is a separate class, not a zero vector.',
              'weighted_mean': 'Uses actual registered sample_weight; sequence cross-entropy is not an L2 mean regressor.',
              'rgb_augmentation': reg['rgb_augmentation'],
              'augmentation_views': len(variants), 'grayscale_views': sum(v['grayscale'] for v in variants),
              'timing': timing,
              'training_configuration': {k: reg.get(k) for k in ('model', 'epochs', 'microbatch',
                 'gradient_accumulation', 'learning_rate', 'projector_learning_rate', 'optimizer',
                 'lora', 'trainable_scope', 'initialization', 'phase_weight_multipliers')},
              'train_object_ids': sorted(grouped),
              'validation_objects': dict(Counter(s['object_id'] for s in samples if s['split']=='validation'))}
    dump(out/'training_distribution.json', result)
    csv_file(out/'training_axes.csv', axes)
    csv_file(out/'training_phases.csv', phases)
    return result, lookup, samples


def requests(ev, ep):
    rows = []
    for entry in ev.read(ep/'policy_requests.json'):
        p = Path(entry['request']); req = ev.read(p)
        reply = ev.read(p.with_name(p.name.replace('_request.json', '_reply.json')))
        ev.pin(p, reply['request_sha256'])
        rows.append((p, req, reply))
    return sorted(rows, key=lambda x: x[1]['raw_index'])


def image_comparison(ev, out, trials, lookup):
    results = []
    for trial, ep in trials:
        if trial['phase'] != 'target_shift' or trial['vision_arm'] != 'fresh':
            continue
        first = [r for r in requests(ev, ep) if r[1]['motions_completed'] == 0 and r[2]['model_inference']]
        assert len(first) == 1
        p, req, rep = first[0]
        channels = req['native_mask_binding']['channels']
        target = ev.array(channels['target']).astype(bool)
        valid = ev.array(channels['depth_valid']).astype(bool)
        depth = ev.array(req['depth'])
        mask = target & valid & np.isfinite(depth) & (depth > 0)
        assert mask.sum() == rep['finger_box']['target_points']
        yy, xx = np.nonzero(mask)
        images = rep['visual_control']['images']
        hashes = []
        for image in images:
            ev.pin(image, rep['files'][image]); hashes.append(sha(image))
        for f, h in req['native_mask_binding']['files'].items():
            ev.pin(f, h)
        inp = ev.read(p.with_name(req['request_id']+'_input.json'))
        results.append({'trial_id': trial['trial_id'], 'object_id': trial['object_id'],
            'repeat': trial['repeat'], 'shift_mm': trial['target_delta_world_m'][0]*1000,
            'request_id': req['request_id'], 'raw_index': req['raw_index'],
            'input_images': images, 'rgb_sha256': hashes[0], 'depth_image_sha256': hashes[1],
            'target_mask_sha256': sha(channels['target']), 'target_pixels': int(mask.sum()),
            'centroid_u': float(xx.mean()), 'centroid_v': float(yy.mean()),
            'image_size_wh': list(Image.open(images[0]).size),
            'instruction_sha256': hashlib.sha256(inp['instruction'].encode()).hexdigest(),
            'input_token_sha256': rep['input_token_sha256'], 'prediction': rep['prediction'],
            'exact_training_label_matches': lookup.get(rep['prediction'], []),
            'target_mask_file': channels['target'], 'request_file': str(p),
            'new_neural_inferences': 0})
    assert len(results) == 8
    pairs = []
    for obj in sorted({r['object_id'] for r in results}):
        for repeat in (1, 2):
            plus = next(r for r in results if r['object_id']==obj and r['repeat']==repeat and r['shift_mm']>0)
            minus = next(r for r in results if r['object_id']==obj and r['repeat']==repeat and r['shift_mm']<0)
            a, b = [ev.array(r['target_mask_file']).astype(bool) for r in (plus, minus)]
            pairs.append({'object_id': obj, 'repeat': repeat,
                'different_RGB': plus['rgb_sha256'] != minus['rgb_sha256'],
                'different_depth_image': plus['depth_image_sha256'] != minus['depth_image_sha256'],
                'same_instruction': plus['instruction_sha256'] == minus['instruction_sha256'],
                'same_raw_output': plus['prediction'] == minus['prediction'],
                'centroid_plus_minus_du_px': plus['centroid_u']-minus['centroid_u'],
                'centroid_plus_minus_dv_px': plus['centroid_v']-minus['centroid_v'],
                'mask_iou': float((a & b).sum()/(a | b).sum()),
                'RGB_MAE_0_255': float(np.mean(abs(np.asarray(Image.open(plus['input_images'][0]), dtype=float)-
                                              np.asarray(Image.open(minus['input_images'][0]), dtype=float))))})
    dump(out/'first_inputs.json', {'requests': results, 'paired_comparisons': pairs})
    csv_file(out/'first_inputs.csv', results); csv_file(out/'first_input_pairs.csv', pairs)
    return results, pairs


def observed_points(ev, req):
    from .advisor32_rules import backproject, points_in_frame
    row = ev.read(req['observation'])
    depth = ev.array(req['depth']); ch = req['native_mask_binding']['channels']
    mask = ev.array(ch['target']).astype(bool) & ev.array(ch['depth_valid']).astype(bool)
    mask &= np.isfinite(depth) & (depth > 0)
    p = backproject(depth, mask, row['intrinsics'])
    return row, points_in_frame(p, row['camera_world_optical'], row['snapshot']['link_world']['right_eef_link'])


def point_metrics(points, gate):
    from .advisor33_geometry import corridor
    box = gate['box']; cross = points[corridor(points, box)]
    inside = points[corridor(points, box, include_z=True)]
    return {'target_points': len(points), 'corridor_points': len(cross), 'inside_points': len(inside),
            'corridor_z_mm_q05_q50_q95': (np.percentile(cross[:, 2], [5,50,95])*1000).tolist() if len(cross) else None,
            'target_y_mm_q05_q50_q95': (np.percentile(points[:, 1], [5,50,95])*1000).tolist(),
            'surface_z_mm': None if gate['near_surface']['median_z_EE_m'] is None else gate['near_surface']['median_z_EE_m']*1000,
            'original_visual_gate_pass': gate['pass'], 'direct_visible_pass': gate['direct_visible_pass'],
            'gravity_column_pass': gate['occlusion_envelope']['pass'],
            'gate_origin': gate['gate_origin']}


def target_contacts(execution, object_id):
    contacts = {k: [] for k in ('right_link27', 'right_link28')}
    for c in execution['contacts']:
        paths = (c['a'], c['b'])
        if not any(p.endswith('/asset_'+object_id) for p in paths):
            continue
        for finger in contacts:
            if any(p.endswith('/'+finger) for p in paths):
                contacts[finger].append(c)
    return {k: {'callbacks': len(v), 'unique_sim_times': len({c['sim_time'] for c in v}),
                'first_time': min((c['sim_time'] for c in v), default=None),
                'last_time': max((c['sim_time'] for c in v), default=None),
                'types': dict(Counter(c['type'] for c in v)),
                'force_N': None, 'force_measured': False}
            for k, v in contacts.items()}


def subtract_gate(gate):
    """Remove only advisor34's extra axial move and 36.5 mm rejection.

    A visual pass merely requests the existing independent closed-model check.
    That check was not run at the counterfactual pre-compensation pose.
    """
    return {'next': 'independent_closed_model_check' if gate['pass'] else 'visual_gate_reject',
            'extra_axial_moves': 0, 'post_compensation_36p5_rejection': False,
            'closed_collision_check_pass': None, 'physical_grasp_success': None}


def axial_review(ev, out, trials, outcomes):
    results = []
    for trial, ep in trials:
        path = ep/'axial_align_001_execution.json'
        if not path.exists():
            continue
        rows = requests(ev, ep)
        candidates = [(i, r) for i, r in enumerate(rows) if r[2]['action'].get('preclosure_axial_correction')]
        assert len(candidates) == 1
        i, (_, before_req, before_rep) = candidates[0]
        assert i+1 < len(rows)
        _, after_req, after_rep = rows[i+1]
        before, p0 = observed_points(ev, before_req); after, p1 = observed_points(ev, after_req)
        before_stats = point_metrics(p0, before_rep['finger_box']); after_stats = point_metrics(p1, after_rep['finger_box'])
        for metric, rep in ((before_stats,before_rep), (after_stats,after_rep)):
            assert metric['corridor_points'] == rep['finger_box']['corridor_points']
            assert metric['inside_points'] == rep['finger_box']['in_hand_points']
        ex = ev.read(path)
        previous = before['teacher_command']['label']
        previous_path = ep/(previous+'_execution.json')
        prev = ev.read(previous_path) if previous_path.exists() else None
        e0 = np.asarray(before['snapshot']['link_world']['right_eef_link']);e1=np.asarray(after['snapshot']['link_world']['right_eef_link'])
        o0 = np.asarray(before['snapshot']['objects']['test_object']['world']);o1=np.asarray(after['snapshot']['objects']['test_object']['world'])
        results.append({'trial_id': trial['trial_id'], 'object_id': trial['object_id'],
            'arm': trial.get('comparison_arm') or trial['vision_arm'], 'repeat': trial.get('repeat'),
            'outcome': outcomes[trial['trial_id']]['outcome'],
            'among_nine_axial_rejections': outcomes[trial['trial_id']]['outcome']=='axial_band_not_reached_after_one_compensation',
            'before_frame': before_req['raw_index'], 'after_frame': after_req['raw_index'],
            'before': before_stats, 'after': after_stats,
            'before_motion_label': previous,
            'before_motion_contacts': target_contacts(prev,trial['object_id']) if prev else None,
            'axial_motion_contacts': target_contacts(ex,trial['object_id']),
            'actual_EE_displacement_local_mm': (e0[:3,:3].T@(e1[:3,3]-e0[:3,3])*1000).tolist(),
            'actual_object_displacement_local_mm': (e0[:3,:3].T@(o1[:3,3]-o0[:3,3])*1000).tolist(),
            'object_translation_mm': float(np.linalg.norm(o1[:3,3]-o0[:3,3])*1000),
            'object_rotation_degrees': float(Rotation.from_matrix(o0[:3,:3].T@o1[:3,:3]).magnitude()*180/np.pi),
            'counterfactual_at_before': subtract_gate(before_rep['finger_box']),
            'counterfactual_at_after_observed_state': subtract_gate(after_rep['finger_box']),
            'counterfactual_scope': 'Only gate replay at recorded states; no unexecuted trajectory or grasp outcome inferred.',
            'force_evidence': 'Only bodies, event type, contact count and sim_time were saved; forces/impulses absent.'})
    assert len(results)==12
    assert sum(r['among_nine_axial_rejections'] for r in results)==9
    dump(out/'axial_review.json',results)
    csv_file(out/'axial_review.csv',results)
    return results


def mesh_point_transform(points, pose):
    p = np.asarray(pose)
    return np.asarray(points)@p[:3,:3].T+p[:3,3]


def geometry_review(ev, out, reg, samples, trials):
    import trimesh
    from scripts.curobo_arx.sdg_stage1.sampler import composed_mesh
    from training.nearfield_arx.capture_review import finger_geometry
    from .advisor33_geometry import finger_box
    from .advisor32_rules import points_in_frame
    robot=reg['learned_policy']['robot_usd'];ev.pin(robot)
    fingers=finger_geometry(Path(robot));projector=SimpleNamespace(fingers=fingers)
    meshes={};mesh_evidence={}
    def mesh_for(obj):
        if obj not in meshes:
            asset=next(o for o in reg['catalog']['objects'] if o['object_id']==obj)
            meshes[obj],mesh_evidence[obj]=composed_mesh(asset['usd_path'])
            for p,h in mesh_evidence[obj]['used_layers'].items():ev.pin(p,h)
        return meshes[obj]
    obbs=[]
    for obj in sorted({s['object_id'] for s in samples if s['split']=='train'}):
        s=next(s for s in samples if s['split']=='train' and s['object_id']==obj)
        ep=Path(s['inputs']['rgb']).parent.parent
        rs=ev.lines(ep/'samples.jsonl'); row=rs[s['raw_index']]
        world=np.asarray(row['snapshot']['objects']['test_object']['world'])
        star=np.asarray(s['offline_target']['Gstar_world'])
        mesh=mesh_for(obj);to_box,extents=trimesh.bounds.oriented_bounds(mesh,angle_digits=1)
        box=world@np.linalg.inv(to_box);vertical=int(np.argmax(abs(box[2,:3])))
        horizontal=[i for i in range(3) if i!=vertical];short=min(horizontal,key=lambda i:extents[i])
        angles=np.rad2deg(np.arccos(np.clip(abs(box[:3,:3].T@star[:3,1]),0,1)))
        obbs.append({'object_id':obj,'source_sample':s['id'],'extents_mm':(extents*1000).tolist(),
            'vertical_axis':vertical,'short_horizontal_axis':short,
            'closing_to_OBB_normals_degrees':angles.tolist(),'closing_to_short_horizontal_normal_degrees':float(angles[short]),
            'short_horizontal_width_mm':float(extents[short]*1000),
            'scope':'Visible-mesh OBB diagnostic only; not local contact-normal or force-closure certification.',
            'Gstar_world':star.tolist(),'object_world':world.tolist(),'OBB_world':box.tolist(),
            'mesh_evidence':mesh_evidence[obj]})
    histories=[];summaries=[]
    for trial,ep in trials:
        if trial['vision_arm']!='fresh' or trial['method']!='D':continue
        first_path=ep/'d_refine_001_execution.json'
        if not first_path.exists():continue
        ex=ev.read(first_path);rows=ev.lines(ep/'samples.jsonl')
        rq=next(r for r in requests(ev,ep) if r[1]['motions_completed']==0 and r[2]['model_inference'])
        first_row=ev.read(rq[1]['observation']);t0=first_row['snapshot']['sim_time']
        motion=[(i,r) for i,r in enumerate(rows) if (r.get('teacher_command') or {}).get('label')=='d_refine_001'
                and r['snapshot']['sim_time']>=t0]
        mesh=mesh_for(trial['object_id']);hist=[]
        for index,row in [(rq[1]['raw_index'],first_row),*motion]:
            snapshot=row['snapshot'];ee=np.asarray(snapshot['link_world']['right_eef_link'])
            world=mesh_point_transform(mesh.vertices,snapshot['objects']['test_object']['world'])
            top=float(world[:,2].max());box=finger_box(projector,row)
            pts=points_in_frame(world,np.eye(4),ee)
            # Conservative whole-object support interval; a local section can be narrower.
            lo,hi=float(pts[:,1].min()),float(pts[:,1].max())
            clear=min(lo-box['y_EE_m'][0],box['y_EE_m'][1]-hi)
            tips=[]
            for name,(v,f) in fingers.items():
                local=points_in_frame(v,snapshot['link_world'][name],ee)
                distal=v[local[:,2]>.015]
                tips.append(float(mesh_point_transform(distal,snapshot['link_world'][name])[:,2].min()))
            gap=min(tips)-top
            h={'trial_id':trial['trial_id'],'raw_index':index,'sim_time':snapshot['sim_time'],
                'seconds_after_H_observation':snapshot['sim_time']-t0,'tip_to_object_top_mm':gap*1000,
                'object_y_support_mm':[lo*1000,hi*1000],
                'open_pad_y_mm':[v*1000 for v in box['y_EE_m']],
                'whole_object_lateral_clearance_mm':clear*1000,
                'below_top_and_lateral_overlap':gap<=0 and clear<0}
            hist.append(h)
        contacts=target_contacts(ex,trial['object_id'])
        times=[c['first_time'] for c in contacts.values() if c['first_time'] is not None]
        first_contact=min(times) if times else None
        at_contact=min(hist,key=lambda h:abs(h['sim_time']-first_contact)) if first_contact is not None else None
        crosses=[h for h in hist if h['tip_to_object_top_mm']<=0]
        aligned=[h for h in hist if h['whole_object_lateral_clearance_mm']>=0]
        summaries.append({'trial_id':trial['trial_id'],'object_id':trial['object_id'],
            'first_top_layer_frame':crosses[0] if crosses else None,
            'first_whole_object_between_jaws_frame':aligned[0] if aligned else None,
            'first_contact_seconds_after_H':first_contact-t0 if first_contact is not None else None,
            'nearest_saved_frame_to_first_contact':at_contact,
            'first_motion_contacts':contacts,
            'overlap_proxy_frames':sum(h['below_top_and_lateral_overlap'] for h in hist),
            'recorded_frames':len(hist),
            'scope':'5Hz visual-mesh top-plane/support-interval proxy with measured finger transforms; actual contacts separate; no continuous collision guarantee.'})
        histories.extend(hist)
    dump(out/'canonical_obb_review.json',obbs)
    dump(out/'first_motion_geometry.json',summaries)
    csv_file(out/'first_motion_frames.csv',histories)
    return obbs,summaries


def main(out):
    out=Path(out).resolve()
    if (out/'diagnosis.json').exists():raise FileExistsError('Preserve completed diagnosis')
    out.mkdir(parents=True,exist_ok=True)
    ev=Evidence();reg=ev.read(OLD/'registration.json')
    ev.pin(__file__)
    ev.pin(ROOT/'research_plan/chat/gemini回复/gemini回复35.md')
    ev.read(OLD/'state.json');ev.read(OLD/'sources.json');ev.pin(OLD/'research.md');ev.pin(OLD/'research_log.md')
    outcomes={};trials=[]
    for trial in reg['seed_trials']:
        p=list(OLD.glob('groups/*/trials/'+trial['trial_id']+'/trial_result.json'))
        assert len(p)==1
        result=ev.read(p[0]);outcomes[trial['trial_id']]=result
        assert len(result['attempts'])==1
        trials.append((trial,Path(result['attempts'][0]['episode'])))
    dist,lookup,samples=training_distribution(ev,out)
    inputs,pairs=image_comparison(ev,out,trials,lookup)
    axial=axial_review(ev,out,trials,outcomes)
    obbs,motions=geometry_review(ev,out,reg,samples,trials)
    result={'completed_utc':datetime.now(timezone.utc).isoformat(),
        'physical_trials':0,'neural_inferences':0,'training_updates':0,
        'existing_completed_trials_read':len(trials),'first_requests_compared':len(inputs),
        'pairs':pairs,'training_distribution':dist,
        'axial_compensations_reviewed':len(axial),'nine_rejections_reviewed':sum(r['among_nine_axial_rejections'] for r in axial),
        'canonical_objects_reviewed':len(obbs),'first_motions_reviewed':len(motions),
        'historical_results_changed':False,'runtime_policy_changed':False,
        'evidence_files':ev.files}
    # Detect changes even if another process writes an input while this CPU job runs.
    for p,h in list(ev.files.items()):ev.pin(p,h)
    dump(out/'diagnosis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','training_distribution','evidence_files')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    main(parser.parse_args().out)
