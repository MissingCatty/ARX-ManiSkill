"""Draw saved GraspGenX poses on their exact observed object input cloud."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_hex
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from graspgenx_partial_geometry import T_EEF_GENERATOR, TCP_M, save, sha


NAMES = {
    '008_pudding_box': 'Pudding box', '007_tuna_fish_can': 'Tuna can',
    '025_mug': 'Mug', '036_wood_block': 'Wood block',
    '043_phillips_screwdriver': 'Phillips screwdriver',
}


def load_json(path):
    return json.loads(path.read_text())


def object_colors(points, cache):
    state = load_json(cache/'state.json')
    camera = np.asarray(state['camera_world_optical'])
    xyz = (points-camera[:3, 3]) @ camera[:3, :3]
    params = load_json(cache/'camera/params.json')
    k = params['intrinsics']
    u = np.rint(xyz[:, 0]/xyz[:, 2]*k['fx']+k['ppx']).astype(int)
    v = np.rint(xyz[:, 1]/xyz[:, 2]*k['fy']+k['ppy']).astype(int)
    rgb = np.asarray(Image.open(cache/'camera/color.png').convert('RGB'))
    mask = np.asarray(Image.open(cache/'camera/mask.png'))
    depth = np.asarray(Image.open(cache/'camera/depth.png'))*params['depth_scale']
    assert (u >= 0).all() and (u < rgb.shape[1]).all()
    assert (v >= 0).all() and (v < rgb.shape[0]).all()
    assert (mask[v, u] > 0).all(), 'Every displayed point must belong to target'
    error = float(np.max(np.abs(depth[v, u]-xyz[:, 2])))
    assert error < 1.e-5, error
    return rgb[v, u]/255., error


def jaw_lines(descriptor):
    """Open jaw inner-edge guides in the model's canonical generator frame.

    +X closes, +Z approaches. These guides use the measured ARX cavity,
    including its XY offset and fingertip depth; they are not collision meshes.
    """
    p = descriptor['parameters']
    ox, oy, _ = p['offset_open']
    half = p['extents_open'][0]/2
    tip = p['fingertip_depth']
    return np.array([
        [[ox-half, oy, 0.], [ox-half, oy, tip]],
        [[ox+half, oy, 0.], [ox+half, oy, tip]],
        [[ox-half, oy, 0.], [ox+half, oy, 0.]],
        [[ox, oy, -.018], [ox, oy, 0.]],
    ])


def load_case(root, entry, local_lines):
    trial = root/'results/graspgenx'/entry['id']
    raw = trial/'graspgenx_raw'
    points = np.load(raw/'model_input_points_world.npy', allow_pickle=False)
    native = np.load(raw/'grasps_generator_world.npy', allow_pickle=False)
    scores = np.load(raw/'scores.npy', allow_pickle=False).reshape(-1)
    tags = load_json(raw/'inference.json')['branch_tags']
    assert len(native) == len(scores) == len(tags)
    assert np.isfinite(points).all() and np.isfinite(native).all()
    assert np.isfinite(scores).all()
    colors, depth_error = object_colors(points, root/'shared_inputs'/entry['id'])
    pool = load_json(trial/'candidates/candidate_pool.json')['candidates']
    case = {'trial': trial, 'entry': entry, 'points': points, 'colors': colors,
            'title': NAMES[entry['model_id']], 'views': {},
            'point_color_depth_error_m': depth_error}
    for scope in ['raw', 'pipeline']:
        if scope == 'raw':
            chosen = np.argsort(-scores, kind='stable')[:10]
            poses = native[chosen].astype(np.float64)
            selected_scores = scores[chosen]
            selected_tags = [tags[i] for i in chosen]
            original_indices = chosen.tolist()
            retractions = [0.]*len(chosen)
            pool_indices = None
        else:
            chosen = np.argsort(-np.asarray([c['score'] for c in pool]), kind='stable')[:10]
            poses = []
            for index in chosen:
                c = pool[index]
                eef = np.asarray(c['pose'], dtype=np.float64).copy()
                eef[:3, 3] -= TCP_M*eef[:3, 2]
                pose = eef @ T_EEF_GENERATOR
                original = native[c['metadata']['source_generator_index']].copy()
                original[:3, 3] -= c['metadata']['insertion_retraction_m']*original[:3, 2]
                assert np.allclose(pose, original, atol=1.e-8)
                poses.append(pose)
            poses = np.asarray(poses)
            selected_scores = np.asarray([pool[i]['score'] for i in chosen])
            selected_tags = [pool[i]['metadata']['source_branch'] for i in chosen]
            original_indices = [pool[i]['metadata']['source_generator_index'] for i in chosen]
            retractions = [pool[i]['metadata']['insertion_retraction_m'] for i in chosen]
            pool_indices = chosen.tolist()
        assert len(poses) > 0
        norm = Normalize(float(selected_scores.min()), float(selected_scores.max()))
        line_colors = plt.get_cmap('RdYlBu_r')(norm(selected_scores))
        wires = np.asarray([local_lines @ p[:3, :3].T+p[:3, 3] for p in poses])
        case['views'][scope] = {
            'poses': poses, 'scores': selected_scores, 'tags': selected_tags,
            'raw_indices': original_indices, 'retractions': retractions,
            'pool_indices': pool_indices, 'wires': wires, 'line_colors': line_colors,
        }
    return case


def scatter_pose(ax, case, scope, elev=28., azim=-55.):
    view = case['views'][scope]
    center = (case['points'].min(0)+case['points'].max(0))/2
    points = (case['points']-center)*1000
    wires = (view['wires']-center)*1000
    ax.scatter(*points.T, c=case['colors'], s=5.0, depthshade=False,
               linewidths=0, alpha=1., rasterized=True)
    # High-score grasps are drawn last, so their lines remain visible.
    for i in reversed(range(len(wires))):
        for segment in wires[i]:
            ax.plot(*segment.T, color=view['line_colors'][i], lw=1.6, alpha=.94)
    total = np.concatenate([points, wires.reshape(-1, 3)])
    lo, hi = total.min(0), total.max(0)
    mid = (lo+hi)/2
    half = np.max(hi-lo)*.56
    for setter, value in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
        setter(value-half, value+half)
    ax.set_box_aspect((1, 1, 1), zoom=1.18)
    ax.set_proj_type('ortho')
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_facecolor('#fafbfc')


def score_legend(container, rect, case, scope, font_size=10):
    ax = container.add_axes(rect)
    ax.axis('off')
    v = case['views'][scope]
    handles = [Line2D([0], [0], color=c, lw=2.8) for c in v['line_colors']]
    labels = [f'#{i+1:02d}  {s:.5f}  {tag.upper()}'
              for i, (s, tag) in enumerate(zip(v['scores'], v['tags']))]
    ax.legend(handles, labels, title='Native model score\nhigh (warm) → low (cool)',
              loc='upper left', frameon=False, fontsize=font_size,
              title_fontsize=font_size, handlelength=1.7, labelspacing=.65)


def individual(case, scope, output):
    fig = plt.figure(figsize=(13, 7.5), facecolor='white')
    left = fig.add_axes([.00, .08, .52, .79], projection='3d')
    right = fig.add_axes([.49, .10, .33, .74], projection='3d')
    scatter_pose(left, case, scope)
    scatter_pose(right, case, scope, elev=90., azim=-90.)
    left.set_title('Oblique view', fontsize=12, pad=-6)
    right.set_title('Top view', fontsize=12, pad=-6)
    score_legend(fig, [.81, .22, .19, .58], case, scope)
    stage = 'RAW model top 10' if scope == 'raw' else 'Pipeline top 10 (before cuRobo)'
    fig.text(.035, .96, f"GraspGenX | {case['title']} | {stage}",
             fontsize=19, weight='bold', va='top')
    fig.text(.035, .906, case['entry']['id'], fontsize=10, color='#576476')
    fig.text(.035, .045,
             f"Exact visible object input: {len(case['points']):,} points | no table, no completion | "
             'open ARX jaw guides: 88 mm', fontsize=10, color='#344152')
    fig.savefig(output, dpi=170, facecolor='white')
    plt.close(fig)


def overview(cases, scope, output):
    fig = plt.figure(figsize=(16, 17), facecolor='white')
    subfigs = fig.subfigures(3, 2, wspace=.015, hspace=.015).ravel()
    for panel, case in zip(subfigs, cases):
        ax = panel.add_axes([.00, .03, .76, .84], projection='3d')
        scatter_pose(ax, case, scope)
        panel.text(.055, .935, case['title'], fontsize=18, weight='bold')
        panel.text(.055, .884, f"{len(case['points']):,} observed object points", fontsize=11,
                   color='#576476')
        score_legend(panel, [.73, .21, .27, .60], case, scope, font_size=10)
    note = subfigs[-1]
    stage = 'Original model outputs' if scope == 'raw' else 'Candidates used by the benchmark'
    note.text(.08, .88, 'GraspGenX · object clouds', fontsize=23, weight='bold', va='top')
    note.text(.08, .76, stage+'\nTop 10 per target, native model score\nRed = higher; blue = lower',
              fontsize=17, linespacing=1.8, va='top')
    note.text(.08, .44, 'Same 5 paired-test scenes\nExact saved partial input clouds\nNo table / background / hidden surface\nOpen ARX jaw guides: 88 mm\nPose validity is evaluated separately',
              fontsize=14, linespacing=1.75, color='#344152', va='top')
    fig.savefig(output, dpi=160, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    root = parser.parse_args().output.resolve()
    registration = load_json(root/'registration.json')
    descriptor_path = root/'preparation/gripper_descriptor.json'
    descriptor = load_json(descriptor_path)
    lines = jaw_lines(descriptor)
    cases = [load_case(root, entry, lines) for entry in registration['schedule']]
    records = []
    for case in cases:
        out = case['trial']/'gg_outputs'
        source_names = ['model_input_points_world.npy', 'grasps_generator_world.npy', 'scores.npy', 'inference.json']
        hashes = {str(case['trial']/'graspgenx_raw'/name): sha(case['trial']/'graspgenx_raw'/name)
                  for name in source_names}
        hashes[str(descriptor_path)] = sha(descriptor_path)
        hashes[str(case['trial']/'candidates/candidate_pool.json')] = sha(case['trial']/'candidates/candidate_pool.json')
        for scope in ['raw', 'pipeline']:
            prefix = 'grasp_raw_top10_object_cloud' if scope == 'raw' else 'grasp_top10_object_cloud'
            image = out/(prefix+'.png')
            individual(case, scope, image)
            v = case['views'][scope]
            manifest = {'scene': case['entry']['id'], 'scope': scope,
                        'input_points': len(case['points']), 'source_sha256': hashes,
                        'color_depth_reprojection_max_error_m': case['point_color_depth_error_m'],
                        'ranking': 'Stable descending original model score; no additional suppression',
                        'pose_frame': 'World, native generator +X closing and +Z approach',
                        'glyph': 'Calibrated ARX open inner-jaw guides, 88mm; not collision meshes',
                        'drawn_native_generator_poses_world': v['poses'],
                        'model_indices': v['raw_indices'], 'pipeline_indices': v['pool_indices'],
                        'scores': v['scores'], 'branches': v['tags'],
                        'retractions_m': v['retractions'], 'line_colors': [to_hex(c) for c in v['line_colors']],
                        'cloud_or_source_poses_modified': False, 'new_inference_or_physical_execution': False}
            save(out/(prefix+'.json'), manifest)
            assert all(sha(path) == value for path, value in hashes.items())
            records.append({'scene': case['entry']['id'], 'scope': scope, 'image': str(image)})
    review = root/'review/graspgenx_object_clouds'
    review.mkdir(parents=True, exist_ok=True)
    for scope in ['raw', 'pipeline']:
        overview(cases, scope, review/(scope+'_top10_overview.png'))
    save(review/'visualization_manifest.json', {'records': records, 'source_arrays_unchanged': True,
         'new_inference_or_physical_execution': False, 'script_sha256': sha(Path(__file__))})
    print(json.dumps({'review': str(review), 'individual_figures': len(records)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
